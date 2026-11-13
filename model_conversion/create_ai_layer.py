#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Create a multi-program AI layer for CMSIS or Zephyr.

One AI layer targets exactly one CPU, optionally with Ethos-U acceleration.
Each input file has its own .pte container, .c, and .h files; a .pt2 input
may contain several named programs that become methods in that .pte. The
generated CMSIS layer combines the CPU fallback operators required by all
output programs. model.clayer names the layer file; model.zephyr enables
additional Zephyr integration files in the same directory.

Without --input, this script processes all declared inputs. With --input NAME,
it updates only the selected input and rebuilds the layer-wide metadata from
all current .pte program artifacts.

    python model_conversion/create_ai_layer.py <solution>.cbuild-mlops.yml

Step 2 of the three-step flow:

    cbuild setup <solution>.csolution.yml --active <target>   # writes *.cbuild-mlops.yml
    python model_conversion/create_ai_layer.py <solution>.cbuild-mlops.yml  # this script
    cbuild <solution>.csolution.yml --active <target>         # compile and link

The generated `model.input` sequence declares `.py`, `.pt2`, or `.pte` input
artifacts. The file extension selects the processing workflow.

The script runs itself in the workspace's .venv (see setup_venv.py). Paths in
the MLOps file are resolved from the solution directory containing that file.
"""

from __future__ import annotations

import io
import argparse
import importlib.metadata
import importlib.util
import inspect
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

PACK = "PyTorch::ExecuTorch"
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
# The current CMSIS-NN pack also needs these for integer-only builds because
# some floating-point declarations are missing their preprocessor guards.
CMSIS_NN_DEFINES = ("ARM_NN_ENABLE_F32", "ARM_NN_ENABLE_F16")


@dataclass(frozen=True)
class ModelInput:
    name: str
    stem: str
    source: Path
    parameters: dict
    cpu_only: bool
    optimize: str | None
    imported: bool

    @property
    def input_type(self) -> str:
        return self.source.suffix.lower()


@dataclass(frozen=True)
class PteInspection:
    """Structured information obtained from an ExecuTorch program."""

    methods: tuple[str, ...]
    delegates: tuple[str, ...]
    accelerators: tuple[str, ...]
    system_configs: tuple[str, ...]
    memory_modes: tuple[str, ...]


def run_in_venv(workspace_dir: Path) -> None:
    """Re-run under the workspace's .venv unless already running in it.

    Deciding by "does torch import" is not enough: a torch installed for the
    host interpreter would keep the export outside the environment with the
    pinned executorch. sys.prefix is the venv directory when running inside it
    (comparing interpreter paths does not work: venv symlinks resolve to the
    base interpreter).
    """
    venv = workspace_dir / ".venv"
    if Path(sys.prefix).resolve() == venv.resolve():
        return
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        sys.exit(
            f"{venv} does not exist.\n"
            "Create it first with model_conversion/setup_venv.sh (Linux/macOS) "
            "or model_conversion/setup_venv.bat (Windows)"
        )
    print(f"[ai_layer] running in {python}", flush=True)
    script = Path(__file__).resolve()
    sys.exit(subprocess.run([str(python), str(script), *sys.argv[1:]]).returncode)


def pack_root() -> Path:
    """The CMSIS pack root: $CMSIS_PACK_ROOT, else cpackget's default."""
    if env := os.environ.get("CMSIS_PACK_ROOT"):
        return Path(env)
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "Arm/Packs"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "arm/packs"


def executorch_version(mlops_file: Path) -> str:
    """Read ExecuTorch's version from CMSIS-Toolbox's generated build metadata.

    The pack lock file can retain versions from an old generated AI layer.
    The active .cbuild.yml files record the packs actually used by the build;
    the .cbuild-idx.yml locates them. No source solution or layer is parsed.
    """
    import yaml

    pack_file = mlops_file.with_name(mlops_file.name.replace(".cbuild-mlops.yml", ".cbuild-pack.yml"))
    resolved: dict[str, set[str]] = {}
    for entry in yaml.safe_load(pack_file.read_text())["cbuild-pack"]["resolved-packs"]:
        name, _, version = entry["resolved-pack"].partition("@")
        selectors = entry.get("selected-by-pack", [])
        if name != PACK or not selectors:
            continue
        resolved.setdefault(version, set()).update(selectors)
    if not resolved:
        sys.exit(f"{pack_file}: {PACK} is not among the resolved packs")

    index_file = mlops_file.with_name(mlops_file.name.replace(".cbuild-mlops.yml", ".cbuild-idx.yml"))
    if index_file.is_file():
        index = yaml.safe_load(index_file.read_text())["build-idx"]
        versions: set[str] = set()
        for context in index.get("cbuilds", []):
            build_file = index_file.parent / context["cbuild"]
            if not build_file.is_file():
                sys.exit(f"{build_file}: missing generated build file. Re-run cbuild setup.")
            build = yaml.safe_load(build_file.read_text())["build"]
            for entry in build.get("packs", []):
                name, _, version = entry["pack"].partition("@")
                if name == PACK:
                    versions.add(version)
        if not versions:
            sys.exit(
                f"{index_file}: no active build uses {PACK}. Re-run cbuild setup."
            )
        if not versions.issubset(resolved):
            sys.exit(
                f"{index_file}: active {PACK} version is not resolved in {pack_file}. "
                "Re-run cbuild setup."
            )
    else:
        # Support standalone MLOps/pack artifacts with an unambiguous resolution.
        versions = {
            version for version, selectors in resolved.items()
            if f"{PACK}@{version}" in selectors
        } or set(resolved)
    if len(versions) != 1:
        sys.exit(
            f"{pack_file}: ambiguous {PACK} versions: {', '.join(sorted(versions))}. "
            "Re-run cbuild setup to generate an unambiguous active build selection."
        )
    return next(iter(versions))


def executorch_pack(version: str) -> Path:
    """Directory of the installed PyTorch::ExecuTorch pack of that version."""
    vendor, _, pack = PACK.partition("::")
    return pack_root() / vendor / pack / version


# Vela options that are arguments of EthosUCompileSpec, and the ones the export
# sets itself: the configuration file comes from vela.ini of the cbuild-mlops.yml.
SPEC_OPTIONS = ("accelerator-config", "system-config", "memory-mode")
EXPORT_OPTIONS = ("config", "output-format", "output-dir")


def vela_options(options: str) -> tuple[dict[str, str], list[str]]:
    """Split vela.options of the cbuild-mlops.yml: the EthosUCompileSpec arguments, and every other option.

    The other options are the `misc:` of the csolution's mlops: node (for
    example `--optimise Size`); they reach Vela as extra flags, written as
    `--name=value`.
    """
    tokens = shlex.split(options)
    spec, extra = {}, []
    while tokens:
        token = tokens.pop(0)
        if not token.startswith("--"):
            sys.exit(f"vela options: unexpected '{token}' in '{options}'")
        name, has_value, value = token[2:].partition("=")
        if not has_value and tokens and not tokens[0].startswith("-"):
            value = tokens.pop(0)
        if name in EXPORT_OPTIONS:
            sys.exit(
                f"vela options: --{name} is set by the export; "
                "the configuration file is vela: ini: of the mlops: node"
            )
        if name in SPEC_OPTIONS:
            if name in spec:
                sys.exit(f"vela options: --{name} is given twice in '{options}'")
            spec[name] = value
        else:
            extra.append(f"--{name}={value}" if value else f"--{name}")
    return spec, extra


def compile_spec(
    mlops: dict, mlops_dir: Path, *, cpu_only: bool = False, optimize: str | None = None
):
    """EthosUCompileSpec for an NPU target; None selects Cortex-M generation."""
    if cpu_only or (
        "input" not in mlops["model"]
        and mlops["model"].get("cpu")
    ):
        return None
    npu = mlops.get("npu")
    if not npu:
        return None

    from executorch.backends.arm.ethosu import EthosUCompileSpec
    vela = mlops.get("vela", {})
    spec, extra = vela_options(vela.get("options", ""))
    if optimize:
        extra = [*extra, f"--optimise={optimize}"]

    target = spec.get("accelerator-config") or f"{npu['type'].lower()}-{npu.get('macs', 256)}"
    kwargs = {
        "target": target,
        "system_config": spec.get("system-config"),
        "memory_mode": spec.get("memory-mode"),
        "extra_flags": extra,
    }
    if vela.get("ini"):
        # ExecuTorch stores the path in the compile spec, and the spec ends up
        # in the .pte program. A path relative to the working directory keeps the
        # program identical between checkouts; Vela resolves it from there.
        kwargs["config_ini"] = os.path.relpath(mlops_dir / vela["ini"])
    print(f"[ai_layer] Vela: {kwargs}")
    return EthosUCompileSpec(**kwargs)


def cpu_target(mlops: dict):
    """Create the Cortex-M backend target from processor.type."""
    from executorch.backends.cortex_m.target_config import CortexMTargetConfig

    core = mlops.get("processor", {}).get("type") or mlops["model"].get("cpu")
    if not core:
        sys.exit("the cbuild-mlops.yml names neither an NPU nor a processor")
    target = core.lower().replace("+", "plus")
    try:
        config = CortexMTargetConfig.from_target_string(target)
    except ValueError as error:
        sys.exit(f"no Cortex-M backend target for processor {core!r}: {error}")
    print(f"[ai_layer] Cortex-M: {config.target_string}, CMSIS-NN {config.backend.name} kernels")
    return config


def convert_parameters(parameters: dict, model_name: str) -> dict:
    """Convert YAML model parameters to Python keyword arguments."""
    params = {}
    for key, value in parameters.items():
        if key == "input-shape":
            params["input_shape"] = tuple(int(d) for d in re.split(r"[x,]", str(value)))
        elif key == "calibration-samples":
            params["calibration_samples"] = int(value)
        else:
            params[key.replace("-", "_")] = value
    if params:
        print(f"[ai_layer] {model_name} parameters: {params}")
    return params


def model_params(mlops: dict) -> dict:
    """Compatibility helper for the legacy single-model form."""
    model = mlops["model"]
    return convert_parameters(
        {
            k: v for k, v in model.items()
            if k not in ("clayer", "name", "zephyr", "cpu", "input")
        },
        model.get("name", "Model"),
    )


def model_inputs(mlops: dict, solution_dir: Path) -> list[ModelInput]:
    """Validate and normalize new and legacy model declarations."""
    model = mlops["model"]
    if "inputs" in model:
        sys.exit("model.inputs is unsupported; use the singular model.input property")
    entries = model.get("input")
    legacy = entries is None
    if legacy:
        legacy_name = model.get("name", "Model")
        legacy_source = WORKSPACE_DIR / "model" / legacy_name / "model.py"
        if not legacy_source.is_file():
            legacy_source = WORKSPACE_DIR / "model" / "model.py"
        if not legacy_source.is_file():
            candidates = sorted((WORKSPACE_DIR / "model").glob("*/model.py"))
            if len(candidates) == 1:
                legacy_source = candidates[0]
        entries = [{
            "name": legacy_name,
            "file": str(legacy_source),
            "parameters": {
                k: v for k, v in model.items()
                if k not in ("clayer", "name", "zephyr", "cpu")
            },
            **({"cpu": model.get("cpu")} if "cpu" in model else {}),
        }]
    if not isinstance(entries, list) or not entries:
        sys.exit("model.input must be a non-empty sequence")

    result, names, stems = [], set(), set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            sys.exit(f"model.input[{index}] must be a mapping")
        unknown_properties = sorted(set(entry) - {"name", "file", "parameters", "cpu", "optimize"})
        if unknown_properties:
            sys.exit(f"model.input[{index}]: unsupported properties {unknown_properties}")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            sys.exit(f"model.input[{index}].name is required")
        if name in names:
            sys.exit(f"duplicate model name {name!r}")
        stem = "model" if legacy else name
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", stem):
            sys.exit(f"model {name!r}: name must be a valid C identifier and file stem")
        if stem in stems:
            sys.exit(f"model name collision for identifier {stem!r}")
        file = entry.get("file")
        if not isinstance(file, str) or not file:
            sys.exit(f"model {name!r}: file is required")
        source = Path(file)
        if not source.is_absolute():
            source = solution_dir / source
        source = source.resolve()
        if not source.is_file():
            sys.exit(f"model {name!r}: input does not exist: {source}")
        input_type = source.suffix.lower()
        if input_type not in (".py", ".pt2", ".pte"):
            sys.exit(f"model {name!r}: unsupported input type {source.suffix!r}")
        parameters = entry.get("parameters", {})
        if not isinstance(parameters, dict):
            sys.exit(f"model {name!r}: parameters must be a mapping")
        if parameters and input_type != ".py":
            sys.exit(f"model {name!r}: parameters are valid only for .py input")
        cpu_only = "cpu" in entry or not mlops.get("npu")
        optimize = entry.get("optimize")
        if optimize is not None and input_type == ".pte":
            sys.exit(f"model {name!r}: optimize is invalid for .pte input")
        if optimize is not None and cpu_only:
            sys.exit(f"model {name!r}: optimize is invalid for CPU-only input")
        if isinstance(optimize, str) and optimize.lower() in ("size", "performance"):
            optimize = optimize.capitalize()
        names.add(name)
        stems.add(stem)
        result.append(ModelInput(
            name, stem, source, convert_parameters(parameters, name), cpu_only,
            str(optimize) if optimize is not None else None, input_type == ".pte"
        ))
    return result


def _accepted_parameter_names(function) -> set[str] | None:
    signature = inspect.signature(function)
    accepts_any = any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values())
    if accepts_any:
        return None
    return {
        name for name, p in signature.parameters.items()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }


def _call_kwargs(function, params: dict) -> dict:
    names = _accepted_parameter_names(function)
    return params if names is None else {key: value for key, value in params.items() if key in names}


def _input_case(value, description: str) -> tuple[tuple, dict]:
    """Normalize one explicit ``(args, kwargs)`` input case."""
    if not isinstance(value, tuple) or len(value) != 2:
        sys.exit(f"{description} must return (args, kwargs)")
    args, kwargs = value
    if not isinstance(args, tuple) or not isinstance(kwargs, dict):
        sys.exit(f"{description} must return (tuple(args), dict(kwargs))")
    return args, kwargs


def register_operator_libraries() -> None:
    """Register the Python operator families supplied by the ExecuTorch pack.

    Import before model capture or PT2 deserialization: archives contain
    operator names, not the Python code or native libraries that define them.
    ATen operators come from torch; ExecuTorch adds portable custom, dimension
    order, device copy, quantized, and Cortex-M operators.
    """
    import torch  # noqa: F401
    import executorch.kernels.quantized  # noqa: F401
    import executorch.kernels.portable  # noqa: F401
    import executorch.backends.cortex_m.ops.operators  # noqa: F401
    import executorch.exir.passes.dim_order_ops_registry  # noqa: F401
    import executorch.exir.passes._device_copy_ops_registry  # noqa: F401


def load_model(source: Path, params: dict, model_name: str):
    """Load one Python workflow and return graph, calibration, and example inputs."""
    import torch

    register_operator_libraries()
    module_name = f"ai_layer_model_{re.sub(r'[^A-Za-z0-9_]', '_', model_name)}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        sys.exit(f"model {model_name!r}: cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        get_model = module.get_model
        get_example_inputs = module.get_example_inputs
        get_calibration_inputs = module.get_calibration_inputs
    except AttributeError as error:
        sys.exit(f"model {model_name!r}: {source} must define {error.name}")
    functions = (get_model, get_example_inputs, get_calibration_inputs)
    accepted = [_accepted_parameter_names(function) for function in functions]
    known = (
        set(params) if any(names is None for names in accepted) else
        {name for names in accepted if names is not None for name in names}
    )
    unknown = sorted(set(params) - known)
    if unknown:
        sys.exit(f"model {model_name!r}: unknown parameters {unknown}")
    model = get_model(**_call_kwargs(get_model, params))
    if not isinstance(model, torch.nn.Module):
        sys.exit(f"model {model_name!r}: get_model must return torch.nn.Module")
    model.eval()
    example = _input_case(
        get_example_inputs(**_call_kwargs(get_example_inputs, params)),
        f"model {model_name!r}: get_example_inputs",
    )
    raw_samples = get_calibration_inputs(**_call_kwargs(get_calibration_inputs, params))
    if not isinstance(raw_samples, (list, tuple)) or not raw_samples:
        sys.exit(f"model {model_name!r}: get_calibration_inputs must return a non-empty sequence")
    samples = [
        _input_case(sample, f"model {model_name!r}: calibration input {index}")
        for index, sample in enumerate(raw_samples)
    ]
    args, kwargs = example
    graph = torch.export.export(model, args, kwargs=kwargs).module()
    strip_guards_fn(graph)
    return graph, samples, example


def strip_guards_fn(graph_module) -> None:
    """Remove the dead guard module emitted by recent Torch export versions."""
    changed = False
    for node in list(graph_module.graph.nodes):
        if node.op == "call_module" and str(node.target) == "_guards_fn":
            graph_module.graph.erase_node(node)
            changed = True
    if changed:
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()


def quantize(graph, quantizer, samples):
    """Annotate, calibrate with every sample, and convert a PT2E graph."""
    import torch
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    prepared = prepare_pt2e(graph, quantizer)
    with torch.no_grad():
        for args, kwargs in samples:
            prepared(*args, **kwargs)
    return convert_pt2e(prepared)


def export_model(spec, source: Path, params: dict, model_name: str) -> bytes:
    """Quantize a Python model and return its serialized ExecuTorch program."""
    import torch
    from executorch.backends.arm.ethosu import EthosUPartitioner
    from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower

    graph, samples, example = load_model(source, params, model_name)
    example_args, example_kwargs = example

    # Quantize the whole graph so the partitioner can move every node into the
    # Ethos-U delegate; only a float<->int8 boundary stays on the CPU.
    quantizer = EthosUQuantizer(spec)
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    quantized = quantize(graph, quantizer, samples)

    edge = to_edge_transform_and_lower(
        torch.export.export(quantized, example_args, kwargs=example_kwargs),
        partitioner=[EthosUPartitioner(spec)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    program = edge.to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False))
    return bytes(program.buffer)


def export_model_cpu(target, source: Path, params: dict, model_name: str) -> bytes:
    """Quantize and lower a Python model for Cortex-M with CMSIS-NN kernels."""
    import torch
    from executorch.backends.cortex_m.edge_compile_config import cortex_m_edge_compile_config
    from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
    from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
    from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower

    graph, samples, example = load_model(source, params, model_name)
    example_args, example_kwargs = example
    quantized = quantize(graph, CortexMQuantizer(use_explicit_layout=True), samples)
    exported = torch.export.export(quantized, example_args, kwargs=example_kwargs)
    strip_guards_fn(exported.graph_module)
    edge = to_edge_transform_and_lower(exported, compile_config=cortex_m_edge_compile_config())
    edge._edge_programs["forward"] = CortexMPassManager(
        edge.exported_program(), target_config=target, use_explicit_layout=True
    ).transform()
    program = edge.to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False))
    return bytes(program.buffer)


def load_pt2_programs(source: Path, input_name: str) -> dict:
    """Load and validate every named ExportedProgram in a PT2 archive."""
    return load_pt2_artifact(source, input_name)[0]


def load_pt2_artifact(source: Path, input_name: str) -> tuple[dict, dict]:
    """Load quantized networks and optional JSON-encoded constant methods."""
    register_operator_libraries()
    from torch.export.pt2_archive._package import load_pt2

    try:
        archive = load_pt2(source)
        programs = dict(archive.exported_programs)
    except Exception as error:
        # Some wheels omit this optional AOT library and its import only logs
        # the failure. Explain the missing package artifact when it is needed.
        if re.search(r"aten(?:::|\.)allclose\.(?:Tensor|out)\b", str(error)):
            import torch
            if not all(hasattr(torch.ops.aten.allclose, name) for name in ("Tensor", "out")):
                sys.exit(
                    f"input {input_name!r}: the PT2 archive needs the portable custom "
                    "allclose operators, but the installed ExecuTorch wheel did not "
                    "register aten::allclose.Tensor and aten::allclose.out. "
                    "Install an ExecuTorch build containing portable_custom_ops_aot_lib; "
                    "the runtime pack alone does not provide Python AOT registration."
                )
        sys.exit(f"input {input_name!r}: cannot load PT2 archive: {error}")
    if not programs:
        sys.exit(f"input {input_name!r}: PT2 archive contains no exported programs")
    for name, program in programs.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            sys.exit(f"input {input_name!r}: invalid program name {name!r}")
        targets = [str(node.target) for node in program.graph_module.graph.nodes]
        if not any(
            "quantized_decomposed" in target
            or "quantize_per_" in target
            or target.startswith("cortex_m.")
            for target in targets
        ):
            sys.exit(f"input {input_name!r}: program {name!r} is not quantized")
    try:
        constants = json.loads(archive.extra_files.get("constant_methods.json", "{}"))
    except (ValueError, TypeError) as error:
        sys.exit(f"input {input_name!r}: invalid constant_methods.json: {error}")
    if not isinstance(constants, dict):
        sys.exit(f"input {input_name!r}: constant_methods.json must contain a mapping")
    for name, value in constants.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name in programs:
            sys.exit(f"input {input_name!r}: invalid or duplicate constant method name {name!r}")
        # JSON scalars and homogeneous lists are supported by ExecuTorch's
        # constant-method emitter. Reject objects and nested lists explicitly.
        scalars = value if isinstance(value, list) else [value]
        if not scalars or type(scalars[0]) not in (bool, int, float, str) or any(
            type(item) is not type(scalars[0]) for item in scalars
        ):
            sys.exit(f"input {input_name!r}: unsupported constant method value for {name!r}")
    return programs, constants


def lower_pt2(spec, target, source: Path, input_name: str) -> tuple[bytes, tuple[str, ...]]:
    """Lower all programs in one quantized PT2 archive into one PTE container."""
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower

    programs, constants = load_pt2_artifact(source, input_name)
    if spec is not None:
        from executorch.backends.arm.ethosu import EthosUPartitioner

        edge = to_edge_transform_and_lower(
            programs,
            constant_methods=constants,
            partitioner=[EthosUPartitioner(spec)],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
    else:
        from executorch.backends.cortex_m.edge_compile_config import cortex_m_edge_compile_config
        from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager

        edge = to_edge_transform_and_lower(
            programs, constant_methods=constants, compile_config=cortex_m_edge_compile_config()
        )
        for method in edge.methods:
            edge._edge_programs[method] = CortexMPassManager(
                edge.exported_program(method), target_config=target, use_explicit_layout=True
            ).transform()
    runtime_program = edge.to_executorch(
        ExecutorchBackendConfig(extract_delegate_segments=False)
    )
    return bytes(runtime_program.buffer), tuple(programs)


class Tee(io.TextIOBase):
    """Copy text to the terminal while retaining it for a model report."""

    def __init__(self, terminal, capture: io.StringIO):
        self.terminal = terminal
        self.capture = capture

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.capture.write(text)
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()


def compiler_flag_value(flags: list[str], option: str) -> str:
    """Return an effective Vela option from EthosUCompileSpec.compiler_flags."""
    prefix = f"--{option}="
    return next((flag[len(prefix) :] for flag in flags if flag.startswith(prefix)), "not specified")


def input_report(
    model: ModelInput,
    operators: list[str],
    cmsis_nn: bool,
    spec=None,
    vela_report: str = "",
    inspection: PteInspection | None = None,
) -> str:
    """Create the reproducible per-input processing report."""
    try:
        source = model.source.relative_to(WORKSPACE_DIR).as_posix()
    except ValueError:
        source = model.source.as_posix()
    lines = [
        f"# Input report: {model.name}", "",
        f"- input: `{source}`",
        f"- input type: `{model.input_type}`",
        f"- execution: `{'CPU only' if model.cpu_only else 'CPU with Ethos-U acceleration'}`",
        f"- parameters: `{json.dumps(model.parameters, sort_keys=True, default=list)}`",
        f"- optimization: `{model.optimize or 'default'}`",
        f"- CMSIS-NN required: `{'yes' if cmsis_nn else 'no'}`",
        f"- methods: `{', '.join(inspection.methods) if inspection else 'unavailable'}`", "",
        "## CPU fallback operators", "",
        *([f"- `{operator}`" for operator in operators] or ["None."]), "",
    ]
    if spec is not None:
        report = vela_report.replace("\r\n", "\n").replace("\r", "\n").strip()
        lines.extend([
            "## Vela", "",
            f"- accelerator: `{spec.target}`",
            f"- system config: `{compiler_flag_value(spec.compiler_flags, 'system-config')}`",
            f"- memory mode: `{compiler_flag_value(spec.compiler_flags, 'memory-mode')}`", "",
            "```text", report, "```", "",
        ])
    return "\n".join(lines)


def vela_summary(model_name: str, spec, report: str) -> str:
    """Compatibility helper for the former standalone Vela report."""
    report = report.replace("\r\n", "\n").replace("\r", "\n").strip()
    return (
        f"# Vela report: {model_name}\n\n"
        f"- accelerator: `{spec.target}`\n"
        f"- system config: `{compiler_flag_value(spec.compiler_flags, 'system-config')}`\n"
        f"- memory mode: `{compiler_flag_value(spec.compiler_flags, 'memory-mode')}`\n\n"
        f"```text\n{report}\n```\n"
    )


def replace_outputs(layer_dir: Path, outputs: dict[str, bytes | str]) -> None:
    """Atomically replace the generated outputs named by this run.

    Files not named by the current run are deliberately left untouched. This
    avoids persistent generator bookkeeping and never guesses whether an
    existing file is generator-owned or user-owned.
    """
    layer_dir.mkdir(parents=True, exist_ok=True)
    # tempfile.TemporaryDirectory deliberately creates a private directory on
    # Windows. Files moved out of it keep that restrictive ACL and cannot be
    # read by build tools running as the interactive user. A normal child
    # directory inherits the destination ACL while still allowing all outputs
    # to be staged before any destination is replaced.
    stage = layer_dir / f".generate-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        for name, contents in outputs.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(contents, bytes):
                path.write_bytes(contents)
            elif isinstance(contents, str):
                path.write_text(contents, encoding="utf-8", newline="\n")
            else:
                raise TypeError(f"unsupported generated output type for {name}")

        for name in outputs:
            destination = layer_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / name, destination)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


FAMILY = {
    "aten": "Portable",
    "dim_order_ops": "Portable",
    "et_copy": "Portable",
    "quantized_decomposed": "Quantized",
    "cortex_m": "Cortex-M",
}
ALIASES = {
    "_native_batch_norm_legit": "native_batch_norm",
    "_native_batch_norm_legit_no_training": "native_batch_norm",
    "copy_": "copy",
    "index_put_": "index_put",
    "_h2d_copy": "device_copy",
    "_d2h_copy": "device_copy",
    "choose_qparams_per_token_asymmetric": "choose_qparams",
    "quantize_per_token": "quantize",
    "dequantize_per_token": "dequantize",
    "embedding_2bit": "embedding2b",
    "embedding_4bit": "embedding4b",
    "quantized_depthwise_conv2d": "quantized_dw_conv2d",
    "quantized_transpose_conv2d": "quantized_tr_conv2d",
}


def pack_components(pack: Path) -> tuple[set[str], set[str]]:
    """Return operator components and the subset whose conditions require CMSIS-NN."""
    root = ET.parse(next(pack.glob("*.pdsc"))).getroot()
    requires = {
        condition.get("id"): [
            requirement.get("condition")
            for requirement in condition
            if requirement.tag == "require" and requirement.get("condition")
        ]
        for condition in root.iter("condition")
    }

    def needs_cmsis_nn(condition: str | None) -> bool:
        return condition == "CMSIS-NN" or any(
            needs_cmsis_nn(requirement) for requirement in requires.get(condition, [])
        )

    operators = {
        name: component.get("condition")
        for component in root.iter("component")
        if component.get("Cgroup") == "ExecuTorch Operators"
        and (name := component.get("Csub")) is not None
    }
    return set(operators), {
        name for name, condition in operators.items() if needs_cmsis_nn(condition)
    }


def operator_components(operator_names: list[str], pack: Path) -> tuple[list[str], bool]:
    """Map ExecuTorch operator names to pack components."""
    available, cmsis_nn = pack_components(pack)
    selected, unknown = set(), []
    for operator in sorted(set(operator_names)):
        ns, separator, op = operator.partition("::")
        if not separator or ns not in FAMILY:
            unknown.append(operator)
            continue
        op = op.split(".", 1)[0]
        exact = f"{FAMILY[ns]} {op}"
        op = ALIASES.get(op, op)
        candidates = [
            exact,
            f"{FAMILY[ns]} {op}",
            f"{FAMILY[ns]} {op.lstrip('_')}",
            f"{FAMILY[ns]} {re.sub(r'_(per_tensor|per_channel|byte|copy|nhwc)$', '', op)}",
        ]
        if match := next((c for c in candidates if c in available), None):
            selected.add(match)
        else:
            unknown.append(operator)
    if unknown:
        raise ValueError(f"no component in {pack.name} for operators {unknown}")
    return sorted(selected), bool(selected & cmsis_nn)


def components(pte: bytes, pack: Path, *, delegated: bool) -> tuple[list[str], list[str], bool]:
    """Legacy helper retained for callers; generation uses operator YAML."""
    namespaces = "|".join(FAMILY)
    names = [
        f"{ns.decode()}::{op.decode()}"
        for ns, op in sorted(set(re.findall(rf"({namespaces})::(\w+)".encode(), pte)))
    ]
    selected, needs_nn = operator_components(names, pack)

    runtime = ["Runtime", "Kernel Utils", "Kernel Registration"]
    if delegated:
        runtime.append("Backend EthosU")
    return runtime, selected, needs_nn


def c_array(pte: bytes, symbol: str = "model_pte") -> str:
    rows = [", ".join(f"0x{b:02x}" for b in pte[i : i + 16]) for i in range(0, len(pte), 16)]
    return (
        "// Generated by create_ai_layer.py -- do not edit.\n"
        f"__attribute__((aligned(16))) const unsigned char {symbol}[] = {{\n  "
        + ",\n  ".join(rows)
        + f"\n}};\nconst unsigned long {symbol}_size = sizeof({symbol});\n"
    )


def header(symbol: str = "model_pte") -> str:
    return f"""// Generated by create_ai_layer.py -- do not edit.
#pragma once
#ifdef __cplusplus
extern "C" {{
#endif
extern const unsigned char {symbol}[];
extern const unsigned long {symbol}_size;
#ifdef __cplusplus
}}
#endif
"""


HEADER = header()


def clayer(
    mlops: dict,
    runtime: list[str],
    operators: list[str],
    cmsis_nn: bool,
    mlops_file: Path,
    version: str,
    models: list[ModelInput] | None = None,
) -> str:
    try:
        mlops_path = os.path.relpath(mlops_file, WORKSPACE_DIR)
    except ValueError:
        # Windows cannot construct a relative path across drive letters. This
        # path is only used in the regeneration hint, so keep it absolute.
        mlops_path = mlops_file
    lines = [
        f"# Generated by create_ai_layer.py from {mlops_file.name} -- do not edit.",
        f"# Re-run `python model_conversion/create_ai_layer.py {mlops_path}` after changing",
        "# model inputs or the csolution's mlops: node.",
        "layer:",
        "  type: AI",
        f"  description: {mlops.get('description', mlops['model'].get('name', 'AI layer'))}",
        "",
        "  packs:",
        f"    - pack: {PACK}@{version}",
        *(["    - pack: ARM::CMSIS-NN"] if cmsis_nn else []),
        "",
        "  define:",
        "    - ET_LOG_ENABLED: 0",
        *[f"    - {define}: 1" for define in CMSIS_NN_DEFINES if cmsis_nn],
        "",
        "  add-path:",
        "    - .",
        "",
        "  components:",
        *[f"    - component: Machine Learning:ExecuTorch:{c}" for c in runtime],
        *[f"    - component: Machine Learning:ExecuTorch Operators:{c}" for c in operators],
        *(["    - component: CMSIS:NN Lib"] if cmsis_nn else []),
        "",
        "  groups:",
    ]
    if models is None:
        models = [ModelInput(mlops['model'].get('name', 'Model'), 'model', Path('model.py'), {}, False, None, False)]
    for model in models:
        lines.extend([
            f"    - group: {model.name}",
            "      files:",
            f"        - file: ./{model.stem}_pte.c",
            f"        - file: ./{model.stem}_pte.h",
        ])
    lines.append("")
    return "\n".join(lines)


def zephyr_cmake(models: list[ModelInput], *, cmsis_nn: bool = False) -> str:
    sources = "\n".join(f'  "${{CMAKE_CURRENT_SOURCE_DIR}}/{m.stem}_pte.c"' for m in models)
    defines = (
        "zephyr_compile_definitions(" + " ".join(f"{define}=1" for define in CMSIS_NN_DEFINES) + ")\n"
        if cmsis_nn else ""
    )
    return f'''# Generated by create_ai_layer.py -- do not edit.
# Include this directory after find_package(Zephyr) creates the app target.
target_sources(app PRIVATE
{sources}
)
target_include_directories(app PRIVATE "${{CMAKE_CURRENT_SOURCE_DIR}}")
{defines}'''


ZEPHYR_CMAKE = zephyr_cmake([ModelInput("Model", "model", Path("model.py"), {}, False, None, False)])

ZEPHYR_COMPONENTS = '''# Generated by create_ai_layer.py -- do not edit.
CONFIG_CMSIS_EXECUTORCH=y

CONFIG_CMSIS_EXECUTORCH_RUNTIME=y
CONFIG_CMSIS_EXECUTORCH_KERNEL_UTILS=y
CONFIG_CMSIS_EXECUTORCH_KERNEL_REGISTRATION=y
'''


def zephyr_metadata(
    operators: list[str], pack: Path, delegated: bool, models: list[ModelInput] | None = None,
    cmsis_nn: bool = False,
) -> dict[str, str]:
    """Generate the Zephyr module files for the selected operator components."""
    symbols = [re.sub(r"[^A-Z0-9]+", "_", operator.upper()).strip("_") for operator in operators]
    kconfig = (pack / "zephyr" / "Kconfig").read_text()
    available = set(re.findall(r"(?m)^config CMSIS_EXECUTORCH_OP_([A-Z0-9_]+)\s*$", kconfig))
    missing = sorted(set(symbols) - available)
    if missing:
        raise ValueError(f"pack Kconfig options missing for {missing}")
    operator_config = "# Generated by create_ai_layer.py -- do not edit.\n" + "".join(
        f"CONFIG_CMSIS_EXECUTORCH_OP_{symbol}=y\n" for symbol in sorted(set(symbols))
    )
    components = ZEPHYR_COMPONENTS
    if delegated:
        components += "CONFIG_CMSIS_EXECUTORCH_BACKEND_ETHOS_U=y\n"
    return {
        "CMakeLists.txt": zephyr_cmake(
            models or [ModelInput("Model", "model", Path("model.py"), {}, False, None, False)],
            cmsis_nn=cmsis_nn,
        ),
        "components.conf": components,
        "operators.conf": operator_config,
    }


def _operator_names(path: Path) -> list[str]:
    import yaml

    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return sorted((content.get("operators") or {}).keys())


def generate_operator_lists(pte_paths: dict[str, Path], stage: Path) -> tuple[dict[str, list[str]], list[str]]:
    """Extract per-input operator lists and combine them in temporary storage."""
    from executorch.codegen.tools.gen_oplist import gen_oplist
    from executorch.codegen.tools import gen_all_oplist

    operator_root = stage / "operators"
    paths, per_input = [], {}
    for name, pte_path in pte_paths.items():
        output = operator_root / name / "selected_operators.yaml"
        output.parent.mkdir(parents=True, exist_ok=True)
        gen_oplist(output_path=output, model_file_path=str(pte_path))
        paths.append(output)
        per_input[name] = _operator_names(output)
    model_list = operator_root / "models.txt"
    model_list.write_text("\n".join(str(path) for path in paths) + "\n", encoding="utf-8")
    combined_dir = operator_root / "all-models"
    combined_dir.mkdir(parents=True)
    gen_all_oplist.main([
        "--model_file_list_path", f"@{model_list}",
        "--output_dir", str(combined_dir),
    ])
    return per_input, _operator_names(combined_dir / "selected_operators.yaml")


def check_executorch_version(expected: str, *, recovery_hint: str = "") -> None:
    """Require the Python exporter to match the runtime pack's base version.

    CMSIS pack versions may add a pack-specific prerelease suffix, for example
    ``1.6.0-zephyrpreview2``.  The corresponding Python distribution remains
    ``1.6.0``.
    """
    try:
        installed = importlib.metadata.version("executorch")
    except importlib.metadata.PackageNotFoundError:
        sys.exit("the executorch Python package is not installed")
    python_version = expected.split("-", 1)[0]
    if installed != python_version:
        sys.exit(
            f"ExecuTorch Python package {installed} does not match "
            f"{PACK}@{expected}{recovery_hint}"
        )


def expected_accelerator(mlops: dict) -> str | None:
    npu = mlops.get("npu")
    if not npu:
        return None
    spec, _ = vela_options(mlops.get("vela", {}).get("options", ""))
    return spec.get("accelerator-config") or f"{npu['type'].lower()}-{npu.get('macs', 256)}"


def expected_ethosu_options(mlops: dict) -> dict[str, str]:
    """Return explicitly configured target properties that a prebuilt PTE must match."""
    if not mlops.get("npu"):
        return {}
    spec, _ = vela_options(mlops.get("vela", {}).get("options", ""))
    return {
        key: spec[key]
        for key in ("system-config", "memory-mode")
        if key in spec
    }


def inspect_pte(pte: bytes) -> PteInspection:
    """Deserialize a PTE and return methods and backend compile specifications."""
    from executorch.exir._serialize._program import deserialize_pte_binary
    from executorch.backends.arm.ethosu import EthosUCompileSpec

    try:
        program = deserialize_pte_binary(pte).program
    except Exception as error:
        raise ValueError(f"invalid ExecuTorch program: {error}") from error
    methods, delegates = [], []
    accelerators, system_configs, memory_modes = set(), set(), set()
    for plan in program.execution_plan:
        methods.append(plan.name)
        for delegate in plan.delegates:
            delegates.append(delegate.id)
            if delegate.id == "EthosUBackend":
                try:
                    spec = EthosUCompileSpec._from_list(delegate.compile_specs)
                except Exception as error:
                    raise ValueError(f"invalid Ethos-U compile specification: {error}") from error
                accelerators.add(spec.target)
                system_configs.add(compiler_flag_value(spec.compiler_flags, "system-config"))
                memory_modes.add(compiler_flag_value(spec.compiler_flags, "memory-mode"))
    if not methods:
        raise ValueError("ExecuTorch program contains no methods")
    return PteInspection(
        tuple(methods), tuple(sorted(set(delegates))), tuple(sorted(accelerators)),
        tuple(sorted(system_configs)), tuple(sorted(memory_modes))
    )


def validate_pte_target(
    model: ModelInput,
    pte: bytes,
    accelerator: str | None,
    expected_options: dict[str, str] | None = None,
) -> PteInspection:
    """Structurally validate a PTE against its declared CPU/Ethos-U execution."""
    try:
        inspection = inspect_pte(pte)
    except ValueError as error:
        sys.exit(f"input {model.name!r}: {error}")
    unknown = sorted(set(inspection.delegates) - {"EthosUBackend"})
    if unknown:
        sys.exit(f"input {model.name!r}: unsupported backend delegates {unknown}")
    delegated = "EthosUBackend" in inspection.delegates
    if model.cpu_only and delegated:
        sys.exit(f"input {model.name!r}: CPU-only input contains an Ethos-U delegate")
    if not model.cpu_only and not delegated:
        sys.exit(f"input {model.name!r}: NPU input contains no Ethos-U delegate")
    if delegated and accelerator and inspection.accelerators != (accelerator,):
        sys.exit(
            f"input {model.name!r}: program targets {list(inspection.accelerators)}, "
            f"not {accelerator}"
        )
    expected_options = expected_options or {}
    checks = {
        "system-config": inspection.system_configs,
        "memory-mode": inspection.memory_modes,
    }
    for option, actual in checks.items():
        expected = expected_options.get(option)
        if delegated and expected and actual != (expected,):
            sys.exit(
                f"input {model.name!r}: program {option} is {list(actual)}, not {expected}"
            )
    return inspection


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a target-specific AI layer for CMSIS or Zephyr from the "
            "model inputs declared in the MLOPS_FILE."
        ),
        epilog="""The input file extension in the MLOPS_FILE selects the workflow:
  .pt2           Backend-lower and serialize as a .pte program
  .pte           Validate and integrate an existing backend-lowered program
  .py            Export, quantize, backend-lower, and serialize the model

examples:
  python model_conversion/create_ai_layer.py app.cbuild-mlops.yml
  python model_conversion/create_ai_layer.py app.cbuild-mlops.yml --input TinyCNN""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mlops_file",
        metavar="MLOPS_FILE",
        help="Path to the generated *.cbuild-mlops.yml file from CMSIS-Toolbox",
    )
    parser.add_argument(
        "--input",
        metavar="NAME",
        help="Process only the named input; by default, process all inputs",
    )
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        parser.exit(2, f"\n{parser.prog}: error: MLOPS_FILE is required\n")
    args = parser.parse_args()
    if not args.mlops_file.endswith(".cbuild-mlops.yml"):
        parser.error("mlops_file must end with .cbuild-mlops.yml")

    # CMSIS-Toolbox paths are relative to the generated MLOps file. The Python
    # environment is workspace-wide, while model sources and generated files
    # belong to the selected solution.
    mlops_file = Path(args.mlops_file).resolve()
    solution_dir = mlops_file.parent
    run_in_venv(WORKSPACE_DIR)

    import yaml

    mlops = yaml.safe_load(mlops_file.read_text())["cbuild-mlops"]
    model = mlops["model"]
    if not isinstance(model.get("clayer"), str) or not model["clayer"].strip():
        sys.exit("model.clayer must be a non-empty file path")
    layer_file = solution_dir / model["clayer"]
    layer_dir = layer_file.parent
    if "zephyr" in model:
        # YAML loads an empty ``zephyr:`` node as None. Treat it as the empty
        # mapping so the concise form and ``zephyr: {}`` have identical meaning.
        if model["zephyr"] is None:
            model["zephyr"] = {}
        elif not isinstance(model["zephyr"], dict):
            sys.exit("model.zephyr must be empty or a mapping, for example: zephyr:")
    recovery_hint = ""
    if layer_file.is_file():
        recovery_hint = (
            "\n\nAn old generated AI clayer may still pin the previous ExecuTorch pack. "
            f"Removing the generated clayer '{layer_file}' may resolve this issue. "
            "Then rerun cbuild setup for your target and rerun create_ai_layer.py "
            "to regenerate the AI layer."
        )
    try:
        version = executorch_version(mlops_file)
    except SystemExit as error:
        if recovery_hint:
            sys.exit(f"{error}{recovery_hint}")
        raise
    check_executorch_version(version, recovery_hint=recovery_hint)
    pack = executorch_pack(version)
    models = model_inputs(mlops, solution_dir)
    accelerator = expected_accelerator(mlops)
    expected_options = expected_ethosu_options(mlops)
    selected = models
    if args.input:
        selected = [entry for entry in models if entry.name == args.input]
        if not selected:
            sys.exit(f"unknown input {args.input!r}")

    outputs: dict[str, bytes | str] = {}
    generated_pte: dict[str, bytes] = {}
    specs, vela_reports, inspections = {}, {}, {}
    target = None
    for entry in selected:
        spec = None if entry.input_type == ".pte" else compile_spec(
            mlops, solution_dir, cpu_only=entry.cpu_only, optimize=entry.optimize
        )
        specs[entry.name] = spec
        if entry.input_type == ".pte":
            pte = entry.source.read_bytes()
            vela_reports[entry.name] = ""
        elif entry.input_type == ".pt2":
            if spec is None:
                target = target or cpu_target(mlops)
            capture = io.StringIO()
            with redirect_stdout(Tee(sys.stdout, capture)):
                pte, _ = lower_pt2(spec, target, entry.source, entry.name)
            vela_reports[entry.name] = capture.getvalue() if spec is not None else ""
        elif spec is None:
            target = target or cpu_target(mlops)
            pte = export_model_cpu(target, entry.source, entry.parameters, entry.name)
            vela_reports[entry.name] = ""
        else:
            capture = io.StringIO()
            with redirect_stdout(Tee(sys.stdout, capture)):
                pte = export_model(spec, entry.source, entry.parameters, entry.name)
            vela_reports[entry.name] = capture.getvalue()
        inspections[entry.name] = validate_pte_target(
            entry, pte, accelerator, expected_options
        )
        generated_pte[entry.name] = pte
        symbol = f"{entry.stem}_pte"
        outputs[f"{entry.stem}.pte"] = pte
        outputs[f"{entry.stem}_pte.c"] = c_array(pte, symbol)
        outputs[f"{entry.stem}_pte.h"] = header(symbol)

    layer_dir.mkdir(parents=True, exist_ok=True)
    operator_stage = layer_dir / f".operators-{uuid.uuid4().hex}"
    operator_stage.mkdir()
    try:
        pte_paths = {}
        for entry in models:
            if entry.name in generated_pte:
                path = operator_stage / "pte" / f"{entry.stem}.pte"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(generated_pte[entry.name])
            else:
                path = layer_dir / f"{entry.stem}.pte"
                if not path.is_file():
                    sys.exit(f"model {entry.name!r}: existing artifact is missing: {path}")
                validate_pte_target(entry, path.read_bytes(), accelerator, expected_options)
            pte_paths[entry.name] = path
        per_input_ops, combined_ops = generate_operator_lists(pte_paths, operator_stage)
    finally:
        shutil.rmtree(operator_stage, ignore_errors=True)

    operators, cmsis_nn = operator_components(combined_ops, pack)
    delegated = any(not entry.cpu_only for entry in models)
    runtime = ["Runtime", "Kernel Utils", "Kernel Registration"]
    if delegated:
        runtime.append("Backend EthosU")
    for entry in selected:
        _, input_cmsis_nn = operator_components(per_input_ops[entry.name], pack)
        outputs[f"{entry.stem}_REPORT.md"] = input_report(
            entry, per_input_ops[entry.name], input_cmsis_nn,
            specs[entry.name], vela_reports[entry.name], inspections[entry.name]
        )
    outputs[layer_file.name] = clayer(
        mlops, runtime, operators, cmsis_nn, mlops_file, version, models
    )
    if "zephyr" in model:
        outputs.update(zephyr_metadata(operators, pack, delegated, models, cmsis_nn=cmsis_nn))
    replace_outputs(layer_dir, outputs)

    print(f"[ai_layer] inputs: {[entry.name for entry in selected]}, operators: {operators}")
    print(f"[ai_layer] wrote {layer_file}")
    if "zephyr" in model:
        print(f"[ai_layer] wrote {layer_dir / 'CMakeLists.txt'}")


if __name__ == "__main__":
    main()
