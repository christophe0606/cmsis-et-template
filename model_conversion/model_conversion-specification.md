# ML Model to ExecuTorch Program Conversion Toolset Specification

## Purpose and directory contract

`model_conversion/` is the complete host-side toolset for processing one or
more `.pt2`, `.pte`, or `.py` input artifacts into deployment-ready ExecuTorch
programs for Arm Cortex-M, optionally with Ethos-U acceleration. It generates
one AI layer for integration into CMSIS- or Zephyr-based projects. Each AI layer
targets exactly one CPU, optionally with Ethos-U acceleration. An input artifact
is a container boundary, not necessarily a single model: it may hold one model
or several named models or programs. Each input entry owns one serialized and
embedded output `.pte` container, which may contain one or more named methods.
The layer also contains the shared runtime and backend components, the union of
CPU fallback-operator requirements, and the optional CMSIS-NN requirements
needed to execute all programs on that CPU.

The directory above `model_conversion/` is the **workspace directory**. It
owns the shared `.venv`. The directory containing a selected
`*.cbuild-mlops.yml` is the **solution directory**. It owns the input artifacts,
solution-specific configuration, pack metadata, and generated AI layer. A
workspace may contain multiple solution directories.

## Deliverables

### `README.md`

The README is the user entry point. It shall give the setup, processing, and
test commands; explain the workspace/solution distinction; and link to this
specification for the complete directory contract.

### `create_ai_layer.py`

This is the input processor and AI-layer generator. Invoke it with exactly one
MLOps file and an optional input selection:

```console
python model_conversion/create_ai_layer.py <solution-dir>/<solution>.cbuild-mlops.yml
python model_conversion/create_ai_layer.py <solution-dir>/<solution>.cbuild-mlops.yml --input <name>
```

It shall locate `.venv` in the workspace directory and restart itself with
that environment's interpreter unless it is already running there. It shall
derive the solution directory from the MLOps file rather than from the current
working directory.

#### Interface terminology

- An **input artifact** is a file supplied through `model.input`. It may
  represent one model or a container of named models or programs. Supported
  extensions are `.pt2`, `.pte`, and `.py`.
- An **output program artifact** is a deployment-ready `.pte` produced or
  accepted by the generator. It is backend-lowered for, and validated against,
  the selected target. It may contain portable CPU fallback operators.
- The **AI layer** comprises the output programs, embedded C arrays,
  dependency metadata, reports, a `*.clayer.yml`, and optional Zephyr files.
  It targets exactly one CPU, optionally with Ethos-U acceleration; all output
  programs shall be compatible with that CPU and, when used, its shared Ethos-U
  configuration.

No one-input/one-model assumption is made. Every declared input entry owns
exactly one output `.pte` container. A `.pt2` archive may contain several named
`ExportedProgram`s representing separate models or methods; they shall become
named methods in that input's output `.pte`. Separate `.pt2` inputs shall not
be combined implicitly into one `.pte`.

The supported workflows are:

| Input | Role | Generator action |
| --- | --- | --- |
| `.pt2` | Recommended lowering input | Validate, backend-lower, serialize as `.pte`, and integrate. |
| `.pte` | Optional prebuilt deployment artifact | Inspect, validate, and integrate unchanged. |
| `.py` | Convenience model-production input | Export, quantize, backend-lower, serialize, and integrate. |

#### MLOps contract

The YAML root shall contain `cbuild-mlops`. The converter uses these
layer-wide properties:

- `processor.type`: required Cortex-M type when `npu` is absent;
- `npu.type`: optional Ethos-U type; its absence selects CPU-only generation;
- `npu.macs`: optional MAC count, defaulting to 256;
- `vela.ini`: optional path relative to the solution directory;
- `vela.options`: generated command-line options;
- `model.clayer`: required output file path relative to the solution directory
  for both CMSIS and Zephyr;
- `model.zephyr`: optional empty node or mapping enabling additional Zephyr
  integration output in the directory containing `model.clayer`;
- `model.input`: generated sequence of input definitions described below.

`model.clayer` shall be present and contain a non-empty string file path. The
converter shall reject missing or invalid paths before model conversion.
Zephyr configurations shall also contain `model.zephyr`.

The authored `mlops.model.input` sequence shall be emitted by CMSIS-Toolbox as
`cbuild-mlops.model.input`. Each entry uses this contract:

```yaml
cbuild-mlops:
  model:
    clayer: ai_layer/ai_layer.clayer.yml
    input:
      - name: TinyCNN
        file: ../model/TinyCNN/model.py
        optimize: size
        parameters:
          input-shape: 1x16x16x3
          calibration-samples: 2
      - name: Vision
        file: ../models/vision.pt2
      - name: CpuModel
        file: ../models/cpu-model.pt2
        cpu:
      - name: ImportedModel
        file: ../models/imported.pte
```

- `name` is required and unique. After validation and sanitization it supplies
  the layer group, output-artifact stem, and C identifier. Duplicate names,
  invalid identifiers, and name-to-identifier collisions are errors.
- `file` is required, is resolved relative to the solution directory, and must
  name a `.pt2`, `.pte`, or `.py` input artifact. The extension selects the
  workflow; no `format` or `process` property is used.
- `parameters` is optional and valid only for `.py`. It is passed to the Python
  workflow functions. Unknown parameters shall fail with an input-qualified
  diagnostic instead of being ignored.
- Presence of `cpu`, including an empty `cpu:` value, selects CPU lowering for
  `.py` and `.pt2`. For `.pte`, it declares the expected CPU-only execution
  mode and shall be cross-checked against the inspected program. Key presence,
  not YAML value truthiness, determines this behavior.
- Without `cpu`, `.py` and `.pt2` use the solution's NPU when present; otherwise
  they use the processor. A `.pte` without `cpu` on an NPU solution is expected
  to contain the matching NPU backend. On a solution without an NPU, every
  input is necessarily CPU-only.
- `optimize` is an optional per-input Vela optimization setting for `.py` and
  `.pt2`. It supplements the layer-wide `vela.options`. It is invalid for a
  CPU-resolved input and for `.pte`, which is never lowered again.

Unsupported extensions and properties that do not apply to the selected input
type are errors.

#### `.pt2` input

A `.pt2` is the boundary between model-specific preparation and target-specific
deployment. It shall contain one or more quantized `ExportedProgram`s and their
input signatures, with no target-specific backend lowering. These programs may
represent distinct models or multiple methods of one model.

The producer owns model definition, named programs, example and calibration
inputs, PTQ or QAT configuration, quantization, and post-quantization accuracy
validation. The producer shall save the resulting quantized program or programs
in the `.pt2` archive.

The producer may include UTF-8 `constant_methods.json` via `package_pt2`'s
`extra_files` argument. Its top-level object maps unique C-identifier method
names to JSON scalars or non-empty homogeneous lists of scalars (Boolean,
integer, float, string). Names shall not collide with network methods. Invalid
JSON, other top-level types, nested objects/lists, mixed lists, and `null`
values shall be rejected. Other extra files shall not be interpreted.

The generator shall pass this dictionary as `constant_methods` to
`to_edge_transform_and_lower` for either backend. These zero-input methods
shall survive serialization unchanged and shall not be quantized or run through
the network lowering passes. A missing metadata file shall mean an empty
constant-method dictionary. Native runtime list results are flattened into
individual primitive outputs.

The TinyCNN multiple-model example shall export `filter` with A16W8
quantization and `classify` with INT8 quantization, plus four positive integer
constants: `IMAGE_WIDTH`, `IMAGE_HEIGHT`, `INPUT_CHANNELS`, and `NUM_CLASSES`.
Batch shall always be 1 for inference. The firmware shall derive image shape
`[1, IMAGE_HEIGHT, IMAGE_WIDTH, INPUT_CHANNELS]` and classifier output shape
`[1, NUM_CLASSES]`, check them against actual tensor metadata, allocate buffers
from their element counts, execute filter then classify, and
compare every final logit with an independent hardcoded reference.

The generator shall:

- load the archive and validate every named program, signature, quantization,
  and compatibility with the selected backend;
- reject unquantized, unsupported, or incompatible programs;
- never quantize or requantize the `.pt2` input;
- partition and lower every program for the selected target, including Vela for
  Ethos-U and the effective per-input `optimize` setting;
- preserve program names and signatures as methods in the output `.pte`; and
- validate the serialized `.pte` before integrating it.

#### `.pte` input

A `.pte` is an optional prebuilt deployment artifact. The generator shall never
run Vela or another backend-lowering pass on it. It shall inspect every method
and accept the file only when structured inspection proves that it is already
backend-lowered and compatible with the selected target and declared execution
mode.

For Cortex-M, validation shall cover the core, backend-specific operators, and
required portable fallback operators. For Ethos-U, it shall cover the
accelerator, system configuration, memory mode, backend delegate, and required
CPU fallback operators. Portable-only programs, unknown backends, mismatched
`cpu` declarations, and programs whose compatibility cannot be established are
errors. Raw byte-string searches are not structured inspection.

If the program does not expose enough information, the generator may use
trusted producer metadata that identifies the backend, target configuration,
ExecuTorch version, required operators, and program hash. Without reliable
inspection or trusted metadata, `.pte` input support shall reject the artifact;
the user shall provide the corresponding quantized `.pt2` instead.

#### `.py` input

A `.py` input is the convenience workflow for simple models and examples. It
shall provide:

- `get_model`, returning an evaluation-mode `torch.nn.Module`;
- `get_example_inputs`, returning `(args, kwargs)` for `torch.export`; and
- `get_calibration_inputs`, returning calibration cases in the same
  `(args, kwargs)` form.

Example inputs define positional and keyword arguments, tensor shapes, and
dtypes. They shall not be inferred from the first calibration sample.
Calibration inputs are representative samples used only for PTQ. The workflow
shall support multiple positional and keyword inputs. Declared `parameters`
shall be passed only to functions that accept them; unknown parameters are
errors.

The generator shall export the module, apply the built-in quantization flow,
calibrate it with every returned case, lower it for the selected backend and
target, serialize it as `.pte`, validate it, and integrate it. CPU lowering
shall use the configured Cortex-M backend and applicable CMSIS-NN operators.
Ethos-U lowering shall delegate supported operations and retain boundary or
unsupported operations as CPU fallbacks.

Custom quantization schemes, QAT, specialized calibration, or model-specific
accuracy validation shall use a producer-owned `.pt2` workflow instead.

The previous scalar `model` form, which describes one Python model-production
input, shall remain accepted for one release cycle and shall normalize
internally to one `.py` input definition. This compatibility form does not
constrain the contents of `.pt2` or `.pte` containers.

The accelerator configuration, system configuration, and memory mode are
layer-wide target properties. All NPU-resolved inputs in one AI layer shall use
the same effective values. Per-input system configurations or memory modes are
not supported.

Every Ethos-U/Vela invocation shall capture the complete textual report while
still showing it in the terminal. Its content shall be included in that input's
`<name>_REPORT.md`, not written as a layer-wide `VELA_SUMMARY.md`.

`--accelerator-config`, `--system-config`, and `--memory-mode` shall initialize
`EthosUCompileSpec`. Other Vela options shall be forwarded as extra flags.
`--config`, `--output-format`, and `--output-dir` are owned by the converter and
shall be rejected. Duplicate compile-spec options and unexpected positional
tokens shall also be rejected.

The matching `<solution>.cbuild-pack.yml` shall be read from the solution
directory. Its selected `PyTorch::ExecuTorch` version determines the pack used
to map the serialized program's CPU operators to CMSIS components.
The generated `<solution>.cbuild-idx.yml` shall locate the active `.cbuild.yml`
files. Their `build.packs` entries shall determine the ExecuTorch version used,
which must also be present in the pack lock file. Older versions retained in
the lock file shall not override the active build selection. The converter
shall not parse source `.csolution.yml` or existing `.clayer.yml` files for
version selection. Missing build files, absent active ExecuTorch packs, or
inconsistent lock-file resolutions shall require setup to be rerun. Without a
build index, prefer a unique exact lock-file resolution, otherwise a unique
remaining resolution. Multiple remaining versions shall fail with an ambiguity
diagnostic.
For output with an existing generated clayer, pack-resolution errors and
Python/runtime version mismatches shall explain that the old clayer may pin a
previous ExecuTorch pack. The diagnostic shall identify the generated clayer
and suggest removing it, rerunning setup for the target, and rerunning the
converter. The converter shall not remove the clayer automatically. A missing
Python package shall not receive this clayer-specific hint.

`model.zephyr` is a custom property under `mlops.model`; it shall be preserved
by CMSIS-Toolbox in the generated `*.cbuild-mlops.yml`. It shall not be placed
directly under `mlops`. A plain empty `zephyr:` node is equivalent to an empty
mapping. A non-empty value shall be a mapping so that future Zephyr-specific
parameters can be added without changing the node type.

The output directory shall be the parent of `model.clayer` for both CMSIS and
Zephyr. It shall be created as needed. For every input it shall receive:

```text
<name>.pte
<name>_pte.c
<name>_pte.h
<name>_REPORT.md
```

`<name>_pte.c/.h` shall export unique `<name>_pte` and `<name>_pte_size`
symbols. The C array shall be aligned to 16 bytes.

`<name>_REPORT.md` shall be a concise, reproducible summary of how the input was
processed and validated for the target CPU/MCU and, when applicable, NPU. It
shall identify the input type and provenance, distinguish generated and
prebuilt `.pte` programs, record applicable parameters and effective
optimization settings, list every method and its CPU fallback operators, and
state whether CMSIS-NN is required. An
Ethos-U report shall additionally contain the shared accelerator, system
configuration, and memory mode; the effective per-input Vela options; and the
complete Vela textual report in a fenced code block. A CPU-only report shall
have no Vela section. Reports shall use UTF-8 and LF line endings.

The directory shall always receive the named `*.clayer.yml` file, including
for Zephyr so other tools can consume the layer. It shall contain one
group per input with that input's C and header files, one shared ExecuTorch
runtime component set, every backend needed by at least one output program, and
the union of CPU fallback-operator components across all output `.pte` files.
Repeated components shall be de-duplicated and sorted. Select Backend EthosU
once if any output program is delegated. Add `ARM::CMSIS-NN` without a version,
`CMSIS:NN Lib` when any selected CPU fallback operator requires CMSIS-NN.
Emit both `ARM_NN_ENABLE_F16: 1` and `ARM_NN_ENABLE_F32: 1` whenever CMSIS-NN
is required, including integer-only operators. This works around missing
preprocessor guards in the current CMSIS-NN pack. Emit neither define when
CMSIS-NN is not required.

With `model.zephyr`, the directory shall additionally receive `CMakeLists.txt`,
`components.conf`, and `operators.conf` alongside the `*.clayer.yml`.
Both integration formats shall use the same validated in-memory layer,
runtime, backend, and operator data. Zephyr generation shall not inspect `.pte`
program files or independently calculate an operator union. The CMake file
shall add every embedded-program C file directly and exactly once to Zephyr's
`app` target for every compiler. The configuration fragments shall enable the
same runtime, backends, and fallback-operator components as the CMSIS layer.
The generated CMake file shall apply the same floating-point CMSIS-NN defines
with `zephyr_compile_definitions`, so they reach both ExecuTorch and CMSIS-NN.

The consuming Zephyr application's root CMake file shall append the generated
configuration fragments to `EXTRA_CONF_FILE` before `find_package(Zephyr)` and
shall add the generated layer directory afterward. The generated layer shall
not defer source ownership or expose output programs through interface sources.
CPU-only configuration fragments shall not enable the Ethos-U backend.

### Fallback-operator extraction and union

The current output `.pte` artifact of every declared input is the
authoritative input for fallback-operator selection. `<name>_REPORT.md` is
generated output and shall never be used as operator-selection input.

For each `.pte` program, the converter shall call ExecuTorch's `gen_oplist` in
staging to produce a temporary, per-input `selected_operators.yaml`. The
per-input list shall supply the fallback operators recorded in that input's
report. The converter shall pass all per-input lists to `gen_all_oplist`, which
produces a temporary combined `selected_operators.yaml`. It shall map the
combined YAML operator keys to components in the resolved pack PDSC and write
the resulting union to the selected CMSIS or Zephyr integration metadata.

The converter shall not search raw `.pte` program bytes or use kernel-definition
`merge_yaml` as a multi-program union. An operator with no matching pack
component and a selected Zephyr operator without a matching Kconfig option are
errors. Temporary operator YAML files shall not be installed in the AI layer.

The installed ExecuTorch Python package version shall match the resolved
`PyTorch::ExecuTorch` pack version. A mismatch shall fail before model export or
input processing or program inspection.

Without `--input`, the generator shall process every declared input and rebuild
all layer-wide metadata. With `--input <name>`, it shall process only the named
input. It shall use the newly staged `.pte` program for that input and the
existing `.pte` program artifacts for all unchanged inputs, run `gen_oplist`
on every one, and rebuild the selected input's report, the operator union,
and the selected CMSIS or Zephyr metadata. Reports and other per-input
artifacts belonging to unchanged inputs shall be retained. An unknown input
name is an error. Partial generation shall fail if another declared `.pte`
program is missing or cannot be inspected.

The generator shall replace the files produced by the current run but shall not
delete other files from the output directory. Obsolete artifacts from removed
inputs are left for explicit removal. The generated MLOps file shall never be
modified.

### Validation boundaries

Validation has separate inputs and stages:

- **Example inputs** define the positional arguments, keyword arguments, tensor
  shapes, and dtypes used by `torch.export`. They are required by `.py` and are
  distinct from calibration data.
- **Calibration inputs** are representative PTQ samples. They are supplied by
  the `.py` workflow or owned by the producer of a `.pt2` input.
- **Validation inputs** are held-out cases used to assess quantized-model
  accuracy and shall not be reused as calibration data. Model-specific
  validation belongs to the `.pt2` producer or `.py` workflow.
- **Artifact validation** checks every generated or supplied output `.pte` for
  its methods, backend lowering, operators, ExecuTorch compatibility, and
  selected-target compatibility before replacing the current AI layer.
- **Application validation** uses the SDS Framework to replay deterministic
  model-boundary inputs and record raw model outputs and post-processed results
  on FVP and hardware. SDS metadata shall describe the method, shape, dtype,
  layout, preprocessing, expected result, and comparison tolerance.

Artifact validation is part of `create_ai_layer`. Application validation is a
separate CI or target-test stage and shall not be conflated with export,
calibration, or artifact inspection.

### `setup_venv.py`

This is the platform-independent environment setup implementation. It shall
create or repair `<workspace>/.venv`, never a solution-local environment. It
shall support Python 3.10 through 3.13 and reject unsupported versions before
dependency installation.

`--recreate` shall rebuild the environment. `--uv` shall use uv for environment
creation and package installation. `--python VERSION` shall select the uv
Python version and shall require `--uv`.

The setup shall install `requirements.txt` first and then install
`requirements-arm-tosa.txt` with `--no-deps`. It shall finish with an import
test of the ExecuTorch Arm quantizer, `executorch.codegen.tools.gen_oplist`, and
`executorch.codegen.tools.gen_all_oplist`.

### `setup_venv.sh`

This is the Linux and macOS entry point. It shall locate `setup_venv.py`
relative to itself, forward all arguments, use `${PYTHON:-python3}` for the
standard setup, and use an isolated uv launcher when `--uv` is requested.

### `setup_venv.bat`

This is the Windows entry point. It shall locate `setup_venv.py` relative to
itself, forward all arguments, use `%PYTHON%` when provided and `python`
otherwise, and use an isolated uv launcher when `--uv` is requested.

### `requirements.txt`

This is the primary dependency manifest for model conversion. Versions shall
remain compatible with the ExecuTorch Arm backend and the supported Python
range. It is the source of truth for the coordinated ExecuTorch, PyTorch,
TorchAO, and Vela versions. The ExecuTorch version shall match the version of
the `PyTorch::ExecuTorch` pack that consumes the generated `.pte` program files.
`gen_oplist` and `gen_all_oplist` come from ExecuTorch, and TorchGen comes from
PyTorch; they require no separate dependency entries.

### `requirements-arm-tosa.txt`

This is the separately installed TOSA serializer manifest. It shall contain
the serializer and required FlatBuffers pin used by the ExecuTorch Arm
backend. It is deliberately installed with `--no-deps` so package metadata
cannot replace the dependency versions selected by the primary environment.

### `model_conversion-specification.md`

This document is the normative directory and behavior specification. Changes
to any deliverable in `model_conversion/` shall be reflected here.

### `test/test_create_ai_layer.py`

This is the unit-test suite for the converter. It shall cover workspace `.venv`
selection, solution-relative path handling, MLOps parsing, input selection,
extension-based dispatch, and property applicability. It shall cover:

- `.py` example and calibration inputs, including multiple positional and
  keyword arguments, unknown parameters, CPU/NPU resolution, and lowering;
- `.pt2` archive validation, multiple named programs in one output `.pte`,
  rejection of unquantized input, and preservation of method names and
  signatures;
- `.pte` structured inspection, declared execution-mode and target validation,
  rejection of portable or unverifiable programs, and proof that Vela is never
  invoked for this input type;
- `cpu` and `optimize` applicability and effective Vela options;
- package/pack version matching, per-input and combined operator-list
  generation, component selection, reports, partial updates, and CMSIS and
  Zephyr artifacts.

Unit tests shall mock export, PT2 loading, backend lowering, Vela, structured
PTE inspection, `gen_oplist`, and `gen_all_oplist.main` so they do not require
an FVP or target hardware. Integration tests shall exercise PT2-to-PTE lowering,
PTE inspection, and operator-list generation with the pinned ExecuTorch
package.

Run it from the workspace directory:

```console
python -m unittest discover -s model_conversion/test -p "test_*.py" -v
```

## Acceptance criteria

- Invocation from any current directory locates the workspace `.venv` and the
  selected solution's files correctly.
- Multiple solutions in workspace subdirectories share `.venv` while keeping
  their input artifacts, configuration, metadata, and output independent.
- The `.pt2`, `.pte`, and `.py` extensions select their specified workflows;
  unsupported extensions and inapplicable properties fail before output is
  replaced.
- An input file is not limited to one model. Each input file has its own
  backend-lowered, target-validated output `.pte` container. Multiple named
  models or programs in one `.pt2` become methods in that output without implicit
  bundling across separate inputs.
- Without `--input`, every declared input is processed. With `--input <name>`,
  only that input is processed while the layer-wide operator union and metadata
  continue to include every declared input.
- A `.pt2` input is never quantized or requantized by the generator; all named
  programs are lowered with the selected target configuration and effective
  `optimize` setting.
- A `.pte` input is never passed through Vela or another lowering pass. It is
  accepted unchanged only when structured inspection or trusted producer
  metadata proves backend and target compatibility.
- A `.py` input obtains example inputs independently from calibration inputs and
  supports multiple positional and keyword arguments.
- Two output programs with overlapping CPU fallback operators produce one copy of each
  runtime and operator component in CMSIS and Zephyr metadata.
- A CPU-only output and an Ethos-U output can share one layer; it selects their
  combined CPU/CMSIS-NN requirements, the Ethos-U backend once, and both program
  arrays without symbol collisions.
- All NPU-resolved inputs use the same effective accelerator configuration, system
  configuration, and memory mode.
- Every input produces an input-qualified report that records provenance,
  methods, applicable parameters and optimization, fallback operators,
  validation, and applicable Vela output.
- Unknown input names, duplicate names, identifier collisions, missing inputs,
  unsupported file types, and incompatible or unverifiable input artifacts fail
  before replacing the complete layer.
- Both `--name value` and `--name=value` Vela option forms work.
- Invalid and duplicate options terminate with a clear error.
- The resolved pack version follows the generated build's pack selection even
  when the lock file retains an older version. Source solution/layer files are
  not parsed. Missing, inconsistent or ambiguous generated metadata and
  mismatches with the installed ExecuTorch package are rejected.
- `.py` parameters are converted to the types expected by each accepting
  workflow function; parameters on `.pt2` or `.pte` are rejected.
- Generated C, header, report, layer, and Zephyr text are deterministic for
  identical inputs and tool versions.
- Both Ethos-U and CPU-only Cortex-M generation select their target-specific
  runtime and operator components.
- `model.clayer` is required for both CMSIS and Zephyr. Zephyr also uses
  `model.zephyr` to generate complete integration metadata alongside the named
  `*.clayer.yml`. Without `model.zephyr`, no Zephyr metadata is generated.
  Both formats use the same validated layer data and do not recalculate the
  operator union.
- Generation never removes untracked or user-owned files.
- The complete unit-test suite passes without target hardware.

