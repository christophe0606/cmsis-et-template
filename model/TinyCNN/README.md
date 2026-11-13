# TinyCNN quantized examples

Run these standalone scripts with the repository's Python environment:

```powershell
.venv/Scripts/python.exe model/TinyCNN/quantize_model.py
.venv/Scripts/python.exe model/TinyCNN/quantize_multiple_models.py
```

`quantize_model.py` produces `TinyCNN.pt2` with one INT8 `forward` method.
`quantize_multiple_models.py` produces **`TinyCNNMultiple.pt2`** containing:

| Method | Network | Input | Output | Quantization |
|---|---|---|---|---|
| `filter` | Per-channel 3x3 box filter | NHWC `[1,16,16,3]` | NHWC `[1,16,16,3]` | A16W8: INT16 activations, INT8 weights |
| `classify` | TinyCNN | NHWC `[1,16,16,3]` | `[1,10]` logits | INT8 activations and weights |

Convolution weights use per-channel quantization. The classifier's linear
weights use per-tensor symmetric quantization because the Cortex-M linear
kernel requires a scalar weight zero point. The classifier is calibrated on
the quantized filter's output. Both programs retain floating-point public
inputs and outputs, with quantization boundaries inside each method.

All image dimensions derive from the script's constants. Inference always
uses batch 1. The archive exports just four integer constants through
`constant_methods.json`: `IMAGE_WIDTH`, `IMAGE_HEIGHT`, `INPUT_CHANNELS`,
and `NUM_CLASSES`.
The [metadata convention](../../model_conversion/README.md#constant-metadata-convention-for-pt2-producers)
explains how the exporter writes these and the converter passes them to
`to_edge_transform_and_lower(constant_methods=...)`.

These scripts have no command-line options, perform no CMSIS setup, and do not
call Vela. `create_ai_layer.py` later lowers both networks into **one**
`TinyCNNMultiple.pte` and embeds it as `TinyCNNMultiple_pte.c/.h`.

## Select the multiple-model example

All six solution files contain a commented `TinyCNNMultiple.pt2` input. Replace
the complete active single-model input entry with this entry, keeping just one
active input. Do not retain the single model's `parameters:` on a `.pt2` input.

```yaml
input:
  - name: TinyCNNMultiple
    file: ../model/TinyCNN/TinyCNNMultiple.pt2
```

For **Zephyr** (`zephyr-m`, `zephyr-u55`, `zephyr-u85`), set this in `prj.conf`:

```ini
CONFIG_EXECUTORCH_MULTIPLE_TEST=y
```

The Kconfig choice selects either the existing single-model C++ test or
`src/executorch_multiple_test.cpp`. Both retain
the existing board initialization in `src/main.c`.

For **CMSIS** (`cmsis-m`, `cmsis-u55`, `cmsis-u85`), comment the default
`projects:` entry and uncomment the supplied `*-multiple.cproject.yml` entry.
In `target-set.images`, replace the default project context with the commented
`*-multiple.Debug` alternative. This separate project uses
`src/app_multiple_main.c` and `src/executorch_multiple_test.cpp` instead of the
single-model `app_main.cpp`.

Then run the usual setup, conversion, and build steps for that solution. Keep
the Python ExecuTorch package and selected pack versions matched (these
examples use Python 1.6.0 and pack `1.6.0-zephyrpreview2`). Re-run setup after
editing the solution so its generated `.cbuild-mlops.yml` selects the new input.
The preview pack is supplied in `Pack/`; install it with the CMSIS pack manager
before setup if it is not already installed. The CMSIS CI workflows install
that repository pack before generating their MLOps configuration.

The C++ test reads these four integers through zero-input constant methods.
It constructs the image shape `[1, IMAGE_HEIGHT, IMAGE_WIDTH, INPUT_CHANNELS]`
and classifier output shape `[1, NUM_CLASSES]`, checks them against the runtime
tensor metadata, and allocates its buffers. It fills the image with `0.25`,
executes `filter`, then feeds that result to `classify`. It compares
all ten logits against a hardcoded reference with a small absolute tolerance
for INT8 rounding differences and prints `Test_result: PASS` or `FAIL`.
Each solution has its own `src/executorch_multiple_test.cpp`, which uses
`EmbeddedModule` directly like the single-model `executorch_test.cpp`.
The integer metadata queries, two network calls, and reference comparison
are all in this C++ runner.

To return to the single-model example, restore its input and CMSIS project
selection or set the Zephyr configuration back to `n`, then regenerate the AI
layer. The generator retains files it did not produce in the current run;
obsolete artifacts can be removed explicitly.
