# Zephyr Cortex-M with Ethos-U55/U65

This Zephyr template deploys ML models on Cortex-M targets with Ethos-U55/U65
acceleration using ExecuTorch.

The supplied configurations use a 256-MAC Ethos-U55 on the Alif Ensemble E8
AI/ML AppKit or the Corstone-300 FVP.

> [!NOTE]
> The Alif E8 hardware target is included temporarily for development and
> validation. It will be removed from the final template.

> [!IMPORTANT]
> This Zephyr template does not use the `ARM::CMSIS-Ethos-U` pack. The Ethos-U
> driver is part of upstream Zephyr 4.5 and uses Zephyr's `hal_ethos_u` module.
> `PyTorch::ExecuTorch` supplies the ExecuTorch runtime and Ethos-U backend.

The solution uses the Arm GNU Toolchain (`compiler: GCC`) by default. Its
CMake and Kconfig configuration also carries the AC6 6.24 compatibility needed
by the `PyTorch::ExecuTorch` preview pack when the solution compiler is changed to AC6.

## Quick Start

### 1. Prepare Zephyr 4.5

Create or select a Zephyr 4.5 workspace and fetch the required modules:

```sh
west init zephyr-workspace --mr v4.5.0
cd zephyr-workspace
west config manifest.project-filter -- +zephyr,+cmsis,+cmsis_6,+hal_ethos_u
west update
west zephyr-export
```

### 2. Select a target-set

- `Corstone-300-FVP` for simulation.
- `AppKit-E8` for an Alif Ensemble E8 AI/ML AppKit using its HP Cortex-M55 and
  256-MAC Ethos-U55.

Enable a command-line selection with `--active`, for example:

```sh
cbuild zephyr-u55/zephyr-u55.csolution.yml --active Corstone-300-FVP
```

The solution describes the E8's Ethos-U55-256 and HP-core Vela memory
configuration. `fvp_config.txt` selects the same 256-MAC NPU and enables fast
functional simulation; reported NPU timing is therefore not cycle-accurate.

### 3. Generate, build, and run

After selecting a target-set, run solution setup and the workspace **Create AI
layer** task so the model is compiled for Ethos-U55-256. Then run **CMSIS:
Build**.

Start the generated Arm FVP configuration for `Corstone-300-FVP`. For
`AppKit-E8`, connect the board's supported debug probe and start the generated
pyOCD configuration. The application prints the detected NPU configuration,
performs inference, and ends with `Test_result: PASS` when validation succeeds.
FVP builds then write EOT (`0x04`) to UART0; together with
`shutdown_on_eot=1` in `fvp_config.txt`, this terminates the simulation without
waiting for `--simlimit`. AppKit builds do not emit the control character.

### Configure with menuconfig

Build the desired target once with the CMSIS Solution extension. From the
repository root, open a terminal using the same tools environment and run:

```powershell
cmake --build zephyr-u55/out/Zephyr_app/Corstone-300-FVP/Debug --target menuconfig
```

Use `--target guiconfig` for the graphical interface. For the physical board,
replace `Corstone-300-FVP` with `AppKit-E8` after building that target once.
Each build directory has its own configuration.

In menuconfig, press `/` to search for an option such as `CMSIS_EXECUTORCH`,
press `?` for its dependencies, and press `Q`, then `Y`, to save. Changes are
written to the build directory's `zephyr/.config`. Copy settings that must
survive a clean build into `prj.conf`; files under `ai_layer/*.conf` are
generated and may be overwritten.

### Alif E8 validation output

The application produces the following output when it runs successfully on the
Alif Ensemble E8 AI/ML AppKit:

```text
Hello World! ensemble_e8_ak/ae822fa0e5597bs0/rtss_hp
Ethos-U55 driver communication check
Device: ethosu55@400e1000, base: 0x400e1000, IRQ: 55
Driver version: 2.0.0
Hardware architecture: 1.1.0
Hardware revision: 2.0 (status 1)
Product revision: 4
Command stream version: 0
Custom DMA: 0
MACs per cycle: 256
Ethos-U55 driver communication OK (ID/configuration read; no inference).
ExecuTorch Ethos-U55-256 application: 9136 byte program
Input shape: [1, 16, 16, 3]
I: Optimizer release nbr: 0 patch: 1
I: Optimizer config. product=0, cmd_stream_version=0, macs_per_cc=8, shram_size=48, custom_dma=0
I: Optimizer config. arch version: 1.0.6
I: Ethos-U config. product=0, cmd_stream_version=0, macs_per_cc=8, shram_size=48, custom_dma=0
I: Ethos-U. arch version=1.1.0
I: handle_command_stream: cmd_stream=0x80231480, cms_length 351 words (1404 bytes)
I: Soft reset ethos-u55
Output: 10 element(s): -0.0142 0.0206 0.0601 -0.0886 0.0791 0.0522 -0.0301 -0.0775 -0.2800 -0.0095
Test_result: PASS
```

## Configuring the Ethos-U target

Follow the
[CMSIS-Ethos-U integration guide](https://arm-software.github.io/CMSIS-Ethos-U/latest/integration/index.html)
when adapting the hardware configuration. Its consistency principles apply to
Zephyr even though this template does not use the `ARM::CMSIS-Ethos-U` pack.

The U55-256 target is represented by the solution's `mlops:` settings,
`CONFIG_ETHOS_U55_256=y` in `prj.conf`, the enabled `ethosu0` devicetree node,
and `ethosu.num_macs=256` in `fvp_config.txt`. The guide's
[256-MAC FVP example](https://arm-software.github.io/CMSIS-Ethos-U/latest/integration/index.html#configure-256-macs-for-fvp-simulation)
explains why these settings and the Vela accelerator selection must agree.

After any NPU change, regenerate the AI layer and rebuild the application.

## How it works

The CMSIS solution does not build the application directly. `cbuild` translates
the selected solution context into a `west build` invocation. The selected
target type, build type, compiler, packs, and `west-defs` become Zephyr board,
CMake, and Kconfig arguments. Zephyr then combines those arguments with the
application files described below.

### Solution-to-west configuration

[`zephyr-u55.csolution.yml`](zephyr-u55.csolution.yml) supplies these values to
west:

| Solution setting | Purpose |
|---|---|
| `west-board` | Selects `mps3/corstone300/fvp` for the FVP or `ensemble_e8_ak/ae822fa0e5597bs0/rtss_hp` for the AppKit. The project-level `board: $west-board$` passes the selected value to west. |
| `DTC_OVERLAY_FILE` | Selects the target-specific devicetree overlay: `fvp.overlay` or `app.overlay`. |
| `EXTRA_CONF_FILE: app.conf` | Adds the AppKit-only Kconfig fragment. It is intentionally absent from the FVP target. |
| `CMAKE_CXX_FLAGS: -includestddef.h` | Applies the current ExecuTorch header compatibility workaround to the FVP build. The single argument keeps the header attached to `-include` when CMSIS-Toolbox invokes CMake. |
| Debug `CONFIG_*` definitions | Enable Zephyr debug information, thread information, and debug optimization for the `Debug` build type. |
| Release `CONFIG_SPEED_OPTIMIZATIONS` | Selects Zephyr's speed optimization for the `Release` build type. |
| `EXTRA_ZEPHYR_MODULES: ${CMSIS_PACK_ROOT}/PyTorch/ExecuTorch/1.6.0-zephyrpreview2` | Makes the installed ExecuTorch pack root an external Zephyr module. This exposes the pack's Kconfig symbols, CMake sources, runtime, operators, and Ethos-U backend. |

The `mlops:` node is consumed before the application build. Solution setup
generates `zephyr-u55.cbuild-mlops.yml`; `create_ai_layer.py` reads that file,
exports and lowers the model for Ethos-U55-256, and writes `ai_layer/`. Changing
the model, accelerator, MAC count, Vela system, or Vela memory mode therefore
requires regenerating the AI layer before rebuilding.

### Configuration files

| File | Scope and function |
|---|---|
| [`prj.conf`](prj.conf) | Common persistent Kconfig configuration for both targets. It enables the Zephyr Ethos-U55-256 driver, C++17, hard-float ABI, cache maintenance, logging, heaps, stack size, and floating-point `printf`. |
| [`app.conf`](app.conf) | AppKit-only Kconfig configuration, selected through `EXTRA_CONF_FILE`. `CONFIG_FLASH_LOAD_OFFSET=0x200000` places the application after the platform firmware in MRAM. Applying this fragment to Corstone-300 would exceed its 512-KiB ITCM flash region and produce AC6 linker error `L6225E`. |
| [`app.overlay`](app.overlay) | AppKit devicetree changes. It declares and enables the Ethos-U55, combines the two SRAM banks into an 8-MiB NPU-accessible region, and selects that region as `zephyr,sram`; the NPU cannot DMA to the board's default Cortex-M55 DTCM. |
| [`fvp.overlay`](fvp.overlay) | Corstone-300 devicetree changes. It keeps the board's existing Ethos-U device and selects ISRAM as writable application memory because the simulated NPU also cannot DMA to DTCM. |
| [`fvp_config.txt`](fvp_config.txt) | Parameters for the FVP executable, not Zephyr. It selects 256 MACs, enables fast functional simulation, sends unbuffered UART0 output directly to the console, shuts down on EOT, and disables unused terminals and visualization. CI captures the same console stream with `tee`. |
| `ai_layer/components.conf` | Generated Kconfig fragment enabling only the ExecuTorch runtime, support libraries, kernel registration, and Ethos-U backend required by the generated layer. |
| `ai_layer/operators.conf` | Generated Kconfig fragment enabling only the operators detected in the lowered model. |
| `ai_layer/CMakeLists.txt` | Generated CMake fragment that attaches `TinyCNN_pte.c` and its include directory directly to Zephyr's `app` target. Direct ownership prevents the model from being compiled twice during AC6 pre-link. |
| `ai_layer/VELA_SUMMARY.md` | Generated record of the effective Vela configuration and Vela's textual compilation report. |

The two generated `ai_layer/*.conf` files must be appended to
`EXTRA_CONF_FILE` before `find_package(Zephyr)`, because Zephyr resolves Kconfig
while that call runs. Afterward, `add_subdirectory(ai_layer)` attaches the
generated model source to the `app` target. This ordering in
[`CMakeLists.txt`](CMakeLists.txt) is significant.

### AC6 compatibility configuration

The application CMake file applies the following adjustments only when CMake
detects ArmClang:

- Compile `op_quantize.cpp` with `-O2` for AC6 6.24 hard-float builds to avoid
  its register-coalescer failure at lower optimization levels.
- Put AC6's libc++ wrapper headers before its C resource headers and force the
  application compatibility header into C++ translation units.
- Add the constructor section, constructor runner, and destructor support that
  are otherwise omitted by Zephyr's generated AC6 scatter configuration.
- Remove Zephyr's `--library_type=microlib` selection so ExecuTorch links with
  the standard Arm C runtime required by libc++.

These settings complement `CONFIG_COMMON_LIBC_MALLOC`, C++17, hard-float, and
the Zephyr-managed heaps in `prj.conf`. They are build integration requirements,
not model-generation settings.

## Multiple-network example

The [TinyCNN multiple-network example](../model/TinyCNN/README.md) exports an
A16W8 `filter` and an INT8 `classify` into one `.pt2`, including runtime shape
constants. This solution includes a commented model input and an alternative
test selection. Follow the linked steps to generate it, select the test, and
compare the filter-then-classify output against a fixed reference.
