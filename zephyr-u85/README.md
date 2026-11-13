# Zephyr Cortex-M with Ethos-U85

This Zephyr template deploys ML models on Cortex-M targets with Ethos-U85
acceleration using ExecuTorch.

The supplied configurations run on the Alif Ensemble E8 AI/ML AppKit or the
Corstone-320 FVP using Zephyr 4.5.

> [!NOTE]
> The Alif E8 hardware target is included temporarily for development and
> validation. It will be removed from the final template.

> [!IMPORTANT]
> This Zephyr template does not use the `ARM::CMSIS-Ethos-U` pack. The Ethos-U
> driver is part of upstream Zephyr 4.5 and uses Zephyr's `hal_ethos_u` module.
> `PyTorch::ExecuTorch` remains an external module and supplies the ExecuTorch
> runtime and Ethos-U backend.

The current solution uses the `PyTorch::ExecuTorch` 1.6.0 Zephyr preview
module. Its path is supplied to Zephyr through `EXTRA_ZEPHYR_MODULES`.
The Zephyr targets use the Arm GNU Toolchain (`compiler: GCC`) by default.
Their CMake and Kconfig configuration also carries the AC6 6.24 compatibility
needed by the `PyTorch::ExecuTorch` preview pack when the solution compiler is changed to
AC6.

## Quick Start

### 1. Prepare Zephyr 4.5

Create or select a Zephyr 4.5 workspace and fetch the modules needed by the
targets:

```sh
west init zephyr-workspace --mr v4.5.0
cd zephyr-workspace
west config manifest.project-filter -- +zephyr,+cmsis,+cmsis_6,+hal_ethos_u
west update
west zephyr-export
```

### 2. Select a target-set

Choose one of these CMSIS-Toolbox target-sets:

- `Corstone-320-FVP` for simulation.
- `AppKit-E8` for an Alif Ensemble E8 AI/ML AppKit using its HP Cortex-M55 and
  256-MAC Ethos-U85.

In command-line operations, enable the selection with `--active`, for example
`cbuild zephyr-u85/zephyr-u85.csolution.yml --active Corstone-320-FVP`.

The solution passes the selected Zephyr board and overlay to `west`. Its
`mlops:` node describes the Ethos-U85-256 and Vela memory configuration used to
generate the model. `fvp_config.txt` configures the Corstone-320 FVP with the
matching 256-MAC NPU, enables the Ethos-U fast functional mode, and routes
Zephyr's UART0 console to standard output. Fast mode is intended for functional
validation; its reported NPU timing is not cycle-accurate.
FVP builds write EOT (`0x04`) to UART0 when the application finishes; together
with `shutdown_on_eot=1` in `fvp_config.txt`, this terminates the simulation
without waiting for `--simlimit`. AppKit builds do not emit the control
character.

### 3. Build and run

Run **CMSIS: Build** in VS Code. For `Corstone-320-FVP`, start the generated
Arm FVP debug configuration. For `AppKit-E8`, connect the board's J-Link probe
and start the generated debug configuration.

The application initializes the Zephyr Ethos-U device, loads the embedded
ExecuTorch program from `ai_layer/TinyCNN_pte.c`, runs inference, and reports the
result on the console.

### Configure with menuconfig

Build the desired target once with the CMSIS Solution extension. From the
repository root, open a terminal using the same tools environment and run:

```powershell
cmake --build zephyr-u85/out/Zephyr_app/Corstone-320-FVP/Debug --target menuconfig
```

Use `--target guiconfig` for the graphical interface. For the physical board,
replace `Corstone-320-FVP` with `AppKit-E8` after building that target once.
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
Ethos-U85 (NPU-HG) driver communication check
Device: npu@49042000, base: 0x49042000, IRQ: 366
Driver version: 2.0.0
Hardware architecture: 2.0.0
Hardware revision: 0.0 (status 1)
Product revision: 7
Command stream version: 1
Custom DMA: 0
MACs per cycle: 256
Ethos-U85 driver communication OK (ID/configuration read; no inference).
ExecuTorch Ethos-U85-256 application: 8656 byte program
Input shape: [1, 16, 16, 3]
I: Optimizer release nbr: 0 patch: 1
I: Optimizer config. product=2, cmd_stream_version=1, macs_per_cc=8, num_axi_ext=1, num_axi_sram=2, custom_dma=0
I: Optimizer config. arch version=2.0.0
I: Ethos-U config. product=2, cmd_stream_version=1, macs_per_cc=8, num_axi_ext=1, num_axi_sram=2, custom_dma=0
I: Ethos-U. arch version=2.0.0
I: handle_command_stream: cmd_stream=0x80231f90, cms_length 216 words (864 bytes)
I: Soft reset ethos-u85
Output: 10 element(s): -0.0142 0.0206 0.0601 -0.0886 0.0791 0.0522 -0.0301 -0.0775 -0.2800 -0.0095
Test_result: PASS
```

## Configuring the Ethos-U target

The configuration-consistency guidance in the
[CMSIS-Ethos-U integration guide](https://arm-software.github.io/CMSIS-Ethos-U/latest/integration/index.html)
also applies to this Zephyr template, although the `ARM::CMSIS-Ethos-U` pack
does not. The default U85-256 target is described by the solution's `mlops:`
settings, `CONFIG_ETHOS_U85_256=y` in `prj.conf`, the target devicetree overlay,
and `mps4_board.subsystem.ethosu.num_macs=256` in `fvp_config.txt`.

The guide provides a
[256-MAC FVP configuration example](https://arm-software.github.io/CMSIS-Ethos-U/latest/integration/index.html#configure-256-macs-for-fvp-simulation).
For Zephyr, apply the equivalent driver configuration through Kconfig and
devicetree. If the NPU configuration changes, keep those settings and the FVP
aligned, regenerate the AI layer so Vela targets the matching accelerator, and
rebuild the application.

## How it works

The CMSIS solution does not build the application directly. `cbuild` translates
the selected solution context into a `west build` invocation. The selected
target type, build type, compiler, packs, and `west-defs` become Zephyr board,
CMake, and Kconfig arguments. Zephyr then combines those arguments with the
application files described below.

### Solution-to-west configuration

[`zephyr-u85.csolution.yml`](zephyr-u85.csolution.yml) supplies these values to
west:

| Solution setting | Purpose |
|---|---|
| `west-board` | Selects `mps4/corstone320/fvp` for the FVP or `ensemble_e8_ak/ae822fa0e5597bs0/rtss_hp` for the AppKit. The project-level `board: $west-board$` passes the selected value to west. |
| `DTC_OVERLAY_FILE` | Selects the target-specific devicetree overlay: `fvp.overlay` or `app.overlay`. |
| `EXTRA_CONF_FILE: app.conf` | Adds the AppKit-only Kconfig fragment. It is intentionally absent from the FVP target. |
| `CMAKE_CXX_FLAGS: -includestddef.h` | Applies the current ExecuTorch header compatibility workaround to the FVP build. The single argument keeps the header attached to `-include` when CMSIS-Toolbox invokes CMake. |
| Debug `CONFIG_*` definitions | Enable Zephyr debug information, thread information, and debug optimization for the `Debug` build type. |
| Release `CONFIG_SPEED_OPTIMIZATIONS` | Selects Zephyr's speed optimization for the `Release` build type. |
| `EXTRA_ZEPHYR_MODULES: ${CMSIS_PACK_ROOT}/PyTorch/ExecuTorch/1.6.0-zephyrpreview2` | Makes the installed ExecuTorch pack root an external Zephyr module. This exposes the pack's Kconfig symbols, CMake sources, runtime, operators, and Ethos-U backend. |

The `mlops:` node is consumed before the application build. Solution setup
generates `zephyr-u85.cbuild-mlops.yml`; `create_ai_layer.py` reads that file,
exports and lowers the model for Ethos-U85-256, and writes `ai_layer/`. Changing
the model, accelerator, MAC count, Vela system, or Vela memory mode therefore
requires regenerating the AI layer before rebuilding.

### Configuration files

| File | Scope and function |
|---|---|
| [`prj.conf`](prj.conf) | Common persistent Kconfig configuration for both targets. It enables the Zephyr Ethos-U85-256 driver, C++17, hard-float ABI, cache maintenance, logging, heaps, stack size, and floating-point `printf`. |
| [`app.conf`](app.conf) | AppKit-only Kconfig configuration, selected through `EXTRA_CONF_FILE`. `CONFIG_FLASH_LOAD_OFFSET=0x200000` places the application after the platform firmware in MRAM. It does not apply to Corstone-320. |
| [`app.overlay`](app.overlay) | AppKit devicetree changes. It declares and enables the Ethos-U85, combines the two SRAM banks into an 8-MiB NPU-accessible region, and selects that region as `zephyr,sram`; the NPU cannot DMA to the board's default Cortex-M55 DTCM. |
| [`fvp.overlay`](fvp.overlay) | Corstone-320 devicetree changes. The board DTS already selects its 4-MiB ISRAM as flash and declares the Ethos-U85. The overlay enables that NPU, corrects the UART0 address, and disables UART3/4 pin-mux initialization that faults on the FVP. |
| [`fvp_config.txt`](fvp_config.txt) | Parameters for the FVP executable, not Zephyr. It selects 256 MACs, enables fast functional simulation, sets the secure reset vectors, configures UART0 as the console, shuts down on EOT, and disables unused terminals and visualization. |
| `ai_layer/components.conf` | Generated Kconfig fragment enabling only the ExecuTorch runtime, support libraries, kernel registration, and Ethos-U backend required by the generated layer. |
| `ai_layer/operators.conf` | Generated Kconfig fragment enabling only the operators detected in the lowered model. |
| `ai_layer/CMakeLists.txt` | Generated CMake fragment that attaches `TinyCNN_pte.c` and its include directory directly to Zephyr's `app` target. Direct ownership prevents the model from being compiled twice during AC6 pre-link. |
| `ai_layer/VELA_SUMMARY.md` | Generated record of the effective Vela configuration and Vela's textual compilation report. |

The two generated `ai_layer/*.conf` files must be appended to
`EXTRA_CONF_FILE` before `find_package(Zephyr)`, because Zephyr resolves Kconfig
while that call runs. Afterward, `add_subdirectory(ai_layer)` attaches the
generated model source to the `app` target. This ordering in
[`CMakeLists.txt`](CMakeLists.txt) is significant.

### Why `app.conf` is target-specific

The 2-MiB load offset describes the AppKit MRAM layout, not the model or the
Ethos-U85. Keeping it out of the common `prj.conf` prevents AppKit platform
assumptions from leaking into the FVP configuration.

Corstone-320 would not exhibit the exact U55 negative-size linker failure even
if the offset were accidentally applied: its selected flash is 4-MiB ISRAM, so
subtracting a 2-MiB offset still leaves a 2-MiB link region. Corstone-300 U55
selects only 512-KiB ITCM as flash; the same offset made the computed region
negative and caused AC6 `L6225E`. The U85 separation is nevertheless required
because the offset is semantically AppKit-specific and unnecessarily halves the
FVP's available image space.

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

## Adapting the template

- Change board, device, overlay, and debugger settings in
  `zephyr-u85.csolution.yml` for the target hardware.
- Change `prj.conf` for application-wide Zephyr settings and use a target
  overlay for the Ethos-U device description when the board does not provide
  one.
- Replace the model through the shared `model/` and `model_conversion/`
  workflow. The generated `ai_layer/components.conf` and `operators.conf`
  select only the required ExecuTorch features.

Do not add `ARM::CMSIS-Ethos-U` to this solution. That pack is used only by the
CMSIS workflow templates in this repository.

## Multiple-network example

The [TinyCNN multiple-network example](../model/TinyCNN/README.md) exports an
A16W8 `filter` and an INT8 `classify` into one `.pt2`, including runtime shape
constants. This solution includes a commented model input and an alternative
test selection. Follow the linked steps to generate it, select the test, and
compare the filter-then-classify output against a fixed reference.
