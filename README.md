# PEQ to FIR Converter
[![GitHub Fork](https://img.shields.io/badge/GitHub-Fork-green?logo=github)](https://github.com/ryzen3100/PEQ-to-FIR) [![GitHub Original](https://img.shields.io/badge/GitHub-Original-blue?logo=github)](https://github.com/grisys83/PEQ-to-FIR)

An actively maintained fork featuring modern packaging and CLI functionality. Converts AutoEQ-style Parametric EQ settings to high-quality FIR filters using scipy.signal.firwin2.

## Key Features

### Dual Interface Support
- **GUI Application**: Intuitive drag-and-drop interface with real-time preview
- **Command Line Interface**: Scriptable batch processing with arguments

### Advanced Filter Options
- **Phase Types**: Linear or Minimum phase output
- **Tap Count Selection**: 2047, 4095 (default), or 8191 taps
- **Bit Depth Options**: 16/24/32-bit output formats
- **Sample Rate Generation**: Simultaneous 44.1kHz & 48kHz output

### Preamp Management
- File-specified preamp (from input file)
- Automatic clipping prevention (adjusts overall gain)

### Output Artifacts
- WAV impulse responses (handled by WAVOutputHandler)
- Text coefficient files (handled by TextOutputHandler)
- JSON metadata with filter characteristics and error analysis (handled by JSONOutputHandler)

## Installation

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install with all dependencies
pip install -e .
```

## Usage

### Command Line Interface
```bash
# Basic conversion
peq2fir input.peq

# Customized conversion
peq2fir input.peq \
  --taps 4095 \
  --phase linear \
  --bit-depth 24 \
  --output my_filters/
```

**Available Options**:
- `--taps [2047|4095|8191]`
- `--phase [linear|minimum]`
- `--bit-depth [16|24|32]`
- `--channels {1,2}` (default: 1)
- `--output <directory>` (default: `output/`)

### GUI Application
```bash
python -m peq2fir.gui
```

### Programmatic Usage
```python
from peq2fir import PEQtoFIR

# Configure converter
converter = PEQtoFIR(fs=48000, num_taps=4095)

# Create filter definitions
peq_filters = [
    {'type': 'peaking', 'freq': 100, 'q': 1.41, 'gain': -3.0},
    {'type': 'highshelf', 'freq': 10000, 'q': 0.707, 'gain': 3.0}
]

# Generate FIR filter
fir_coeffs = converter.design_fir_filter(
    peq_filters,
    use_file_preamp=True,
    use_auto_preamp=True
)
```

## PEQ Format Guidelines

### AutoEQ Format
```text
Preamp: -3.7 dB
Filter 1: ON PK Fc 65 Hz Gain -4.5 dB Q 0.8
Filter 2: ON HS Fc 10000 Hz Gain 2.0 dB Q 0.707
```

### Simple Format
```text
peaking 100 1.41 -3.0
highshelf 10000 0.707 3.0
```

> Note: Stereo output files include `Stereo_` in the filename.

## Output Files Structure
When processing `input.peq`, output includes:
- `input_FIR_Linear_4095taps_44100Hz.wav`
- `input_FIR_Linear_4095taps_48000Hz.wav`
- `input_FIR_Linear_4095taps_44100Hz.txt`
- `input_FIR_Linear_4095taps_48000Hz.txt`
- `input_FIR_metadata.json` (with filter characteristics and error analysis)
- `input_FIR_Stereo_Linear_4095taps_44100Hz.wav` (stereo output example)

## Architecture

- **Core functionality**:
  - Implemented in `peq2fir/converter.py`
  - Contains `PEQtoFIR` class with biquad response calculation and FIR design
  - Unified preamp handling via `get_final_target_response`
  - File I/O separated through output handler interfaces

- **Interfaces**:
  - `peq2fir/gui.py`: Tkinter-based GUI with visualization
  - `peq2fir/cli.py`: Command-line interface with comprehensive argument support
  - `peq2fir/output_handler.py`: Abstract base classes for output handlers

- **Output handlers**:
  - `WAVOutputHandler`: Generates WAV impulse responses (16/24/32-bit)
  - `TextOutputHandler`: Generates text coefficient files
  - `JSONOutputHandler`: Generates metadata with filter characteristics and error analysis

## Recommended Settings

| Use Case               | Taps | Phase   | Bit Depth |
|------------------------|------|---------|-----------|
| Music Listening        | 4095 | Linear  | 24        |
| Real-time Processing   | 2047 | Minimum | 16        |
| Studio Mastering       | 8191 | Linear  | 32        |

## Credits

- **Original Implementation**: [grisys83](https://github.com/grisys83/PEQ-to-FIR)
- **Modern Fork**: [ryzen3100](https://github.com/ryzen3100/PEQ-to-FIR) - Added CLI, packaging, and test suite
- **Core Technology**: [scipy.signal.firwin2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin2.html)
- **Filter Format**: [AutoEQ](https://github.com/jaakkopasanen/AutoEQ) by jaakkopasanen

## License

MIT License

> *Note: This fork maintains compatibility with the original project while modernizing the development experience.*