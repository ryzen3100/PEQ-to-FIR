# PEQ to FIR Converter

Convert AutoEQ-style Parametric EQ settings to high-quality FIR filters using scipy.signal.firwin2.

## Features

- **Intuitive GUI**: Text editor with drag & drop support
- **Multiple Filter Types**: Peaking, Low Shelf, High Shelf
- **Optimized Settings**: 
  - 2047, 4095 (default), 8191 taps
  - Linear Phase / Minimum Phase selection
  - Simultaneous 44.1 kHz / 48 kHz generation
  - 16-bit / 24-bit / 32-bit float output
- **Automatic Preamp**: Auto gain adjustment for clipping prevention
- **Real-time Preview**: Frequency response visualization
- **Multiple Output Formats**: WAV, TXT, JSON (with metadata)

## Installation

```bash
# GUI version installation
pip install -r requirements_gui.txt

# Or install base libraries only
pip install -r requirements.txt
```

## Usage

### Run GUI Application

```bash
python peq_to_fir_gui.py
```

### Programmatic Usage

```python
from peq_to_fir_converter import PEQtoFIR

# PEQ settings
peq_filters = [
    {'type': 'peaking', 'freq': 100, 'q': 1.41, 'gain': -3.0},
    {'type': 'highshelf', 'freq': 10000, 'q': 0.707, 'gain': 3.0}
]

# Convert
converter = PEQtoFIR(fs=48000, num_taps=4095)
fir_coeffs = converter.design_fir_filter(peq_filters, apply_preamp=True)
```

## PEQ Format

### AutoEQ Format
```
Filter 1: ON PK Fc 100 Hz Gain -3.0 dB Q 1.41
Filter 2: ON HS Fc 10000 Hz Gain 3.0 dB Q 0.707
```

### Simple Format
```
peaking 100 1.41 -3.0
highshelf 10000 0.707 3.0
```

## Recommended Settings

- **General Music Listening**: 4095 taps, Linear Phase
- **Real-time Monitoring**: 2047 taps, Minimum Phase
- **Maximum Precision**: 8191 taps, Linear Phase

## Output Files

- `FIR_linear_4095taps_44100Hz.wav`: 44.1 kHz impulse response
- `FIR_linear_4095taps_48000Hz.wav`: 48 kHz impulse response
- `FIR_linear_4095taps_44100Hz.txt`: Filter coefficients (text)
- `FIR_linear_4095taps_48000Hz.txt`: Filter coefficients (text)
- `filter_metadata.json`: Settings and performance metrics

## Web Version

An accurate web version is available at: https://grisys83.github.io/PEQ-to-FIR/

The web version implements an accurate JavaScript port of scipy.signal.firwin2 and produces results within 0.5dB of the Python version.

## Credits

- **scipy.signal.firwin2** - The reference implementation for FIR filter design
- **AutoEQ** - Parametric EQ format and methodology by jaakkopasanen
- Developed with assistance from **Claude (Anthropic)** and **Gemini 2.5 Pro (Google)**

## License

MIT License