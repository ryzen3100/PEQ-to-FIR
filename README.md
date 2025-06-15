# PEQ to FIR Converter

Convert AutoEQ-style Parametric EQ settings to high-quality FIR filters using scipy.signal.firwin2.

AutoEQ 스타일의 Parametric EQ 설정을 고품질 FIR 필터로 변환하는 도구입니다.

## 특징

- **직관적인 GUI**: 텍스트 에디터와 드래그 앤 드롭 지원
- **다양한 필터 타입**: Peaking, Low Shelf, High Shelf
- **최적화된 설정**: 
  - 4095 taps (기본값)
  - Linear Phase / Minimum Phase 선택 가능
  - 44.1 kHz / 48 kHz 동시 생성
  - 16-bit / 24-bit / 32-bit float 출력
- **자동 Preamp**: 클리핑 방지를 위한 자동 게인 조정
- **실시간 미리보기**: 주파수 응답 그래프
- **다양한 출력 형식**: WAV, TXT, JSON (메타데이터 포함)

## 설치

```bash
# GUI 버전 설치
pip install -r requirements_gui.txt

# 또는 기본 라이브러리만 설치
pip install -r requirements.txt
```

## 사용법

### GUI 애플리케이션 실행

```bash
python peq_to_fir_gui.py
```

### 프로그래밍 방식 사용

```python
from peq_to_fir_converter import PEQtoFIR

# PEQ 설정
peq_filters = [
    {'type': 'peaking', 'freq': 100, 'q': 1.41, 'gain': -3.0},
    {'type': 'highshelf', 'freq': 10000, 'q': 0.707, 'gain': 3.0}
]

# 변환
converter = PEQtoFIR(fs=48000, num_taps=4095)
fir_coeffs = converter.design_fir_filter(peq_filters, apply_preamp=True)
```

## PEQ 형식

### AutoEQ 형식
```
Filter 1: ON PK Fc 100 Hz Gain -3.0 dB Q 1.41
Filter 2: ON HS Fc 10000 Hz Gain 3.0 dB Q 0.707
```

### 간단한 형식
```
peaking 100 1.41 -3.0
highshelf 10000 0.707 3.0
```

## 권장 설정

- **일반 음악 감상**: 4096 taps, Linear Phase
- **실시간 모니터링**: 2048 taps, Minimum Phase
- **최고 정밀도**: 8192 taps, Linear Phase

## 출력 파일

- `FIR_linear_4095taps_44100Hz.wav`: 44.1 kHz 임펄스 응답
- `FIR_linear_4095taps_48000Hz.wav`: 48 kHz 임펄스 응답
- `FIR_linear_4095taps_44100Hz.txt`: 필터 계수 (텍스트)
- `FIR_linear_4095taps_48000Hz.txt`: 필터 계수 (텍스트)
- `filter_metadata.json`: 설정 및 성능 메트릭

## Web Version

An accurate web version is available at: https://[your-username].github.io/PEQ-to-FIR/

The web version implements an accurate JavaScript port of scipy.signal.firwin2 and produces results within 0.5dB of the Python version.

## Credits

- **scipy.signal.firwin2** - The reference implementation for FIR filter design
- **AutoEQ** - Parametric EQ format and methodology by jaakkopasanen
- Developed with assistance from **Claude (Anthropic)** and **Gemini 2.5 Pro (Google)**

## License

MIT License