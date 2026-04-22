# AcousticProbe

FMCW acoustic sensing toolkit for iPhone — contactless Human Activity Recognition via passive acoustic probes.

**Paper:** *AcousticProbe: Augmenting Smartphone Continuous Human Activity Recognition via Passive Acoustic Probes*

## Overview

Uses the iPhone's built-in speaker and microphone to transmit/receive inaudible 18–22 kHz FMCW chirps for contactless sensing:

- Respiration monitoring
- Micro-gesture tracking
- General human activity recognition (HAR)

## Repository Structure

```
AcousticProbe/
├── FMCWEngine.swift       # iOS audio engine: chirp generation + recording
├── ContentView.swift      # SwiftUI interface
├── AcousticProbeApp.swift # App entry point
├── analyze_fmcw.py        # Full FMCW processing pipeline (Python)
└── test_signal.py         # Quick signal validation script
```

## iOS App

**Requirements:** Xcode, iPhone (tested on iPhone 16/17 Pro)

Signal parameters:
| Parameter | Value |
|-----------|-------|
| Frequency range | 18–22 kHz |
| Bandwidth | 4 kHz |
| Sweep duration | 20 ms/chirp |
| Sample rate | 48 kHz |

## Python Analysis Pipeline

```bash
pip install numpy scipy matplotlib
python analyze_fmcw.py recording.wav
```

Outputs:
1. Raw spectrogram
2. Range profile heatmap (background subtracted)
3. Tracked distance over time
4. Phase-based displacement / respiration waveform + rate estimate

## Pipeline

```
Chirp TX (18–22 kHz) → Speaker
Mic → Bandpass filter → Mix (Tx × Rx) → Lowpass (5 kHz)
→ FFT range profile → Background subtraction
→ Peak tracking → Distance curve
→ Phase unwrapping → Displacement / respiration
```
