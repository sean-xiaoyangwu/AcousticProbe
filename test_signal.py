"""
Quick FMCW signal validation script.
Usage:  python test_signal.py fmcw_<timestamp>.wav
"""

import sys
import json
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from pathlib import Path

def load_params(wav_path: Path) -> dict:
    json_path = wav_path.with_suffix(".json")
    if json_path.exists():
        return json.loads(json_path.read_text())
    # Defaults
    return {"sample_rate_hz": 48000, "f_start_hz": 18000,
            "f_end_hz": 22000, "sweep_duration_s": 0.02}

def make_chirp(params: dict) -> np.ndarray:
    fs   = params["sample_rate_hz"]
    f0   = params["f_start_hz"]
    f1   = params["f_end_hz"]
    T    = params["sweep_duration_s"]
    N    = int(fs * T)
    t    = np.arange(N) / fs
    B    = f1 - f0
    phi  = 2 * np.pi * (f0 * t + (B / (2 * T)) * t**2)
    return np.sin(phi).astype(np.float32)

def main(wav_path: str):
    path   = Path(wav_path)
    params = load_params(path)
    fs_wav, data = wav.read(str(path))

    # Normalize to float [-1, 1]
    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    if data.ndim > 1:
        data = data[:, 0]

    fs     = params["sample_rate_hz"]
    chirp  = make_chirp(params)
    T      = params["sweep_duration_s"]
    N      = int(fs * T)

    print(f"File       : {path.name}")
    print(f"WAV rate   : {fs_wav} Hz  |  Expected: {int(fs)} Hz")
    print(f"Duration   : {len(data)/fs_wav:.2f} s  ({len(data)} samples)")
    print(f"Chirps     : ~{len(data)/(fs_wav*T):.0f}")

    # ── Plot 1: spectrogram of received signal ──────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle(f"AcousticProbe — {path.name}", fontsize=13)

    ax = axes[0]
    ax.specgram(data, Fs=fs_wav, NFFT=512, noverlap=480, cmap="inferno")
    ax.set_title("Spectrogram (received mic signal)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(15000, 25000)
    ax.axhline(params["f_start_hz"], color="cyan", lw=0.8, ls="--", label="f_start")
    ax.axhline(params["f_end_hz"],   color="lime",  lw=0.8, ls="--", label="f_end")
    ax.legend(loc="upper right", fontsize=8)

    # ── Plot 2: FMCW range profile (first 5 chirps) ─────────────────────────
    n_chirps = min(5, len(data) // N)
    range_profiles = []
    for i in range(n_chirps):
        seg   = data[i*N : (i+1)*N]
        mixed = seg * chirp[:len(seg)]          # multiply received × reference
        win   = mixed * np.hanning(len(mixed))
        prof  = np.abs(np.fft.rfft(win, n=N*4)) # zero-pad for resolution
        range_profiles.append(prof)

    range_profiles = np.array(range_profiles)
    freq_axis = np.fft.rfftfreq(N*4, d=1/fs)
    # Convert beat frequency to range: r = (c / (2*B/T)) * f_beat
    c = 343.0
    range_axis = freq_axis * c * T / (2 * (params["f_end_hz"] - params["f_start_hz"]))

    ax2 = axes[1]
    for i, rp in enumerate(range_profiles):
        ax2.plot(range_axis, 20*np.log10(rp + 1e-9), alpha=0.7, label=f"chirp {i}")
    ax2.set_xlim(0, 5)
    ax2.set_xlabel("Range estimate (m)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_title("Range profile — first 5 chirps")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Plot 3: time-domain waveform (first 200 ms) ─────────────────────────
    show = min(int(0.2 * fs_wav), len(data))
    t_ax = np.arange(show) / fs_wav * 1000
    axes[2].plot(t_ax, data[:show], lw=0.5)
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_title("Time-domain waveform (first 200 ms)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = path.with_suffix(".png")
    plt.savefig(str(out), dpi=150)
    print(f"\nSaved plot → {out}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_signal.py <recording.wav>")
        sys.exit(1)
    main(sys.argv[1])
