"""
AcousticProbe — Full FMCW Analysis Pipeline
Based on 6.808 Lab 4 methodology, extended for HAR.

Usage:
    python analyze_fmcw.py <recording.wav>
"""

import sys
import json
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt

# ── Parameters ────────────────────────────────────────────────────────────────

def load_params(wav_path: Path) -> dict:
    json_path = wav_path.with_suffix(".json")
    if json_path.exists():
        return json.loads(json_path.read_text())
    return {"sample_rate_hz": 48000, "f_start_hz": 18000,
            "f_end_hz": 22000, "sweep_duration_s": 0.020}

# ── Filters ───────────────────────────────────────────────────────────────────

def bandpass(data, low, high, fs, order=5):
    sos = signal.butter(order, [low, high], btype='band', fs=fs, output='sos')
    return signal.sosfilt(sos, data)

def lowpass(data, cutoff, fs, order=5):
    sos = signal.butter(order, cutoff, btype='low', fs=fs, output='sos')
    return signal.sosfilt(sos, data)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# ── Chirp reference ───────────────────────────────────────────────────────────

def make_chirp(params: dict) -> np.ndarray:
    fs = params["sample_rate_hz"]
    f0 = params["f_start_hz"]
    f1 = params["f_end_hz"]
    T  = params["sweep_duration_s"]
    N  = int(fs * T)
    t  = np.arange(N) / fs
    B  = f1 - f0
    return np.sin(2 * np.pi * (f0 * t + (B / (2 * T)) * t**2)).astype(np.float32)

# ── Core FMCW pipeline ────────────────────────────────────────────────────────

def process(wav_path: str):
    path   = Path(wav_path)
    params = load_params(path)
    fs_raw, data = wavfile.read(str(path))

    # Normalize
    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    if data.ndim > 1:
        data = data[:, 0]

    fs  = params["sample_rate_hz"]
    f0  = params["f_start_hz"]
    f1  = params["f_end_hz"]
    T   = params["sweep_duration_s"]
    N   = int(fs * T)       # samples per chirp
    B   = f1 - f0

    print(f"File        : {path.name}")
    print(f"Duration    : {len(data)/fs:.2f} s")

    # ── Step 1: bandpass filter received signal ───────────────────────────────
    rx_filtered = bandpass(data, f0 - 500, f1 + 500, fs)

    # ── Step 2: segment into chirps, drop first 1 s (transient) ──────────────
    drop        = int(1.0 / T)
    num_chirps  = len(rx_filtered) // N
    rx_filtered = rx_filtered[:num_chirps * N]
    rx_data     = rx_filtered.reshape(num_chirps, N)

    chirp_ref   = make_chirp(params)
    tx_data     = np.tile(chirp_ref, (num_chirps, 1))

    rx_data = rx_data[drop:]
    tx_data = tx_data[drop:]
    num_chirps -= drop
    t_chirps = np.arange(num_chirps) * T

    print(f"Chirps used : {num_chirps}  (dropped first {drop})")

    # ── Step 3: mix (Tx × Rx) + lowpass → beat signal ────────────────────────
    LOWPASS_CUTOFF = 5000
    mixed     = rx_data * tx_data
    mixed_lp  = np.apply_along_axis(lambda x: lowpass(x, LOWPASS_CUTOFF, fs), 1, mixed)

    # ── Step 4: FFT range profile ─────────────────────────────────────────────
    NFFT      = N * 4
    range_ffts = np.abs(np.fft.rfft(mixed_lp, n=NFFT, axis=1))
    freq_axis  = np.fft.rfftfreq(NFFT, d=1/fs)
    c          = 343.0
    range_axis = freq_axis * c * T / (2 * B)   # beat freq → distance (m)

    # ── Step 5: background subtraction (consecutive frame diff) ───────────────
    bg_sub = np.diff(range_ffts, axis=0)        # shape: (num_chirps-1, NFFT//2+1)
    bg_sub = np.abs(bg_sub)
    t_bg   = t_chirps[1:]

    # ── Step 6: track peak within 0–3 m window ────────────────────────────────
    max_range_idx = np.searchsorted(range_axis, 3.0)
    bg_window     = bg_sub[:, :max_range_idx]
    raw_peak_idx  = np.argmax(bg_window, axis=1)

    # Median filter + moving average for smoothing
    MA  = 5
    MED = 7
    smoothed_idx = signal.medfilt(raw_peak_idx.astype(float), MED)
    smoothed_idx = moving_average(smoothed_idx, MA)
    t_smooth     = t_bg[MA // 2 : MA // 2 + len(smoothed_idx)]

    peak_distances = range_axis[smoothed_idx.astype(int)]

    # ── Step 7: phase-based displacement (for respiration) ────────────────────
    # Use the dominant range bin; extract complex phase over time
    median_bin     = int(np.median(raw_peak_idx))
    complex_ffts   = np.fft.rfft(mixed_lp, n=NFFT, axis=1)
    phase_series   = np.angle(complex_ffts[:, median_bin])
    phase_unwrapped = np.unwrap(phase_series)
    # Convert phase → displacement: Δd = Δφ * c / (4π * fc)
    fc             = (f0 + f1) / 2
    displacement_mm = (phase_unwrapped - phase_unwrapped[0]) * c / (4 * np.pi * fc) * 1000

    # Low-pass at 3 Hz to isolate respiration (~0.1–0.5 Hz range)
    resp_signal = lowpass(displacement_mm, 3.0, 1/T, order=3)

    # Estimate respiration rate via FFT of displacement
    resp_fft   = np.abs(np.fft.rfft(resp_signal - np.mean(resp_signal)))
    resp_freqs = np.fft.rfftfreq(len(resp_signal), d=T)
    # Only look at 0.1–1 Hz (6–60 breaths/min)
    resp_mask  = (resp_freqs >= 0.1) & (resp_freqs <= 1.0)
    if resp_mask.any():
        peak_resp_freq = resp_freqs[resp_mask][np.argmax(resp_fft[resp_mask])]
        print(f"Est. respiration rate: {peak_resp_freq * 60:.1f} breaths/min")

    # ── Plotting ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(13, 11))
    fig.suptitle(f"AcousticProbe — {path.name}", fontsize=13)

    # 1. Spectrogram of raw mic signal
    ax = axes[0]
    ax.specgram(data, Fs=fs, NFFT=512, noverlap=480, cmap="inferno")
    ax.set_ylim(15000, 25000)
    ax.axhline(f0, color="cyan", lw=0.8, ls="--")
    ax.axhline(f1, color="lime",  lw=0.8, ls="--")
    ax.set_title("Spectrogram (raw mic)")
    ax.set_ylabel("Freq (Hz)")
    ax.set_xlabel("Time (s)")

    # 2. Range profile heatmap (background subtracted)
    ax2 = axes[1]
    r_plot = range_axis[:max_range_idx]
    im = ax2.pcolormesh(t_bg, r_plot, bg_window.T, shading='gouraud', cmap='hot')
    ax2.plot(t_smooth, peak_distances[:len(t_smooth)], 'c-', lw=1.2, label='peak dist')
    ax2.set_title("Range profile (background subtracted)")
    ax2.set_ylabel("Distance (m)")
    ax2.set_xlabel("Time (s)")
    ax2.legend(fontsize=8)
    plt.colorbar(im, ax=ax2)

    # 3. Distance over time
    ax3 = axes[2]
    ax3.plot(t_smooth, peak_distances[:len(t_smooth)], lw=1.2)
    ax3.set_title("Tracked distance (smoothed)")
    ax3.set_ylabel("Distance (m)")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True, alpha=0.3)

    # 4. Phase displacement (respiration)
    ax4 = axes[3]
    ax4.plot(t_chirps, displacement_mm, lw=0.8, alpha=0.5, label='raw phase disp.')
    ax4.plot(t_chirps, resp_signal, lw=1.5, color='red', label='<3 Hz (respiration)')
    ax4.set_title("Phase-based displacement (HAR: respiration)")
    ax4.set_ylabel("Displacement (mm)")
    ax4.set_xlabel("Time (s)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    out = path.with_name(path.stem + "_analyzed.png")
    plt.savefig(str(out), dpi=150)
    print(f"Saved → {out}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_fmcw.py <recording.wav>")
        sys.exit(1)
    process(sys.argv[1])
