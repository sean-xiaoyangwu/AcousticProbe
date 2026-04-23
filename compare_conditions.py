"""
AcousticProbe — Multi-condition comparison
Usage: python compare_conditions.py <bare.wav> <spiral3.wav> <spiral1_5.wav> <zigzag.wav>
Or:    python compare_conditions.py  (auto-detects 4 newest WAVs in ~/Downloads)
"""

import sys
import json
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

LABELS = ["Bare iPhone", "Spiral 3-turn", "Spiral 1.5-turn", "Zigzag"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_params(wav_path: Path) -> dict:
    j = wav_path.with_suffix(".json")
    if j.exists():
        return json.loads(j.read_text())
    return {"sample_rate_hz": 48000, "f_start_hz": 18000,
            "f_end_hz": 22000, "sweep_duration_s": 0.020}

def make_chirp(p):
    fs, f0, f1, T = p["sample_rate_hz"], p["f_start_hz"], p["f_end_hz"], p["sweep_duration_s"]
    N = int(fs * T); B = f1 - f0
    t = np.arange(N) / fs
    return np.sin(2 * np.pi * (f0 * t + (B / (2*T)) * t**2)).astype(np.float32)

def bandpass(x, lo, hi, fs, order=5):
    sos = signal.butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return signal.sosfilt(sos, x)

def lowpass(x, cut, fs, order=5):
    sos = signal.butter(order, cut, btype='low', fs=fs, output='sos')
    return signal.sosfilt(sos, x)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def process_one(wav_path: Path):
    p = load_params(wav_path)
    fs_raw, data = wavfile.read(str(wav_path))
    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    if data.ndim > 1:
        data = data[:, 0]

    fs, f0, f1, T = p["sample_rate_hz"], p["f_start_hz"], p["f_end_hz"], p["sweep_duration_s"]
    N = int(fs * T); B = f1 - f0
    drop = int(1.0 / T)

    rx = bandpass(data, f0-500, f1+500, fs)
    n  = (len(rx) // N) * N
    rx = rx[:n].reshape(-1, N)
    tx = np.tile(make_chirp(p), (rx.shape[0], 1))
    rx, tx = rx[drop:], tx[drop:]

    mixed = lowpass((rx * tx).reshape(-1), 5000, fs).reshape(rx.shape)

    NFFT      = N * 4
    ffts      = np.fft.rfft(mixed, n=NFFT, axis=1)
    mag_ffts  = np.abs(ffts)
    freq_ax   = np.fft.rfftfreq(NFFT, d=1/fs)
    range_ax  = freq_ax * 343 * T / (2 * B)

    # Background subtraction
    bg    = np.abs(np.diff(mag_ffts, axis=0))
    t_bg  = np.arange(bg.shape[0]) * T

    # Peak tracking (0–2 m)
    max_idx   = np.searchsorted(range_ax, 2.0)
    peak_idx  = np.argmax(bg[:, :max_idx], axis=1)
    smoothed  = signal.medfilt(peak_idx.astype(float), 7)
    smoothed  = moving_average(smoothed, 5)
    peak_dist = range_ax[smoothed.astype(int)]

    # SNR: ratio of peak power to mean noise floor in range profile
    mean_prof  = np.mean(mag_ffts, axis=0)
    peak_power = np.max(mean_prof[:max_idx])
    noise_floor = np.median(mean_prof[:max_idx])
    snr_db     = 20 * np.log10(peak_power / (noise_floor + 1e-9))

    # Phase displacement
    med_bin      = int(np.median(peak_idx))
    phase        = np.unwrap(np.angle(ffts[:, med_bin]))
    fc           = (f0 + f1) / 2
    disp_mm      = (phase - phase[0]) * 343 / (4 * np.pi * fc) * 1000
    resp         = lowpass(disp_mm, 3.0, 1/T, order=3)

    # Respiration rate
    resp_fft   = np.abs(np.fft.rfft(resp - np.mean(resp)))
    resp_freqs = np.fft.rfftfreq(len(resp), d=T)
    mask       = (resp_freqs >= 0.1) & (resp_freqs <= 1.0)
    resp_rate  = 0.0
    if mask.any():
        resp_rate = resp_freqs[mask][np.argmax(resp_fft[mask])] * 60

    # Peak sharpness: max / mean of top-10 around peak (higher = sharper)
    w = 10
    region = mean_prof[max(0, med_bin-w):med_bin+w]
    sharpness = np.max(region) / (np.mean(region) + 1e-9)

    return {
        "snr_db":      snr_db,
        "sharpness":   sharpness,
        "resp_rate":   resp_rate,
        "peak_dist":   peak_dist,
        "t_bg":        t_bg,
        "bg":          bg,
        "range_ax":    range_ax,
        "max_idx":     max_idx,
        "disp_mm":     disp_mm,
        "resp":        resp,
        "t_chirps":    np.arange(rx.shape[0]) * T,
        "mean_prof":   mean_prof,
        "freq_ax":     freq_ax,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main(paths):
    results = []
    for i, p in enumerate(paths):
        print(f"Processing [{LABELS[i]}]: {Path(p).name} ...")
        results.append(process_one(Path(p)))

    # ── Figure 1: Summary comparison ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("AcousticProbe — Condition Comparison", fontsize=14)

    # 1a. SNR bar chart
    ax = axes[0, 0]
    snrs = [r["snr_db"] for r in results]
    bars = ax.bar(LABELS, snrs, color=COLORS)
    ax.bar_label(bars, fmt="%.1f dB", padding=3, fontsize=9)
    ax.set_title("SNR (higher = better)")
    ax.set_ylabel("dB")
    ax.tick_params(axis='x', labelsize=8)

    # 1b. Peak sharpness bar chart
    ax = axes[0, 1]
    sharpness = [r["sharpness"] for r in results]
    bars = ax.bar(LABELS, sharpness, color=COLORS)
    ax.bar_label(bars, fmt="%.1f×", padding=3, fontsize=9)
    ax.set_title("Peak Sharpness (higher = better)")
    ax.set_ylabel("Peak / Mean ratio")
    ax.tick_params(axis='x', labelsize=8)

    # 1c. Distance tracking stability (std dev)
    ax = axes[1, 0]
    stds = [np.std(r["peak_dist"]) * 100 for r in results]  # in cm
    bars = ax.bar(LABELS, stds, color=COLORS)
    ax.bar_label(bars, fmt="%.1f cm", padding=3, fontsize=9)
    ax.set_title("Distance Tracking Jitter (lower = better)")
    ax.set_ylabel("Std dev (cm)")
    ax.tick_params(axis='x', labelsize=8)

    # 1d. Respiration rate
    ax = axes[1, 1]
    rates = [r["resp_rate"] for r in results]
    bars = ax.bar(LABELS, rates, color=COLORS)
    ax.bar_label(bars, fmt="%.1f bpm", padding=3, fontsize=9)
    ax.axhline(12, color='gray', ls='--', lw=0.8, label='Normal min (12)')
    ax.axhline(20, color='gray', ls=':',  lw=0.8, label='Normal max (20)')
    ax.set_title("Estimated Respiration Rate")
    ax.set_ylabel("breaths/min")
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    out1 = Path(paths[0]).parent / "comparison_summary.png"
    plt.savefig(str(out1), dpi=150)
    print(f"Saved → {out1}")

    # ── Figure 2: Range profiles overlay ─────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle("Range Profile & Phase Displacement Comparison", fontsize=13)

    ax = axes2[0]
    for i, r in enumerate(results):
        profile = 20 * np.log10(r["mean_prof"][:r["max_idx"]] + 1e-9)
        ax.plot(r["range_ax"][:r["max_idx"]], profile,
                color=COLORS[i], lw=1.5, label=LABELS[i])
    ax.set_title("Mean Range Profile (dB)")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Magnitude (dB)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes2[1]
    for i, r in enumerate(results):
        t = r["t_chirps"]
        ax.plot(t, r["resp"], color=COLORS[i], lw=1.5, label=LABELS[i])
    ax.set_title("Respiration Waveform (<3 Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (mm)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out2 = Path(paths[0]).parent / "comparison_detail.png"
    plt.savefig(str(out2), dpi=150)
    print(f"Saved → {out2}")

    # ── Print table ───────────────────────────────────────────────────────────
    print("\n─── Results Table ───────────────────────────────────────")
    print(f"{'Condition':<20} {'SNR (dB)':>10} {'Sharpness':>12} {'Jitter (cm)':>12} {'Resp (bpm)':>12}")
    print("─" * 68)
    for i, r in enumerate(results):
        print(f"{LABELS[i]:<20} {r['snr_db']:>10.1f} {r['sharpness']:>12.1f} "
              f"{np.std(r['peak_dist'])*100:>12.1f} {r['resp_rate']:>12.1f}")
    print("─" * 68)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 5:
        main(sys.argv[1:])
    else:
        # Auto-detect 4 newest WAVs in ~/Downloads
        downloads = Path.home() / "Downloads"
        wavs = sorted(downloads.glob("fmcw_*.wav"),
                      key=lambda p: p.stat().st_mtime)[-4:]
        if len(wavs) < 4:
            print(f"Found only {len(wavs)} WAV files. Need 4.")
            print("Usage: python compare_conditions.py <bare.wav> <spiral3.wav> <spiral1_5.wav> <zigzag.wav>")
            sys.exit(1)
        print("Auto-detected files (oldest → newest):")
        for i, w in enumerate(wavs):
            print(f"  [{LABELS[i]}] {w.name}")
        main([str(w) for w in wavs])
