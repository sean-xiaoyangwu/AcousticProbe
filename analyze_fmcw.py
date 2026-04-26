"""
AcousticProbe — Full FMCW Analysis Pipeline
Based on 6.808 Lab 4 methodology, extended for HAR.

Primary output : range-bin displacement  (gesture detection, cm scale)
Secondary output: phase-based displacement (breathing detection, mm scale)

Usage:
    python analyze_fmcw.py <recording.wav> [--target <distance_m>]

Options:
    --target <m>   Expected target distance in metres (e.g. --target 0.5).
                   Constrains bin search to ±0.15 m window around this value.
"""

import sys
import argparse
import json
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt


def load_params(wav_path: Path) -> dict:
    json_path = wav_path.with_suffix(".json")
    if json_path.exists():
        return json.loads(json_path.read_text())
    return {"sample_rate_hz": 48000, "f_start_hz": 18000,
            "f_end_hz": 22000, "sweep_duration_s": 0.020}


def bandpass(data, low, high, fs, order=5):
    sos = signal.butter(order, [low, high], btype='band', fs=fs, output='sos')
    return signal.sosfilt(sos, data)

def lowpass(data, cutoff, fs, order=5):
    sos = signal.butter(order, cutoff, btype='low', fs=fs, output='sos')
    return signal.sosfilt(sos, data)

def highpass_zp(data, cutoff, fs, order=2):
    sos = signal.butter(order, cutoff, btype='high', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, data)

def bandpass_zp(data, low, high, fs, order=2):
    sos = signal.butter(order, [low, high], btype='band', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, data)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def make_chirp(params: dict) -> np.ndarray:
    fs = params["sample_rate_hz"]
    f0 = params["f_start_hz"]
    f1 = params["f_end_hz"]
    T  = params["sweep_duration_s"]
    N  = int(fs * T)
    t  = np.arange(N) / fs
    B  = f1 - f0
    return np.sin(2 * np.pi * (f0 * t + (B / (2 * T)) * t**2)).astype(np.float32)


def process(wav_path: str, target_dist: float = None, motion_threshold: float = 3.0):
    path   = Path(wav_path)
    params = load_params(path)
    fs_raw, data = wavfile.read(str(path))

    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    if data.ndim > 1:
        data = data[:, 0]

    fs = params["sample_rate_hz"]
    f0 = params["f_start_hz"]
    f1 = params["f_end_hz"]
    T  = params["sweep_duration_s"]
    N  = int(fs * T)
    B  = f1 - f0
    c  = 343.0

    print(f"File     : {path.name}")
    print(f"Duration : {len(data)/fs:.2f} s")

    # ── Step 1: bandpass + segment into chirps ────────────────────────────────
    rx_filtered = bandpass(data, f0 - 500, f1 + 500, fs)
    drop        = int(1.0 / T)
    num_chirps  = len(rx_filtered) // N
    rx_data     = rx_filtered[:num_chirps * N].reshape(num_chirps, N)
    tx_data     = np.tile(make_chirp(params), (num_chirps, 1))

    rx_data    = rx_data[drop:]
    tx_data    = tx_data[drop:]
    num_chirps -= drop
    t_chirps   = np.arange(num_chirps) * T
    print(f"Chirps   : {num_chirps}  (dropped first {drop})")

    # ── Step 2: mix + lowpass → beat signal ──────────────────────────────────
    mixed    = rx_data * tx_data
    mixed_lp = np.apply_along_axis(
        lambda x: lowpass(x, 5000, fs), 1, mixed)

    # ── Step 3: FFT → complex range profile (computed once) ──────────────────
    NFFT         = N * 4
    complex_ffts = np.fft.rfft(mixed_lp, n=NFFT, axis=1)
    range_ffts   = np.abs(complex_ffts)
    freq_axis    = np.fft.rfftfreq(NFFT, d=1/fs)
    range_axis   = freq_axis * c * T / (2 * B)
    bin_m        = range_axis[1] - range_axis[0]   # metres per bin
    max_range_idx = np.searchsorted(range_axis, 3.0)

    print(f"Range res: {bin_m*1000:.1f} mm/bin")

    # ── Step 4 & 5: frame-diff background subtraction ────────────────────────
    # Consecutive diff (Lab 4 style): suppresses static clutter, highlights motion.
    diff_bg = np.abs(np.diff(range_ffts[:, :max_range_idx], axis=0))
    t_diff  = t_chirps[1:]

    # ── Target bin selection via normalised frame-diff ─────────────────────
    # Near-field leakage has huge absolute amplitude but low RELATIVE variation.
    # Person motion causes large RELATIVE change (bin fills / empties).
    # Dividing by per-frame amplitude reveals person location over leakage.
    MIN_RANGE_M  = 0.25   # excludes near-field tube leakage (<0.25 m)
    SEARCH_WIN_M = 0.20
    mean_range   = range_ffts[:, :max_range_idx].mean(axis=0)   # kept for plot
    mean_diff_prof = diff_bg.mean(axis=0)                        # absolute mean diff

    if target_dist is not None:
        lo = np.searchsorted(range_axis, max(MIN_RANGE_M, target_dist - SEARCH_WIN_M))
        hi = min(np.searchsorted(range_axis, target_dist + SEARCH_WIN_M), max_range_idx)
        target_bin = int(np.argmax(mean_diff_prof[lo:hi])) + lo
        print(f"Target   : {target_dist:.2f} m → search [{range_axis[lo]:.2f}, {range_axis[hi-1]:.2f}] m")
    else:
        min_idx    = np.searchsorted(range_axis, MIN_RANGE_M)
        target_bin = int(np.argmax(mean_diff_prof[min_idx:])) + min_idx

    target_m = range_axis[target_bin]
    print(f"Target bin: {target_bin} → {target_m:.2f} m")

    # ── Step 6: noise-gated argmax distance tracking ─────────────────────────
    # Only update position when peak diff exceeds noise threshold; otherwise
    # hold last known position. This prevents noisy jumps during quiet periods.
    PEAK_WIN = 80
    win_lo   = max(0, target_bin - PEAK_WIN)
    win_hi   = min(max_range_idx, target_bin + PEAK_WIN)

    win_slice = diff_bg[:, win_lo:win_hi]
    win_max   = win_slice.max(axis=1)
    local_peak = np.argmax(win_slice, axis=1)

    # Noise floor: 30th percentile of per-chirp peak values (robust to motion events)
    noise_floor = np.percentile(win_max, 30)
    MOTION_THRESH = motion_threshold * noise_floor

    # Hold-last-value gating: only move when motion exceeds threshold
    global_peak = np.empty(len(local_peak), dtype=float)
    last_valid  = float(target_bin)
    for i, (mx, pk) in enumerate(zip(win_max, local_peak)):
        if mx > MOTION_THRESH:
            last_valid = float(pk + win_lo)
        global_peak[i] = last_valid

    # Raw argmax (no gate) — kept for comparison plot
    raw_peak_m = range_axis[(local_peak + win_lo)]

    MA, MED = 5, 7
    sm_peak   = signal.medfilt(global_peak, MED)
    sm_peak   = moving_average(sm_peak, MA)
    t_tracked = t_diff[MA // 2: MA // 2 + len(sm_peak)]

    tracked_dist_m = range_axis[sm_peak.astype(int)]    # absolute distance (m)
    print(f"Motion thresh: {motion_threshold:.1f}× noise floor = {MOTION_THRESH:.4f} ({(win_max > MOTION_THRESH).mean()*100:.1f}% frames active)")

    # ── Step 7: phase displacement (breathing, sub-mm) ────────────────────────
    phase_raw        = np.angle(complex_ffts[:, target_bin])
    phase_unwrapped  = np.unwrap(phase_raw)
    fc               = (f0 + f1) / 2
    disp_mm          = (phase_unwrapped - phase_unwrapped[0]) * c / (4 * np.pi * fc) * 1000
    disp_detrended   = signal.detrend(disp_mm)
    disp_detrended   = highpass_zp(disp_detrended, 0.05, 1/T)
    resp_signal      = bandpass_zp(disp_detrended, 0.1, 3.0, 1/T)

    resp_fft   = np.abs(np.fft.rfft(resp_signal))
    resp_freqs = np.fft.rfftfreq(len(resp_signal), d=T)
    resp_mask  = (resp_freqs >= 0.1) & (resp_freqs <= 1.0)
    if resp_mask.any():
        peak_f = resp_freqs[resp_mask][np.argmax(resp_fft[resp_mask])]
        print(f"Resp rate: {peak_f * 60:.1f} breaths/min")

    # ── Plotting ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(5, 1, figsize=(13, 14))
    fig.suptitle(f"AcousticProbe — {path.name}", fontsize=13)

    # 1. Spectrogram
    ax = axes[0]
    ax.specgram(data, Fs=fs, NFFT=512, noverlap=480, cmap="inferno")
    ax.set_ylim(15000, 25000)
    ax.axhline(f0, color="cyan", lw=0.8, ls="--")
    ax.axhline(f1, color="lime",  lw=0.8, ls="--")
    ax.set_title("Spectrogram (raw mic)")
    ax.set_ylabel("Freq (Hz)"); ax.set_xlabel("Time (s)")

    # 2. Range profile heatmap — consecutive diff, Lab 4 style
    ax2 = axes[1]
    im = ax2.pcolormesh(t_diff, range_axis[:max_range_idx],
                        diff_bg.T, shading='gouraud', cmap='hot')
    ax2.plot(t_tracked, tracked_dist_m, 'c-', lw=1.2, label='tracked dist')
    ax2.axhline(target_m, color='yellow', lw=1.0, ls='--',
                label=f'target ({target_m:.2f} m)')
    ax2.axhline(range_axis[win_lo],    color='white', lw=0.6, ls=':')
    ax2.axhline(range_axis[win_hi - 1], color='white', lw=0.6, ls=':')
    ax2.set_title("Range profile (frame diff) — motion events")
    ax2.set_ylabel("Distance (m)"); ax2.set_xlabel("Time (s)")
    ax2.legend(fontsize=8)
    plt.colorbar(im, ax=ax2)

    # 3. Tracked distance — PRIMARY gesture output (absolute, metres)
    ax3 = axes[2]
    ax3.plot(t_diff, raw_peak_m, lw=0.6, alpha=0.35, color='gray', label='raw argmax (no gate)')
    ax3.plot(t_tracked, tracked_dist_m, lw=1.4, color='steelblue', label='gated & smoothed')
    ax3.axhline(target_m, color='gray', lw=0.8, ls='--', label=f'baseline {target_m:.2f} m')
    ax3.set_title("Tracked distance (gesture) — absolute position of dominant motion")
    ax3.set_ylabel("Distance (m)")
    ax3.set_xlabel("Time (s)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Phase displacement — SECONDARY breathing output
    ax4 = axes[3]
    ax4.plot(t_chirps, disp_detrended, lw=0.8, alpha=0.5,
             color='steelblue', label='detrended phase disp.')
    ax4.plot(t_chirps, resp_signal, lw=1.5, color='red',
             label='0.1–3 Hz (breathing)')
    ax4.set_title("Phase displacement (breathing) — sub-mm, valid only when target is still")
    ax4.set_ylabel("Displacement (mm)")
    ax4.set_xlabel("Time (s)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Mean frame-diff profile — shows where motion energy is concentrated
    ax5 = axes[4]
    ax5.plot(range_axis[:max_range_idx], mean_diff_prof, lw=1.0, label='mean |Δamplitude|')
    ax5.axvline(MIN_RANGE_M, color='gray', lw=0.8, ls='--',
                label=f'near-field cutoff ({MIN_RANGE_M} m)')
    ax5.axvline(target_m, color='red', lw=1.2, ls='--',
                label=f'target bin ({target_m:.2f} m)')
    ax5.axvspan(range_axis[win_lo], range_axis[win_hi - 1],
                alpha=0.12, color='red', label='tracking window')
    ax5.set_title("Mean frame-diff profile — motion energy >0.25 m (target selection)")
    ax5.set_xlabel("Distance (m)"); ax5.set_ylabel("Mean |ΔAmplitude|")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    out = path.with_name(path.stem + "_analyzed.png")
    plt.savefig(str(out), dpi=150)
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AcousticProbe FMCW analyser")
    parser.add_argument("wav", help="Path to WAV recording")
    parser.add_argument("--target", type=float, default=None, metavar="M",
                        help="Expected target distance in metres (e.g. 0.5)")
    args = parser.parse_args()
    process(args.wav, target_dist=args.target)
