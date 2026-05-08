#!/usr/bin/env python3
"""
AcousticProbe — Baseline Validation Experiment
================================================
Analyzes bare iPhone recordings at 4 distances × 2 breathing rates
to answer 5 questions:

1. Background noise profile (bare.wav, extract 60s from middle)
2. Effective detection range of bare iPhone
3. Can bare iPhone detect breathing at all?
4. Detection accuracy (detected BPM vs ground truth)
5. Is the breathing frequency range 0.1–3 Hz?

FMCW params: 18–22 kHz, BW=4 kHz, sweep=20ms, Fs=48 kHz, amp=0.5 (-6 dBFS)

Usage:
    python baseline_validation.py          # GUI file picker
    python baseline_validation.py *.wav    # CLI
"""

import numpy as np
import json
import sys
import argparse
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d
from numpy import hanning

# =====================================================================
# FMCW Constants
# =====================================================================
F_START     = 18000
F_END       = 22000
BANDWIDTH   = F_END - F_START   # 4000 Hz
SWEEP_T     = 0.020             # 20 ms
SAMPLE_RATE = 48000
C_SOUND     = 343.0
N_CHIRP     = int(SAMPLE_RATE * SWEEP_T)  # 960 samples
FC          = (F_START + F_END) / 2       # 20 kHz

# =====================================================================
# Ground truth table
# =====================================================================
GROUND_TRUTH = {
    "bare":         {"dist_cm": None, "bpm": None,   "type": "no_person"},
    # 9-file baseline 2026-05-07
    "10cm-8breath": {"dist_cm": 10,   "bpm": 8.0,    "type": "slow"},
    "10cm-16breath":{"dist_cm": 10,   "bpm": 16.0,   "type": "normal"},
    "20cm-7breath": {"dist_cm": 20,   "bpm": 7.0,    "type": "slow"},
    "20cm-16breath":{"dist_cm": 20,   "bpm": 16.0,   "type": "normal"},
    "30cm-7breath": {"dist_cm": 30,   "bpm": 7.0,    "type": "slow"},
    "30cm-16breath":{"dist_cm": 30,   "bpm": 16.0,   "type": "normal"},
    "50cm-7breath": {"dist_cm": 50,   "bpm": 7.0,    "type": "slow"},
    "50cm-16breath":{"dist_cm": 50,   "bpm": 16.0,   "type": "normal"},
    # test_2 batch 2026-05-07 (90s recordings, amp 0.5/0.8 A/B)
    # NOTE: dist_cm placeholder — update if actual recording distance known
    "1_30s 0.5 23breath": {"dist_cm": 20, "bpm": 15.33, "type": "test_amp0.5"},
    "1_30s 0.5 24breath": {"dist_cm": 20, "bpm": 16.00, "type": "test_amp0.5"},
    "1_30s 0.8 24breath": {"dist_cm": 20, "bpm": 16.00, "type": "test_amp0.8"},
}

# =====================================================================
# Signal helpers
# =====================================================================
def bandpass(sig, f_lo, f_hi, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [f_lo / nyq, f_hi / nyq], btype='band')
    return filtfilt(b, a, sig)

def highpass(sig, f_cut, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, f_cut / nyq, btype='high')
    return filtfilt(b, a, sig)

def lowpass(sig, f_cut, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, f_cut / nyq, btype='low')
    return filtfilt(b, a, sig)

def make_chirp(n, f0, f1, fs):
    t = np.arange(n) / fs
    T = n / fs
    return np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * T) * t**2))

def load_wav(path, trim_middle_60s=False):
    """Load WAV file. If trim_middle_60s, extract 60s from the center."""
    sr, data = wavfile.read(str(path))
    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr}"
    if data.dtype != np.float32:
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        else:
            data = data.astype(np.float32)
    if data.ndim > 1:
        data = data[:, 0]
    
    if trim_middle_60s and len(data) > 60 * sr:
        mid = len(data) // 2
        half = 30 * sr
        data = data[mid - half : mid + half]
    
    return data

# =====================================================================
# Target-bin selection — breath-score based
# =====================================================================
def _breath_score_at_bin(ffts_at_bin, fs_slow):
    """
    Compute 'breath score' = peak/median magnitude in 0.1-1 Hz of HP-filtered
    phase displacement at this bin. Higher score = bin contains a more
    pronounced periodic signal in the breathing band.

    This is more robust than frame-diff argmax because:
      - frame-diff picks any motion (sway, multipath leakage, etc.)
      - breath-score specifically rewards periodic motion in 0.1-1 Hz
    """
    phase = np.unwrap(np.angle(ffts_at_bin))
    n = len(phase)
    coeffs = np.polyfit(np.arange(n), phase, 1)
    phase_dt = phase - np.polyval(coeffs, np.arange(n))
    try:
        phase_hp = highpass(phase_dt, 0.1, fs_slow)
    except Exception:
        return 0.0
    nfft_local = max(n, int(fs_slow * 120))
    spec = np.abs(np.fft.rfft(phase_hp, n=nfft_local))
    freqs_local = np.fft.rfftfreq(nfft_local, d=1 / fs_slow)
    mask = (freqs_local >= 0.1) & (freqs_local <= 1.0)
    if not mask.any():
        return 0.0
    return float(spec[mask].max() / (np.median(spec[mask]) + 1e-9))


# =====================================================================
# FMCW Demodulation
# =====================================================================
def fmcw_demod(raw_signal, bg_ffts_mean=None,
               expected_dist_m=None, search_win_m=0.15,
               default_dist_m=None, default_win_m=0.20,
               use_cfar=True, cfar_win=21):
    """
    FMCW demodulation. Returns phase displacement and range profile.

    Parameters
    ----------
    bg_ffts_mean : complex array, optional
        Mean complex FFT from bare recording for coherent subtraction.
    expected_dist_m : float, optional
        Ground-truth target distance hint (m). If provided, restricts
        target-bin search to ±search_win_m around it. Highest priority.
    search_win_m : float
        Search half-window width when expected_dist_m is given.
    default_dist_m : float, optional
        User-configurable default target distance to use when expected_dist_m
        is not provided (e.g., for files not in GROUND_TRUTH). Lower priority
        than expected_dist_m.
    default_win_m : float
        Search half-window width when default_dist_m is used.
    use_cfar : bool
        If True (default), apply CFAR-style local baseline subtraction to the
        breath_score curve before argmax — suppresses bins where periodic
        content is part of a wide clutter spread (HVAC, multipath cluster).
    cfar_win : int
        CFAR averaging window (in bins). Applied with mode='nearest' boundary.

    Target-bin selection is breath-score based: within the candidate range,
    pick the bin whose HP-filtered phase has the highest 0.1-1 Hz peak/median.
    """
    rx = bandpass(raw_signal, F_START - 500, F_END + 500, SAMPLE_RATE)

    chirp_ref = make_chirp(N_CHIRP, F_START, F_END, SAMPLE_RATE)
    search_len = min(3 * N_CHIRP, len(rx))
    corr = np.abs(np.correlate(rx[:search_len], chirp_ref, mode='valid'))
    offset = int(np.argmax(corr))
    rx = rx[offset:]

    n_full = (len(rx) // N_CHIRP) * N_CHIRP
    rx = rx[:n_full].reshape(-1, N_CHIRP)
    drop = int(1.0 / SWEEP_T)
    rx = rx[drop:]
    n_chirps = rx.shape[0]

    tx = np.tile(chirp_ref, (n_chirps, 1))
    mixed = np.apply_along_axis(lambda x: lowpass(x, 5000, SAMPLE_RATE), 1, rx * tx)

    NFFT = N_CHIRP * 4
    hann = hanning(N_CHIRP)
    ffts = np.fft.rfft(mixed * hann, n=NFFT, axis=1)
    mag = np.abs(ffts)
    freq_axis = np.fft.rfftfreq(NFFT, d=1 / SAMPLE_RATE)
    range_axis = freq_axis * C_SOUND * SWEEP_T / (2 * BANDWIDTH)

    range_profile_raw = np.mean(mag, axis=0)
    ffts_mean = np.mean(ffts, axis=0)

    # Background subtraction (used for range-profile visualization only)
    if bg_ffts_mean is not None:
        bg = bg_ffts_mean[:ffts.shape[1]]
        mag_sub = np.abs(ffts - bg[np.newaxis, :])
        range_profile_sub = np.mean(mag_sub, axis=0)
    else:
        range_profile_sub = range_profile_raw

    # Target-bin search range (priority: expected > default > full)
    if expected_dist_m is not None:
        lo = max(0.05, expected_dist_m - search_win_m)
        hi = min(2.0,  expected_dist_m + search_win_m)
        prior_source = "GT"
    elif default_dist_m is not None:
        lo = max(0.05, default_dist_m - default_win_m)
        hi = min(2.0,  default_dist_m + default_win_m)
        prior_source = "default"
    else:
        lo, hi = 0.05, 2.0
        prior_source = "none"
    valid_idx = np.where((range_axis > lo) & (range_axis < hi))[0]

    fs_slow = 1.0 / SWEEP_T

    # Breath-score-based bin selection (replaces frame-diff argmax)
    if len(valid_idx) == 0:
        target_bin = int(np.argmax(np.mean(np.abs(np.diff(mag, axis=0)), axis=0)))
        breath_scores_raw = None
        breath_scores = None
    else:
        breath_scores_raw = np.array(
            [_breath_score_at_bin(ffts[:, b], fs_slow) for b in valid_idx])

        # CFAR-style local baseline subtraction:
        # subtracts each bin's local-mean breath_score so that wide clutter
        # spreads (HVAC, multipath cluster) cancel, leaving sharp single-bin
        # peaks (real targets).
        if use_cfar and len(breath_scores_raw) > cfar_win:
            local_mean = uniform_filter1d(breath_scores_raw,
                                          size=cfar_win, mode='nearest')
            breath_scores = np.maximum(breath_scores_raw - local_mean, 0)
        else:
            breath_scores = breath_scores_raw

        target_bin = int(valid_idx[int(np.argmax(breath_scores))])
    target_dist = range_axis[target_bin]

    # Phase extraction at chosen target bin
    phase = np.unwrap(np.angle(ffts[:, target_bin]))
    wavelength = C_SOUND / FC
    disp_mm = phase * wavelength / (4 * np.pi) * 1000
    disp_mm -= np.polyval(np.polyfit(np.arange(len(disp_mm)), disp_mm, 1),
                          np.arange(len(disp_mm)))

    return {
        "disp_mm": disp_mm,
        "fs_slow": fs_slow,
        "range_profile_raw": range_profile_raw,
        "range_profile_sub": range_profile_sub,
        "range_axis": range_axis,
        "target_bin": target_bin,
        "target_dist_m": target_dist,
        "ffts_mean": ffts_mean,
        "n_chirps": n_chirps,
        "search_window": (lo, hi),
        "prior_source": prior_source,
        "breath_score_at_target": (float(breath_scores.max())
                                    if breath_scores is not None else None),
        "breath_score_at_target_raw": (float(breath_scores_raw.max())
                                        if breath_scores_raw is not None else None),
    }

# =====================================================================
# Spectrum Analysis
# =====================================================================
def analyze_spectrum(disp_mm, fs):
    """Full spectrum, no preset range."""
    N = len(disp_mm)
    nfft = max(N, int(fs * 120))
    fft_mag = np.abs(np.fft.rfft(disp_mm, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1/fs)
    return freqs, fft_mag

def find_spectral_peaks(freqs, fft_mag, min_hz=0.05, max_hz=2.0, n_peaks=5):
    """Find top N peaks in spectrum."""
    mask = (freqs >= min_hz) & (freqs <= max_hz)
    temp = fft_mag.copy()
    temp[~mask] = 0
    median_mag = np.median(fft_mag[mask])
    
    peaks = []
    for _ in range(n_peaks):
        idx = np.argmax(temp)
        if temp[idx] <= 0:
            break
        f = freqs[idx]
        peaks.append({
            "freq_hz": round(f, 4),
            "bpm": round(f * 60, 1),
            "magnitude": float(temp[idx]),
            "confidence": round(float(temp[idx] / (median_mag + 1e-9)), 1),
        })
        suppress = (freqs >= f - 0.03) & (freqs <= f + 0.03)
        temp[suppress] = 0
    return peaks

# =====================================================================
# Process single file
# =====================================================================
def process_file(wav_path, bg_ffts_mean=None, is_bare=False,
                 expected_dist_m=None, default_dist_m=None,
                 default_win_m=0.20, use_cfar=True):
    raw = load_wav(wav_path, trim_middle_60s=is_bare)
    duration = len(raw) / SAMPLE_RATE

    result = fmcw_demod(raw, bg_ffts_mean=bg_ffts_mean,
                        expected_dist_m=expected_dist_m,
                        default_dist_m=default_dist_m,
                        default_win_m=default_win_m,
                        use_cfar=use_cfar)
    disp = result["disp_mm"]
    fs = result["fs_slow"]
    
    # Full spectrum (no filter)
    freqs_raw, fft_raw = analyze_spectrum(disp, fs)
    peaks_raw = find_spectral_peaks(freqs_raw, fft_raw)
    
    # Highpass 0.1 Hz (remove drift)
    disp_hp = highpass(disp, 0.1, fs)
    freqs_hp, fft_hp = analyze_spectrum(disp_hp, fs)
    peaks_hp = find_spectral_peaks(freqs_hp, fft_hp, min_hz=0.1)
    
    # Bandpass 0.1–0.8 Hz (breathing band)
    disp_bp = bandpass(disp, 0.1, 0.8, fs)
    freqs_bp, fft_bp = analyze_spectrum(disp_bp, fs)
    peaks_bp = find_spectral_peaks(freqs_bp, fft_bp, min_hz=0.1, max_hz=0.8)
    
    return {
        "duration_s": duration,
        "target_dist_m": result["target_dist_m"],
        "expected_dist_m": expected_dist_m,
        "search_window": result["search_window"],
        "prior_source": result.get("prior_source"),
        "breath_score_at_target": result["breath_score_at_target"],
        "breath_score_at_target_raw": result.get("breath_score_at_target_raw"),
        "disp_mm": disp,
        "disp_hp": disp_hp,
        "disp_bp": disp_bp,
        "fs_slow": fs,
        "range_profile_raw": result["range_profile_raw"],
        "range_profile_sub": result["range_profile_sub"],
        "range_axis": result["range_axis"],
        "ffts_mean": result["ffts_mean"],
        "freqs_raw": freqs_raw, "fft_raw": fft_raw, "peaks_raw": peaks_raw,
        "freqs_hp": freqs_hp, "fft_hp": fft_hp, "peaks_hp": peaks_hp,
        "freqs_bp": freqs_bp, "fft_bp": fft_bp, "peaks_bp": peaks_bp,
    }

# =====================================================================
# Plotting
# =====================================================================
def plot_all(all_results, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    distances = [10, 20, 30, 50]
    colors = {"normal": "#0EA5E9", "slow": "#10B981", "no_person": "#94A3B8"}
    
    # ── Figure 1: Distance × BreathRate validation matrix ────────────
    fig, axes = plt.subplots(4, 4, figsize=(24, 16))
    fig.suptitle("AcousticProbe — Baseline Validation\n"
                 "Does the detected peak shift with breathing rate? (normal ~16 bpm vs slow ~7-8 bpm)",
                 fontsize=16, fontweight='bold')
    
    col_titles = [
        "Raw Displacement (no filter)",
        "Full Spectrum (0–60 bpm)",
        "After HP 0.1 Hz",
        "HP Spectrum (0–60 bpm) — breathing peak?"
    ]
    
    for row, dist in enumerate(distances):
        # Find normal and slow files for this distance
        normal_key = f"{dist}cm-16breath"
        slow_key = f"{dist}cm-8breath" if dist == 10 else f"{dist}cm-7breath"
        
        for key, color, label in [(normal_key, colors["normal"], "normal"),
                                   (slow_key, colors["slow"], "slow")]:
            if key not in all_results:
                continue
            r = all_results[key]
            gt = GROUND_TRUTH.get(key, {})
            gt_bpm = gt.get("bpm", 0)
            fs = r["fs_slow"]
            t = np.arange(len(r["disp_mm"])) / fs
            t_hp = np.arange(len(r["disp_hp"])) / fs
            
            # Col 0: Raw displacement
            axes[row, 0].plot(t, r["disp_mm"], color=color, linewidth=0.5,
                            alpha=0.7, label=f"{label} (GT={gt_bpm} bpm)")
            
            # Col 1: Raw spectrum zoomed
            bpm = r["freqs_raw"] * 60
            m = bpm <= 60
            axes[row, 1].plot(bpm[m], r["fft_raw"][m], color=color, linewidth=1,
                            alpha=0.7, label=label)
            if gt_bpm:
                axes[row, 1].axvline(x=gt_bpm, color=color, linestyle=':', alpha=0.5)
            
            # Col 2: HP displacement
            axes[row, 2].plot(t_hp, r["disp_hp"], color=color, linewidth=0.5,
                            alpha=0.7, label=label)
            
            # Col 3: HP spectrum
            bpm_hp = r["freqs_hp"] * 60
            m_hp = bpm_hp <= 60
            axes[row, 3].plot(bpm_hp[m_hp], r["fft_hp"][m_hp], color=color,
                            linewidth=1.5, alpha=0.8, label=label)
            if gt_bpm:
                axes[row, 3].axvline(x=gt_bpm, color=color, linestyle='--',
                                   alpha=0.6, linewidth=2,
                                   label=f"GT {gt_bpm} bpm")
        
        # Add bare reference to spectrum plots
        if "bare" in all_results:
            rb = all_results["bare"]
            bpm_b = rb["freqs_hp"] * 60
            m_b = bpm_b <= 60
            axes[row, 3].plot(bpm_b[m_b], rb["fft_hp"][m_b], color=colors["no_person"],
                            linewidth=0.8, alpha=0.4, label="bare (no person)")
        
        # Labels
        for col in range(4):
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].legend(fontsize=7, loc='upper right')
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=11, fontweight='bold')
            if row == 3:
                axes[row, col].set_xlabel("Time (s)" if col % 2 == 0 else "Frequency (bpm)")
        
        axes[row, 0].set_ylabel(f"{dist} cm\n\nmm")
        axes[row, 2].set_ylabel("mm")
        
        # Shade typical breathing range
        for col in [1, 3]:
            axes[row, col].axvspan(6, 20, alpha=0.05, color='green')
    
    plt.tight_layout()
    p1 = out_dir / "validation_matrix.png"
    fig.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {p1}")
    
    # ── Figure 2: Detection accuracy summary ─────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Baseline Validation — Detection Summary", fontsize=16, fontweight='bold')
    
    # Panel A: Detected BPM vs Ground Truth
    ax = axes[0]
    ax.set_title("A) Accuracy: Detected vs Ground Truth BPM")
    ax.plot([5, 20], [5, 20], 'k--', alpha=0.3, label="Perfect match")
    
    for key, r in all_results.items():
        gt = GROUND_TRUTH.get(key, {})
        if gt.get("bpm") is None:
            continue
        gt_bpm = gt["bpm"]
        dist = gt["dist_cm"]
        
        # Best peak in HP spectrum within ±5 bpm of ground truth
        best_peak = None
        best_diff = 999
        for p in r["peaks_hp"]:
            diff = abs(p["bpm"] - gt_bpm)
            if diff < best_diff:
                best_diff = diff
                best_peak = p
        
        if best_peak:
            marker = 'o' if gt.get("type") == "normal" else 's'
            color = plt.cm.viridis(dist / 60)
            ax.scatter(gt_bpm, best_peak["bpm"], c=[color], s=80,
                      marker=marker, edgecolors='black', linewidths=0.5,
                      label=f"{key} (conf={best_peak['confidence']}×)")
    
    ax.set_xlabel("Ground Truth BPM")
    ax.set_ylabel("Detected BPM (nearest HP peak)")
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(4, 22)
    ax.set_ylim(4, 22)
    
    # Panel B: Confidence vs Distance
    ax = axes[1]
    ax.set_title("B) Confidence vs Distance")
    
    for btype, marker, label in [("normal", 'o', "16 bpm"), ("slow", 's', "7-8 bpm")]:
        dists_plot = []
        confs_plot = []
        for key, r in all_results.items():
            gt = GROUND_TRUTH.get(key, {})
            if gt.get("type") != btype:
                continue
            gt_bpm = gt["bpm"]
            # Find peak nearest to GT
            best_conf = 0
            for p in r["peaks_hp"]:
                if abs(p["bpm"] - gt_bpm) < 3:
                    best_conf = max(best_conf, p["confidence"])
            dists_plot.append(gt["dist_cm"])
            confs_plot.append(best_conf)
        
        if dists_plot:
            ax.plot(dists_plot, confs_plot, f'-{marker}', linewidth=2,
                   markersize=8, label=f"{label} (breathing peak)")
    
    # Bare noise floor
    if "bare" in all_results:
        bare_max_conf = max(p["confidence"] for p in all_results["bare"]["peaks_hp"])
        ax.axhline(y=bare_max_conf, color='gray', linestyle='--', alpha=0.5,
                  label=f"Bare noise floor ({bare_max_conf:.1f}×)")
    
    ax.set_xlabel("Distance (cm)")
    ax.set_ylabel("Confidence (×)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([10, 20, 30, 50])
    
    # Panel C: BPM Error vs Distance
    ax = axes[2]
    ax.set_title("C) BPM Error vs Distance")
    
    for btype, marker, label in [("normal", 'o', "16 bpm"), ("slow", 's', "7-8 bpm")]:
        dists_plot = []
        errors_plot = []
        for key, r in all_results.items():
            gt = GROUND_TRUTH.get(key, {})
            if gt.get("type") != btype:
                continue
            gt_bpm = gt["bpm"]
            best_diff = 999
            for p in r["peaks_hp"]:
                diff = abs(p["bpm"] - gt_bpm)
                if diff < best_diff:
                    best_diff = diff
            dists_plot.append(gt["dist_cm"])
            errors_plot.append(best_diff)
        
        if dists_plot:
            ax.plot(dists_plot, errors_plot, f'-{marker}', linewidth=2,
                   markersize=8, label=label)
    
    ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.3, label="±2 bpm tolerance")
    ax.set_xlabel("Distance (cm)")
    ax.set_ylabel("|Detected - Ground Truth| (bpm)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([10, 20, 30, 50])
    
    plt.tight_layout()
    p2 = out_dir / "validation_summary.png"
    fig.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {p2}")
    
    # ── Figure 3: Range profiles ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle("Range Profiles", fontsize=14, fontweight='bold')
    palette = ['#94A3B8', '#F59E0B', '#0EA5E9', '#10B981', '#8B5CF6',
               '#EF4444', '#F97316', '#06B6D4', '#A855F7']
    
    for i, (name, r) in enumerate(all_results.items()):
        ra = r["range_axis"]
        m = ra < 2.0
        c = palette[i % len(palette)]
        
        axes[0].plot(ra[m], 20*np.log10(r["range_profile_raw"][m]+1e-30),
                    color=c, linewidth=1, label=name, alpha=0.7)
        
        if name != "bare":
            axes[1].plot(ra[m], 20*np.log10(r["range_profile_sub"][m]+1e-30),
                        color=c, linewidth=1,
                        label=f"{name} (t={r['target_dist_m']:.2f}m)", alpha=0.7)
            axes[1].axvline(x=r["target_dist_m"], color=c, linestyle=':', alpha=0.3)
    
    axes[0].set_title("Raw Range Profile")
    axes[1].set_title("After Background Subtraction")
    for ax in axes:
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Magnitude (dB)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    p3 = out_dir / "validation_range_profiles.png"
    fig.savefig(p3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {p3}")
    
    # ── Figure 4: Frequency range validation ─────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(24, 8))
    fig.suptitle("Q5: Is breathing in 0.1–3 Hz? Full spectrum (0–5 Hz) for all files",
                 fontsize=14, fontweight='bold')
    
    all_keys = [k for k in all_results if k != "bare"]
    for i, key in enumerate(all_keys[:8]):
        ax = axes[i // 4, i % 4]
        r = all_results[key]
        gt = GROUND_TRUTH.get(key, {})
        gt_bpm = gt.get("bpm", 0)
        
        freqs = r["freqs_hp"]
        fft_mag = r["fft_hp"]
        hz_axis = freqs
        m = hz_axis <= 5.0
        
        ax.plot(hz_axis[m], fft_mag[m], linewidth=1, color='#1E293B')
        ax.axvspan(0.1, 0.8, alpha=0.1, color='green', label="0.1–0.8 Hz")
        ax.axvspan(0.1, 3.0, alpha=0.05, color='blue', label="0.1–3.0 Hz")
        
        if gt_bpm:
            gt_hz = gt_bpm / 60
            ax.axvline(x=gt_hz, color='red', linestyle='--', linewidth=2,
                      label=f"GT: {gt_bpm} bpm ({gt_hz:.2f} Hz)")
        
        ax.set_title(f"{key}", fontsize=10, fontweight='bold')
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    p4 = out_dir / "validation_freq_range.png"
    fig.savefig(p4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {p4}")
    
    return [p1, p2, p3, p4]

# =====================================================================
# Print results
# =====================================================================
def print_results(all_results):
    print()
    print("=" * 80)
    print("  BASELINE VALIDATION — RESULTS")
    print("=" * 80)
    
    # Per-file details
    for key, r in all_results.items():
        gt = GROUND_TRUTH.get(key, {})
        gt_bpm = gt.get("bpm")
        gt_dist = gt.get("dist_cm")
        
        print(f"\n  [{key}]  dist={gt_dist}cm  GT={gt_bpm}bpm  "
              f"duration={r['duration_s']:.1f}s  target_bin={r['target_dist_m']:.3f}m")
        
        print(f"    HP spectrum peaks (>0.1 Hz):")
        for j, p in enumerate(r["peaks_hp"][:5]):
            match = ""
            if gt_bpm and abs(p["bpm"] - gt_bpm) < 2.0:
                match = " ← MATCH GT"
            elif gt_bpm and abs(p["bpm"] - gt_bpm) < 4.0:
                match = " ← close to GT"
            print(f"      {j+1}. {p['bpm']:6.1f} bpm  (conf {p['confidence']:5.1f}×){match}")
    
    # ── Summary table ────────────────────────────────────────────────
    print()
    print("=" * 88)
    print("  DETECTION SUMMARY")
    print("=" * 88)
    print(f"  {'File':<16s} {'Dist':>5s} {'TgtBin':>7s} {'GT BPM':>7s} "
          f"{'Detect':>7s} {'Err':>5s} {'Conf':>6s} {'BareLocal':>10s} {'Detected?':>10s}")
    print(f"  {'─'*16} {'─'*5} {'─'*7} {'─'*7} {'─'*7} {'─'*5} {'─'*6} {'─'*10} {'─'*10}")

    # Helper: bare's confidence near a target frequency
    def bare_conf_near(freq_hz, tol=0.05):
        if "bare" not in all_results:
            return 0.0
        bare = all_results["bare"]
        freqs = bare["freqs_hp"]
        spec  = bare["fft_hp"]
        m_band = (freqs >= 0.1) & (freqs <= 1.0)
        if not m_band.any():
            return 0.0
        median = np.median(spec[m_band])
        m_local = (freqs >= freq_hz - tol) & (freqs <= freq_hz + tol)
        if not m_local.any():
            return 0.0
        return float(spec[m_local].max() / (median + 1e-9))

    bare_max = 0
    if "bare" in all_results:
        bare_max = max(p["confidence"] for p in all_results["bare"]["peaks_hp"])

    for key, r in all_results.items():
        gt = GROUND_TRUTH.get(key, {})
        gt_bpm = gt.get("bpm")
        gt_dist = gt.get("dist_cm")

        if gt_bpm is None:
            print(f"  {key:<16s} {'—':>5s} {r['target_dist_m']*100:>5.0f}cm "
                  f"{'—':>7s} {'—':>7s} {'—':>5s} "
                  f"{bare_max:>5.1f}× {'(global)':>10s} {'NOISE REF':>10s}")
            continue

        # Find best matching peak (nearest GT bpm)
        best_peak = None
        best_diff = 999
        for p in r["peaks_hp"]:
            diff = abs(p["bpm"] - gt_bpm)
            if diff < best_diff:
                best_diff = diff
                best_peak = p

        if best_peak:
            bare_local = bare_conf_near(best_peak["freq_hz"])
            # New criterion: error <2bpm AND conf > 1.5× bare-at-this-freq
            # AND conf > 3.0 absolute floor
            ok_err  = best_diff < 2.0
            ok_conf = (best_peak["confidence"] > max(3.0, 1.5 * bare_local))
            detected = "YES" if (ok_err and ok_conf) else \
                       "MAYBE" if best_diff < 3.0 else "NO"
            print(f"  {key:<16s} {gt_dist:>4d}cm {r['target_dist_m']*100:>5.0f}cm "
                  f"{gt_bpm:>6.1f}  {best_peak['bpm']:>6.1f} "
                  f"{best_diff:>4.1f} {best_peak['confidence']:>5.1f}× "
                  f"{bare_local:>9.1f}× {detected:>10s}")

    print(f"\n  Detection criteria: |error| < 2 bpm AND")
    print(f"                       confidence > max(3.0, 1.5 × bare_local_conf)")
    print(f"  bare_local_conf = bare's HP-spectrum peak/median within ±0.05 Hz of detected freq")
    
    # ── Answer the 5 questions ───────────────────────────────────────
    print()
    print("=" * 80)
    print("  ANSWERS TO 5 QUESTIONS")
    print("=" * 80)
    print("""
  Q1: Background noise profile
      → Extracted 60s from middle of bare.wav. Max HP confidence = {:.1f}×.
      → Any breathing detection must exceed this threshold.

  Q2: Effective detection range
      → Check summary table above. "YES" = detected at that distance.
      → Distance where detection changes from YES to NO = effective range.

  Q3: Can bare iPhone detect breathing?
      → Check if ANY file shows YES in the table above.

  Q4: Accuracy
      → Check Error column. < 2 bpm = accurate detection.

  Q5: Is breathing in 0.1–3 Hz?
      → See validation_freq_range.png.
      → Normal breathing (16 bpm) = 0.27 Hz — YES, within 0.1–3 Hz.
      → Slow breathing (7-8 bpm) = 0.12–0.13 Hz — YES, within 0.1–3 Hz.
      → Recommended range: 0.1–0.8 Hz (covers 6–48 bpm).
""".format(bare_max))

# =====================================================================
# GUI
# =====================================================================
def gui_pick_and_run():
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import threading, platform, subprocess
    
    root = tk.Tk()
    root.title("AcousticProbe — Baseline Validation")
    root.geometry("650x420")
    root.configure(bg="#F8FAFC")
    
    files_list = []
    status_var = tk.StringVar(value="Select all WAV files (bare + distance files).")
    
    tk.Label(root, text="Baseline Validation", font=("Helvetica", 18, "bold"),
             bg="#F8FAFC", fg="#1E293B").pack(pady=(12, 0))
    tk.Label(root, text="Distance × Breathing Rate — 5 Questions",
             font=("Helvetica", 10), bg="#F8FAFC", fg="#64748B").pack(pady=(0, 8))
    
    list_frame = tk.Frame(root, bg="#F8FAFC")
    list_frame.pack(fill='both', expand=True, padx=15, pady=5)
    listbox = tk.Listbox(list_frame, font=("Courier", 11), height=8, selectmode='extended')
    scrollbar = tk.Scrollbar(list_frame, command=listbox.yview)
    listbox.configure(yscrollcommand=scrollbar.set)
    listbox.pack(side='left', fill='both', expand=True)
    scrollbar.pack(side='right', fill='y')
    
    # Target distance + CFAR controls
    ctrl_frame = tk.Frame(root, bg="#F8FAFC")
    ctrl_frame.pack(fill='x', padx=15, pady=(8, 4))
    tk.Label(ctrl_frame, text="Default target dist (cm):",
             bg="#F8FAFC", fg="#1E293B",
             font=("Helvetica", 10)).pack(side='left', padx=(0, 6))
    target_var = tk.StringVar(value="")
    tk.Entry(ctrl_frame, textvariable=target_var, width=6,
             font=("Helvetica", 11)).pack(side='left', padx=(0, 12))
    tk.Label(ctrl_frame, text="(blank = no prior; GROUND_TRUTH overrides)",
             bg="#F8FAFC", fg="#94A3B8",
             font=("Helvetica", 9)).pack(side='left', padx=(0, 12))
    cfar_var = tk.BooleanVar(value=True)
    tk.Checkbutton(ctrl_frame, text="CFAR baseline subtraction",
                   variable=cfar_var, bg="#F8FAFC", fg="#1E293B",
                   font=("Helvetica", 10)).pack(side='left')

    btn_frame = tk.Frame(root, bg="#F8FAFC")
    btn_frame.pack(fill='x', padx=15, pady=5)

    def add_files():
        paths = filedialog.askopenfilenames(
            title="Select WAV files",
            filetypes=[("WAV files", "*.wav"), ("All", "*.*")])
        for p in paths:
            files_list.append(p)
            listbox.insert('end', Path(p).name)
        status_var.set(f"{len(files_list)} file(s) selected.")
    
    def clear_all():
        listbox.delete(0, 'end')
        files_list.clear()
        status_var.set("Select WAV files.")
    
    tk.Button(btn_frame, text="+ Add Files", command=add_files,
              font=("Helvetica", 11), width=12).pack(side='left', padx=3)
    tk.Button(btn_frame, text="Clear All", command=clear_all,
              font=("Helvetica", 11), width=10).pack(side='left', padx=3)
    
    def run_analysis():
        if not files_list:
            messagebox.showwarning("No files", "Please add WAV files.")
            return
        # Parse target distance from GUI
        target_str = target_var.get().strip()
        try:
            default_dist_m = _parse_dist(target_str) if target_str else None
        except ValueError:
            messagebox.showerror("Bad input",
                f"Invalid target distance: '{target_str}'. "
                f"Use e.g. '25', '25cm' or '0.25'.")
            return
        use_cfar = cfar_var.get()

        run_btn.config(state='disabled')
        status_var.set("Processing...")

        def worker():
            try:
                out_dir = Path(files_list[0]).parent / "validation_output"
                out_dir.mkdir(exist_ok=True)

                # Step 1: Process bare first
                bg_ffts = None
                all_results = {}

                for fpath in files_list:
                    short = Path(fpath).stem.lower()
                    if short == "bare":
                        root.after(0, lambda: status_var.set("Processing bare..."))
                        r = process_file(fpath, is_bare=True,
                                         default_dist_m=default_dist_m,
                                         use_cfar=use_cfar)
                        bg_ffts = r["ffts_mean"]
                        all_results["bare"] = r
                        break

                # Step 2: Process all other files
                for i, fpath in enumerate(files_list):
                    short = Path(fpath).stem.lower()
                    if short == "bare":
                        continue
                    root.after(0, lambda f=short, idx=i:
                               status_var.set(f"[{idx+1}/{len(files_list)}] {f}..."))
                    gt = GROUND_TRUTH.get(short, {})
                    exp_d = gt.get("dist_cm")
                    exp_d = exp_d / 100.0 if exp_d is not None else None
                    r = process_file(fpath, bg_ffts_mean=bg_ffts,
                                     expected_dist_m=exp_d,
                                     default_dist_m=default_dist_m,
                                     use_cfar=use_cfar)
                    all_results[short] = r
                
                print_results(all_results)
                root.after(0, lambda: status_var.set("Generating plots..."))
                plot_all(all_results, out_dir)
                
                # Save JSON
                json_out = {}
                for name, r in all_results.items():
                    gt = GROUND_TRUTH.get(name, {})
                    json_out[name] = {
                        "ground_truth_bpm": gt.get("bpm"),
                        "ground_truth_dist_cm": gt.get("dist_cm"),
                        "duration_s": r["duration_s"],
                        "target_dist_m": round(r["target_dist_m"], 3),
                        "peaks_hp": r["peaks_hp"][:5],
                        "peaks_bp": r["peaks_bp"][:5],
                    }
                (out_dir / "validation_results.json").write_text(
                    json.dumps(json_out, indent=2))
                
                root.after(0, lambda: status_var.set(f"Done! → {out_dir.name}/"))
                if platform.system() == "Darwin":
                    subprocess.Popen(["open", str(out_dir)])
                elif platform.system() == "Windows":
                    subprocess.Popen(["explorer", str(out_dir)])
                    
            except Exception as e:
                root.after(0, lambda: messagebox.showerror("Error", str(e)))
                import traceback; traceback.print_exc()
            finally:
                root.after(0, lambda: run_btn.config(state='normal'))
        
        threading.Thread(target=worker, daemon=True).start()
    
    run_btn = tk.Button(root, text="▶  Run Validation", command=run_analysis,
                        font=("Helvetica", 13, "bold"), bg="#0EA5E9", fg="white",
                        width=20, height=1)
    run_btn.pack(pady=10)
    tk.Label(root, textvariable=status_var, font=("Helvetica", 10),
             bg="#F8FAFC", fg="#64748B").pack(pady=(0, 10))
    root.mainloop()

# =====================================================================
# CLI
# =====================================================================
def cli_run(file_paths, default_dist_m=None, default_win_m=0.20,
            use_cfar=True):
    out_dir = Path(file_paths[0]).parent / "validation_output"
    out_dir.mkdir(exist_ok=True)

    bg_ffts = None
    all_results = {}

    # Bare first
    for fpath in file_paths:
        short = Path(fpath).stem.lower()
        if short == "bare":
            print(f"  [bare] Processing (extract middle 60s)...")
            r = process_file(fpath, is_bare=True,
                             default_dist_m=default_dist_m,
                             default_win_m=default_win_m,
                             use_cfar=use_cfar)
            bg_ffts = r["ffts_mean"]
            all_results["bare"] = r
            break
    
    # All others
    for fpath in file_paths:
        short = Path(fpath).stem.lower()
        if short == "bare":
            continue
        gt = GROUND_TRUTH.get(short, {})
        exp_d = gt.get("dist_cm")
        exp_d = exp_d / 100.0 if exp_d is not None else None
        if exp_d is not None:
            prior = f"  prior=GT {exp_d*100:.0f}cm ±15cm"
        elif default_dist_m is not None:
            prior = f"  prior=default {default_dist_m*100:.0f}cm ±{default_win_m*100:.0f}cm"
        else:
            prior = "  prior=none (full search)"
        cfar_tag = " +CFAR" if use_cfar else " no-CFAR"
        print(f"  [{short}] Processing...{prior}{cfar_tag}")
        r = process_file(fpath, bg_ffts_mean=bg_ffts,
                         expected_dist_m=exp_d,
                         default_dist_m=default_dist_m,
                         default_win_m=default_win_m,
                         use_cfar=use_cfar)
        all_results[short] = r
    
    print_results(all_results)
    plot_all(all_results, out_dir)
    
    json_out = {}
    for name, r in all_results.items():
        gt = GROUND_TRUTH.get(name, {})
        json_out[name] = {
            "ground_truth_bpm": gt.get("bpm"),
            "ground_truth_dist_cm": gt.get("dist_cm"),
            "duration_s": r["duration_s"],
            "target_dist_m": round(r["target_dist_m"], 3),
            "expected_dist_m": r.get("expected_dist_m"),
            "search_window_m": r.get("search_window"),
            "prior_source": r.get("prior_source"),
            "breath_score_at_target": r.get("breath_score_at_target"),
            "breath_score_at_target_raw": r.get("breath_score_at_target_raw"),
            "peaks_hp": r["peaks_hp"][:5],
            "peaks_bp": r["peaks_bp"][:5],
        }
    (out_dir / "validation_results.json").write_text(json.dumps(json_out, indent=2))
    print(f"\n  All outputs → {out_dir}/")

# =====================================================================
def _parse_dist(s):
    """Parse '25cm', '0.25', '0.25m' → metres (float)."""
    s = s.strip().lower()
    if s.endswith('cm'):
        return float(s[:-2]) / 100.0
    if s.endswith('mm'):
        return float(s[:-2]) / 1000.0
    if s.endswith('m'):
        return float(s[:-1])
    return float(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AcousticProbe — Baseline Validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('files', nargs='*',
                        help='WAV files to analyze (omit for GUI)')
    parser.add_argument('--target', type=str, default=None,
                        metavar='DIST',
                        help="Default target distance, e.g. '25cm' or '0.25'. "
                             "Used for files NOT in GROUND_TRUTH. "
                             "GROUND_TRUTH entries always override this.")
    parser.add_argument('--target-win', type=float, default=0.20,
                        metavar='M',
                        help='Default search half-window width (m).')
    parser.add_argument('--no-cfar', action='store_true',
                        help='Disable CFAR-style local baseline subtraction.')
    args = parser.parse_args()

    if args.files:
        default_dist_m = _parse_dist(args.target) if args.target else None
        use_cfar = not args.no_cfar

        print("=" * 60)
        print("  AcousticProbe — Baseline Validation")
        print("=" * 60)
        if default_dist_m is not None:
            print(f"  Default target: {default_dist_m*100:.0f} cm "
                  f"(±{args.target_win*100:.0f} cm)")
        else:
            print(f"  Default target: none (full search 5-200 cm)")
        print(f"  CFAR baseline subtraction: {'ON' if use_cfar else 'OFF'}")
        print("=" * 60)

        cli_run(args.files,
                default_dist_m=default_dist_m,
                default_win_m=args.target_win,
                use_cfar=use_cfar)
    else:
        gui_pick_and_run()