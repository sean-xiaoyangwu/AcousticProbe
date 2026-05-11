import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import re

import numpy as np
from scipy.io import wavfile
from scipy import signal as sig
import matplotlib.pyplot as plt


from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ============================================================
# AcousticProbe — Multi-condition comparison v2
# Supports two experiment modes:
#   1) structure: first file = Bare + human breathing baseline
#   2) control:   first file = No-subject control; remaining files = breathing conditions
# ============================================================

LABELS_DEFAULT = [
    "Bare + breathing",
    "Tube A + breathing",
    "Tube B + breathing",
    "Tube C",
    "Tube D",
]
CONTROL_LABELS_DEFAULT = [
    "No-subject control",
    "Tube A + breathing",
    "Tube B + breathing",
    "Tube C",
    "Tube D",
]
COLORS = ["#6C7A89", "#E74C3C", "#27AE60", "#2980B9", "#8E44AD"]
BG = "#FAFAFA"
TXT = "#2C3E50"
GRID = "#D0D7DE"
MUTED = "#7F8C8D"

GOOD = "#27AE60"
BAD = "#C0392B"
WARN = "#E67E22"

# Breathing detection settings
BPM_SEARCH_MIN = 8.0
BPM_SEARCH_MAX = 30.0
BPM_NORMAL_MIN = 12.0
BPM_NORMAL_MAX = 20.0
CONFIDENCE_THRESHOLD = 3.0
SNR_THRESHOLD_DB = 0.0

# Trim beginning/end to reduce record-start/stop artifacts
TRIM_SECONDS = 3.0


def load_params(p):
    j = Path(p).with_suffix(".json")
    if j.exists():
        return json.loads(j.read_text())
    return {
        "sample_rate_hz": 48000,
        "f_start_hz": 18000,
        "f_end_hz": 22000,
        "sweep_duration_s": 0.020,
    }


def make_chirp(p):
    fs = p["sample_rate_hz"]
    f0 = p["f_start_hz"]
    f1 = p["f_end_hz"]
    T = p["sweep_duration_s"]
    N = int(fs * T)
    B = f1 - f0
    t = np.arange(N) / fs
    return np.sin(2 * np.pi * (f0 * t + (B / (2 * T)) * t**2)).astype(np.float32)


def bp(x, lo, hi, fs, o=5):
    return sig.sosfilt(sig.butter(o, [lo, hi], btype="band", fs=fs, output="sos"), x)


def lp(x, c, fs, o=5):
    return sig.sosfilt(sig.butter(o, c, btype="low", fs=fs, output="sos"), x)


def hp_zp(x, c, fs, o=2):
    return sig.sosfiltfilt(sig.butter(o, c, btype="high", fs=fs, output="sos"), x)


def bp_zp(x, lo, hi, fs, o=2):
    return sig.sosfiltfilt(
        sig.butter(o, [lo, hi], btype="band", fs=fs, output="sos"), x
    )


def ma(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def safe_div(a, b, eps=1e-12):
    return a / (b + eps)


def parse_gt_bpm(value):
    """Accept None, a scalar, a comma-separated string, or a list of per-file BPM values."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if "," in raw:
            vals = []
            for part in raw.split(","):
                part = part.strip()
                vals.append(None if part == "" else float(part))
            return vals
        return float(raw)
    if isinstance(value, (list, tuple)):
        return [None if v is None or v == "" else float(v) for v in value]
    return value


def gt_for_index(gt_bpm, idx):
    """Return the ground-truth BPM for one condition index."""
    gt_bpm = parse_gt_bpm(gt_bpm)
    if gt_bpm is None:
        return None
    if isinstance(gt_bpm, list):
        if idx < len(gt_bpm):
            return gt_bpm[idx]
        return None
    return gt_bpm


def gt_to_json(gt_bpm):
    gt_bpm = parse_gt_bpm(gt_bpm)
    if isinstance(gt_bpm, list):
        return gt_bpm
    return gt_bpm


def gt_label(gt_bpm):
    gt_bpm = parse_gt_bpm(gt_bpm)
    if gt_bpm is None:
        return "None"
    if isinstance(gt_bpm, list):
        return ", ".join("—" if v is None else f"{v:.2f}" for v in gt_bpm)
    return f"{gt_bpm:.2f}"


def smooth_for_plot(x):
    if len(x) < 9:
        return x
    win = min(len(x) - 1 if len(x) % 2 == 0 else len(x), 31)
    if win < 7:
        return x
    if win % 2 == 0:
        win -= 1
    return sig.savgol_filter(x, win, 3)


def integrate_bandpower(freqs, pxx, lo, hi):
    m = (freqs >= lo) & (freqs <= hi)
    if not np.any(m):
        return 0.0
    # NumPy 2.x/3.x compatible replacement for deprecated np.trapz
    return np.trapezoid(pxx[m], freqs[m])


def bandpower_welch(x, fs, lo, hi):
    if len(x) < 8:
        return 0.0
    f, pxx = sig.welch(x, fs=fs, nperseg=min(256, len(x)))
    return integrate_bandpower(f, pxx, lo, hi)


def detect_peaks_for_breathing(resp_signal, fs_c):
    y = smooth_for_plot(resp_signal)
    if len(y) < 10:
        return np.array([], dtype=int), y

    min_distance_s = 1.5
    distance = max(1, int(min_distance_s * fs_c))
    prominence = max(np.std(y) * 0.25, 1e-3)
    peaks, _ = sig.find_peaks(y, distance=distance, prominence=prominence)
    return peaks, y


def trim_signal_arrays(dd, rs, tc, fs_c, trim_s=TRIM_SECONDS):
    trim_n = int(trim_s * fs_c)
    if len(dd) > 2 * trim_n + 20:
        return dd[trim_n:-trim_n], rs[trim_n:-trim_n], tc[trim_n:-trim_n], trim_n
    return dd, rs, tc, 0


def process_one(wav_path, gt_bpm=None, is_control=False):
    wav_path = Path(wav_path)
    p = load_params(wav_path)
    _, data = wavfile.read(str(wav_path))

    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    if data.ndim > 1:
        data = data[:, 0]

    fs = p["sample_rate_hz"]
    f0 = p["f_start_hz"]
    f1 = p["f_end_hz"]
    T = p["sweep_duration_s"]
    N = int(fs * T)
    B = f1 - f0
    c = 343.0
    fc = (f0 + f1) / 2
    drop = int(1.0 / T)
    fs_c = 1.0 / T

    rx = bp(data, f0 - 500, f1 + 500, fs)
    n = (len(rx) // N) * N
    rx = rx[:n].reshape(-1, N)
    tx = np.tile(make_chirp(p), (rx.shape[0], 1))
    rx, tx = rx[drop:], tx[drop:]
    nc = rx.shape[0]

    mixed = np.apply_along_axis(lambda x: lp(x, 5000, fs), 1, rx * tx)
    NFFT = N * 4
    ffts = np.fft.rfft(mixed, n=NFFT, axis=1)
    mag = np.abs(ffts)
    fax = np.fft.rfftfreq(NFFT, d=1 / fs)
    rax = fax * c * T / (2 * B)
    mi = np.searchsorted(rax, 2.0)
    tc = np.arange(nc) * T

    bg = np.abs(np.diff(mag[:, :mi], axis=0))
    mbg = bg.mean(axis=0)
    mn = np.searchsorted(rax, 0.25)
    tb = int(np.argmax(mbg[mn:])) + mn
    tm = rax[tb]
    mp = np.mean(mag, axis=0)

    pi2 = np.argmax(bg, axis=1)
    sm = sig.medfilt(pi2.astype(float), 7)
    sm = ma(sm, 5)
    pd = rax[sm.astype(int)]

    pr = np.angle(ffts[:, tb])
    pu = np.unwrap(pr)
    dm = (pu - pu[0]) * c / (4 * np.pi * fc) * 1000
    dd = sig.detrend(dm)
    dd = hp_zp(dd, 0.05, fs_c)
    rs = bp_zp(dd, 0.10, 3.0, fs_c)

    # Trim metrics to reduce start/stop artifacts, but keep raw full arrays for diagnostics.
    dd_m, rs_m, tc_m, trim_n = trim_signal_arrays(dd, rs, tc, fs_c)

    # Primary metrics
    signal_power = bandpower_welch(dd_m, fs_c, 0.10, 0.60)
    noise_power = bandpower_welch(dd_m, fs_c, 0.60, 3.0)
    rsnr = 10 * np.log10(safe_div(signal_power, noise_power))

    resp_fft = np.abs(np.fft.rfft(rs_m))
    resp_freqs = np.fft.rfftfreq(len(rs_m), d=T)
    bpm_axis = resp_freqs * 60.0
    search_mask = (bpm_axis >= BPM_SEARCH_MIN) & (bpm_axis <= BPM_SEARCH_MAX)

    if np.any(search_mask):
        local_spec = resp_fft[search_mask]
        local_bpm = bpm_axis[search_mask]
        pk_idx = int(np.argmax(local_spec))
        peak_mag = float(local_spec[pk_idx])
        median_mag = float(np.median(local_spec) + 1e-9)
        breath_conf = peak_mag / median_mag
        resp_rate = float(local_bpm[pk_idx])
    else:
        breath_conf = 0.0
        resp_rate = 0.0

    disp_amp = float(np.percentile(rs_m, 97.5) - np.percentile(rs_m, 2.5))
    tracking_jitter = float(np.std(pd) * 100)

    detection_reliable = bool(
        (breath_conf >= CONFIDENCE_THRESHOLD)
        and (rsnr >= SNR_THRESHOLD_DB)
        and (BPM_SEARCH_MIN <= resp_rate <= BPM_SEARCH_MAX)
    )

    # Control condition should not be evaluated as breathing accuracy.
    if is_control:
        bpm_error = None
        detection_reliable = False
    else:
        bpm_error = (
            abs(resp_rate - gt_bpm) if (gt_bpm is not None and resp_rate > 0) else None
        )

    # Diagnostic metrics, saved but not used as main claim.
    target_power = mp[tb]
    clutter = np.concatenate([mp[mn : max(mn, tb - 10)], mp[min(tb + 10, mi) : mi]])
    clutter_power = np.mean(clutter) if len(clutter) > 0 else 1e-9
    csr = 20 * np.log10(safe_div(target_power, clutter_power))
    phase_coh = float(np.abs(np.mean(np.exp(1j * np.diff(pr)))))

    peaks, resp_plot = detect_peaks_for_breathing(rs_m, fs_c)
    peak_times = tc_m[peaks] if len(peaks) else np.array([])
    if len(peak_times) >= 2:
        peak_bpm = 60.0 / np.median(np.diff(peak_times))
    else:
        peak_bpm = None

    return {
        "is_control": bool(is_control),
        "resp_snr_db": float(rsnr),
        "csr_db": float(csr),
        "coherence": float(phase_coh),
        "breath_confidence": float(breath_conf),
        "resp_rate_bpm": float(resp_rate),
        "disp_amplitude_mm": float(disp_amp),
        "tracking_jitter_cm": float(tracking_jitter),
        "bpm_error": None if bpm_error is None else float(bpm_error),
        "detection_reliable": bool(detection_reliable),
        "peak_bpm": None if peak_bpm is None else float(peak_bpm),
        "target_m": float(tm),
        "range_ax": rax,
        "max_idx": int(mi),
        "mean_prof": mp,
        "resp_signal": rs,
        "resp_signal_metric": rs_m,
        "resp_signal_smooth": resp_plot,
        "disp_detrended": dd,
        "disp_detrended_metric": dd_m,
        "t_chirps": tc,
        "t_metric": tc_m,
        "resp_fft": resp_fft,
        "resp_freqs": resp_freqs,
        "peak_indices": peaks,
        "peak_times": peak_times,
        "trim_n": int(trim_n),
    }


# ============================================================
# Evaluation metrics beyond single-trial detection
# ============================================================


def reliability_gate(r, gt_bpm=None, max_bpm_error=4.0):
    """Conservative gate for whether a breathing condition is considered detected."""
    if r.get("is_control", False):
        return False
    ok = (
        r["breath_confidence"] >= CONFIDENCE_THRESHOLD
        and r["resp_snr_db"] >= SNR_THRESHOLD_DB
        and BPM_SEARCH_MIN <= r["resp_rate_bpm"] <= BPM_SEARCH_MAX
        and r["disp_amplitude_mm"] >= 3.0
    )
    if gt_bpm is not None and r.get("bpm_error") is not None:
        ok = ok and (r["bpm_error"] <= max_bpm_error)
    return bool(ok)


def baseline_context(mode):
    """Return wording for baseline-dependent metrics."""
    if mode == "control":
        return {
            "table2_title": "Evaluation Metrics vs No-subject Control",
            "snr_name": "SNR Δ vs Control",
            "amp_name": "Response Ratio vs Control",
            "baseline_label": "no-subject control",
            "snr_json_key": "snr_delta_vs_control_db",
            "amp_json_key": "response_ratio_vs_control",
        }
    return {
        "table2_title": "Evaluation Metrics vs Bare Breathing Baseline",
        "snr_name": "SNR Improvement vs Bare",
        "amp_name": "Amplitude Ratio vs Bare",
        "baseline_label": "bare + breathing baseline",
        "snr_json_key": "snr_improvement_vs_bare_db",
        "amp_json_key": "amplitude_ratio_vs_bare",
    }


def compute_evaluation_metrics(results, labels, gt_bpm=None, mode="structure"):
    """
    Compute baseline comparison metrics.

    Important distinction:
      - structure mode: first file is Bare + breathing, so deltas are structural gains.
      - control mode: first file is No-subject control, so deltas are response increases vs empty scene.
    """
    ctx = baseline_context(mode)
    if not results:
        return [], ctx

    base = results[0]
    base_snr = base["resp_snr_db"]
    base_amp = base["disp_amplitude_mm"]

    # Group trials by exact condition label for future repeated recordings.
    # With only one file per label, detection rate is intentionally N/A.
    groups = {}
    for label, r in zip(labels, results):
        groups.setdefault(label, []).append(r)

    rows = []
    for i, (label, r) in enumerate(zip(labels, results)):
        if i == 0:
            continue
        trials = groups[label]
        n_trials = len(trials)
        if n_trials >= 5:
            n_detected = sum(reliability_gate(t, gt_bpm=None) for t in trials)
            detection_rate = 100.0 * n_detected / n_trials
            detection_rate_txt = f"{detection_rate:.0f}%"
        else:
            detection_rate = None
            detection_rate_txt = "N/A (need ≥5)"

        bpm_error_txt = "—" if r["bpm_error"] is None else f"{r['bpm_error']:.1f}"
        rows.append(
            {
                "condition": label,
                "snr_delta_db": float(r["resp_snr_db"] - base_snr),
                "amp_ratio": float(safe_div(r["disp_amplitude_mm"], base_amp)),
                "detection_rate": detection_rate,
                "detection_rate_txt": detection_rate_txt,
                "directivity_gain_db": None,
                "directivity_gain_txt": "N/A (need 0°/90°)",
                "bpm_error": r["bpm_error"],
                "bpm_error_txt": bpm_error_txt,
                "reliable": bool(r["detection_reliable"]),
                ctx["snr_json_key"]: float(r["resp_snr_db"] - base_snr),
                ctx["amp_json_key"]: float(safe_div(r["disp_amplitude_mm"], base_amp)),
            }
        )
    return rows, ctx


def format_terminal_tables(results, labels, gt_bpm=None, mode="structure"):
    eval_rows, ctx = compute_evaluation_metrics(
        results, labels, gt_bpm=gt_bpm, mode=mode
    )
    lines = []
    lines.append("\nTable 1: Per-condition Detection Metrics")
    lines.append("-" * 102)
    lines.append(
        f"{'Condition':<24} {'Role':<10} {'GT BPM':>7} {'RespSNR':>8} {'Amp':>9} {'Conf':>8} {'BPM':>7} {'BPM err':>8} {'Reliable':>9}"
    )
    lines.append(
        f"{'':<24} {'':<10} {'':>7} {'(dB)':>8} {'(mm)':>9} {'(×)':>8} {'':>7} {'(bpm)':>8} {'':>9}"
    )
    lines.append("-" * 102)
    for i, (label, r) in enumerate(zip(labels, results)):
        role = "control" if r["is_control"] else "breathing"
        err = "—" if r["bpm_error"] is None else f"{r['bpm_error']:.1f}"
        rel = "Yes" if r["detection_reliable"] else "No"
        gt_i = gt_for_index(gt_bpm, i)
        gt_txt = "—" if gt_i is None or r["is_control"] else f"{gt_i:.1f}"
        lines.append(
            f"{label:<24} {role:<10} {gt_txt:>7} {r['resp_snr_db']:>8.1f} {r['disp_amplitude_mm']:>9.1f} "
            f"{r['breath_confidence']:>7.1f}× {r['resp_rate_bpm']:>7.1f} {err:>8} {rel:>9}"
        )

    lines.append(f"\nTable 2: {ctx['table2_title']}")
    lines.append("-" * 104)
    lines.append(
        f"{'Condition':<24} {ctx['snr_name']:<24} {ctx['amp_name']:<24} {'Detection Rate':<18} {'BPM Error':<10} {'Reliable':<9}"
    )
    lines.append(f"{'':<24} {'(dB)':<24} {'(×)':<24} {'(%)':<18} {'(bpm)':<10} {'':<9}")
    lines.append("-" * 104)
    for row in eval_rows:
        rel = "Yes" if row["reliable"] else "No"
        lines.append(
            f"{row['condition']:<24} {row['snr_delta_db']:+.1f}{'':<20} {row['amp_ratio']:.1f}×{'':<20} "
            f"{row['detection_rate_txt']:<18} {row['bpm_error_txt']:<10} {rel:<9}"
        )
    lines.append("-" * 104)
    if mode == "control":
        lines.append(
            "Note: Table 2 compares breathing conditions against the no-subject control. These are not structural gains vs bare phone."
        )
    else:
        lines.append(
            "Note: Table 2 compares structured conditions against Bare + breathing. These are structural gains vs bare phone."
        )
    lines.append(
        "Directivity Gain is intentionally left as N/A until matched 0° and 90° recordings are provided."
    )
    return "\n".join(lines)


def save_text_report(out_path, results, labels, gt_bpm, mode):
    lines = []
    lines.append(f"AcousticProbe — metrics summary ({mode} mode)")
    lines.append("")
    for label, r in zip(labels, results):
        rate = f"{r['resp_rate_bpm']:.1f} bpm" if r["resp_rate_bpm"] > 0 else "N/A"
        peak_rate = f"{r['peak_bpm']:.1f} bpm" if r["peak_bpm"] is not None else "N/A"
        bpm_err_txt = "—" if r["bpm_error"] is None else f"{r['bpm_error']:.1f}"
        role = "false-positive control" if r["is_control"] else "breathing condition"
        lines.append(
            f"{label} ({role}): amp={r['disp_amplitude_mm']:.2f} mm, "
            f"conf={r['breath_confidence']:.2f}x, SNR={r['resp_snr_db']:.2f} dB, "
            f"rate(spec)={rate}, rate(peaks)={peak_rate}, "
            f"reliable={'yes' if r['detection_reliable'] else 'no'}, "
            f"BPM error={bpm_err_txt}"
        )
    if gt_bpm is not None:
        lines.append("")
        lines.append(f"Ground truth BPM for conditions: {gt_label(gt_bpm)}")

    lines.append(format_terminal_tables(results, labels, gt_bpm=gt_bpm, mode=mode))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def make_metrics_dashboard(
    results, labels, colors, out_path, gt_bpm=None, mode="structure"
):
    nc = len(results)
    x = np.arange(nc)

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.15],
        width_ratios=[1.0, 1.0],
        hspace=0.30,
        wspace=0.22,
    )

    if mode == "control":
        title = "AcousticProbe — Breathing Detection with No-Subject Control"
    else:
        title = "AcousticProbe — Structure Comparison for Breathing Detection"
    fig.suptitle(title, fontsize=18, fontweight="bold", color=TXT)

    amp = [r["disp_amplitude_mm"] for r in results]
    conf = [r["breath_confidence"] for r in results]
    bpm_vals = [r["resp_rate_bpm"] for r in results]
    peak_bpm_vals = [
        np.nan if r["peak_bpm"] is None else r["peak_bpm"] for r in results
    ]

    # A. Amplitude
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(BG)
    bars = ax1.bar(x, amp, color=colors, edgecolor="white", linewidth=1.0)
    ax1.set_title(
        "A. Breathing-band response amplitude", fontsize=13, fontweight="bold"
    )
    ax1.set_ylabel("Peak-to-peak displacement (mm)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0)
    ax1.grid(axis="y", alpha=0.25, color=GRID)

    ref = amp[0] if amp[0] != 0 else 1e-9
    for i, (b, v) in enumerate(zip(bars, amp)):
        text = f"{v:.1f} mm"
        if mode == "structure" and i > 0:
            text += f"\n({safe_div(v, ref):.1f}x vs bare)"
        elif mode == "control" and i > 0:
            text += f"\n({safe_div(v, ref):.1f}x vs control)"
        ax1.text(
            b.get_x() + b.get_width() / 2,
            v + max(amp) * 0.02,
            text,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=TXT,
        )
    ax1.set_ylim(0, max(amp) * 1.22)

    # B. Confidence
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(BG)
    bars = ax2.bar(x, conf, color=colors, edgecolor="white", linewidth=1.0)
    ax2.axhline(
        CONFIDENCE_THRESHOLD,
        color="#C0392B",
        linestyle="--",
        linewidth=1.5,
        label=f"Confidence threshold = {CONFIDENCE_THRESHOLD:.0f}x",
    )
    ax2.set_title("B. Spectral confidence", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Peak / median in search band (x)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(axis="y", alpha=0.25, color=GRID)
    ax2.legend(frameon=False, fontsize=9, loc="upper left")
    for b, v, r in zip(bars, conf, results):
        if r["is_control"]:
            status = "control"
        else:
            status = "reliable" if r["detection_reliable"] else "weak"
        ax2.text(
            b.get_x() + b.get_width() / 2,
            v + max(conf) * 0.03,
            f"{v:.1f}x\n{status}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=TXT,
        )
    ax2.set_ylim(0, max(conf) * 1.28)

    # C. BPM comparison
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(BG)
    w = 0.36
    bars1 = ax3.bar(
        x - w / 2,
        bpm_vals,
        width=w,
        color=colors,
        edgecolor="white",
        linewidth=1.0,
        label="Spectrum peak BPM",
    )
    bars2 = ax3.bar(
        x + w / 2,
        peak_bpm_vals,
        width=w,
        color="none",
        edgecolor=colors,
        linewidth=2.0,
        linestyle="--",
        label="Peak interval BPM",
    )
    ax3.axhspan(
        BPM_NORMAL_MIN,
        BPM_NORMAL_MAX,
        color="#27AE60",
        alpha=0.08,
        label="Typical adult resting range",
    )
    gt_vals = [gt_for_index(gt_bpm, i) for i in range(len(results))]
    gt_plot_vals = [v for v in gt_vals if v is not None]
    if gt_plot_vals:
        if len(set(round(v, 3) for v in gt_plot_vals)) == 1:
            ax3.axhline(
                gt_plot_vals[0],
                color="#8E44AD",
                linestyle=":",
                linewidth=2,
                label=f"Ground truth = {gt_plot_vals[0]:.1f} bpm",
            )
        else:
            gt_x = [i for i, v in enumerate(gt_vals) if v is not None]
            gt_y = [v for v in gt_vals if v is not None]
            ax3.scatter(
                gt_x,
                gt_y,
                color="#8E44AD",
                marker="x",
                s=60,
                label="Per-condition ground truth",
            )
    ax3.set_title("C. Candidate breathing rate", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Breaths per minute (bpm)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.grid(axis="y", alpha=0.25, color=GRID)
    ax3.legend(frameon=False, fontsize=9, loc="upper left")
    ymax = max(
        max(bpm_vals + (gt_plot_vals if gt_plot_vals else [0])),
        max([v for v in peak_bpm_vals if not np.isnan(v)] + [20]),
    )
    for i, (b, v) in enumerate(zip(bars1, bpm_vals)):
        text = f"{v:.1f}"
        if results[i]["is_control"]:
            text += "\nFP candidate"
        ax3.text(
            b.get_x() + b.get_width() / 2,
            v + ymax * 0.03,
            text,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=TXT,
        )
    for b, v in zip(bars2, peak_bpm_vals):
        if not np.isnan(v):
            ax3.text(
                b.get_x() + b.get_width() / 2,
                v + ymax * 0.03,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=TXT,
            )

    # D. Summary table and takeaway
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    ax4.set_facecolor(BG)

    headers = ["Condition", "Role", "Amp", "Conf", "SNR", "BPM", "BPM err", "Reliable?"]
    table_data = []
    for label, r in zip(labels, results):
        role = "Control" if r["is_control"] else "Breathing"
        table_data.append(
            [
                label,
                role,
                f"{r['disp_amplitude_mm']:.1f}",
                f"{r['breath_confidence']:.1f}x",
                f"{r['resp_snr_db']:.1f}",
                f"{r['resp_rate_bpm']:.1f}",
                "—" if r["bpm_error"] is None else f"{r['bpm_error']:.1f}",
                "Yes" if r["detection_reliable"] else "No",
            ]
        )

    table = ax4.table(
        cellText=table_data,
        colLabels=headers,
        loc="upper center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.08, 1.55)
    for j in range(len(headers)):
        table[(0, j)].set_facecolor("#D5D8DC")
        table[(0, j)].set_text_props(weight="bold")

    best_amp_idx = int(np.argmax(amp))
    best_conf_idx = int(np.argmax(conf))
    reliable_idx = [
        i
        for i, r in enumerate(results)
        if r["detection_reliable"] and not r["is_control"]
    ]
    best_bpm_idx = None
    if any(gt_for_index(gt_bpm, i) is not None for i in reliable_idx) and reliable_idx:
        best_bpm_idx = min(
            reliable_idx,
            key=lambda i: (
                results[i]["bpm_error"] if results[i]["bpm_error"] is not None else 1e9
            ),
        )

    for i, r in enumerate(results):
        row = i + 1
        if r["is_control"]:
            table[(row, 1)].set_facecolor("#F8F9F9")
            table[(row, 7)].set_facecolor("#FDEDEC")
        if i == best_amp_idx:
            table[(row, 2)].set_facecolor("#D5F5E3")
        if i == best_conf_idx:
            table[(row, 3)].set_facecolor("#D5F5E3")
        if best_bpm_idx is not None and i == best_bpm_idx:
            table[(row, 6)].set_facecolor("#D5F5E3")
        if r["detection_reliable"]:
            table[(row, 7)].set_facecolor("#D5F5E3")

    takeaways = []
    if mode == "control":
        takeaways.append(
            "• First condition is a no-subject false-positive control, not a structure baseline."
        )
        takeaways.append(
            f"• Control candidate BPM is ignored for accuracy because no breathing is present."
        )
    else:
        takeaways.append(
            "• First condition is the real no-structure breathing baseline."
        )
        takeaways.append("• Gains are computed relative to Bare + breathing.")
    takeaways.append(
        f"• Strongest amplitude: {labels[best_amp_idx]} ({amp[best_amp_idx]:.1f} mm)"
    )
    takeaways.append(
        f"• Highest confidence: {labels[best_conf_idx]} ({conf[best_conf_idx]:.1f}x)"
    )
    if best_bpm_idx is not None:
        takeaways.append(
            f"• Best reliable BPM accuracy: {labels[best_bpm_idx]} ({results[best_bpm_idx]['bpm_error']:.1f} bpm error)"
        )
    else:
        takeaways.append("• No reliable BPM accuracy winner under current thresholds.")

    ax4.text(
        0.0,
        0.05,
        "\n".join(takeaways),
        ha="left",
        va="bottom",
        fontsize=9.5,
        color=TXT,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9F9", edgecolor="#CCD1D1"),
    )

    plt.savefig(out_path, dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def make_evaluation_tables_figure(
    results, labels, colors, out_path, gt_bpm=None, mode="structure"
):
    """Dedicated paper-style table figure with single-condition and baseline-comparison metrics."""
    eval_rows, ctx = compute_evaluation_metrics(
        results, labels, gt_bpm=gt_bpm, mode=mode
    )

    fig = plt.figure(figsize=(15, 8), facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.35)
    fig.suptitle(
        "AcousticProbe — Evaluation Metrics", fontsize=18, fontweight="bold", color=TXT
    )

    # Table 1
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    ax1.set_title(
        "Table 1. Per-condition Detection Metrics",
        fontsize=13,
        fontweight="bold",
        loc="left",
        color=TXT,
    )
    headers1 = [
        "Condition",
        "Role",
        "Resp SNR\n(dB)",
        "Amplitude\n(mm)",
        "Confidence\n(×)",
        "BPM",
        "BPM Error\n(bpm)",
        "Reliable?",
    ]
    data1 = []
    for label, r in zip(labels, results):
        data1.append(
            [
                label,
                "Control" if r["is_control"] else "Breathing",
                f"{r['resp_snr_db']:.1f}",
                f"{r['disp_amplitude_mm']:.1f}",
                f"{r['breath_confidence']:.1f}×",
                f"{r['resp_rate_bpm']:.1f}",
                "—" if r["bpm_error"] is None else f"{r['bpm_error']:.1f}",
                "Yes" if r["detection_reliable"] else "No",
            ]
        )
    t1 = ax1.table(
        cellText=data1,
        colLabels=headers1,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    t1.auto_set_font_size(False)
    t1.set_fontsize(9)
    t1.scale(1.0, 1.65)
    for j in range(len(headers1)):
        t1[(0, j)].set_facecolor("#D5D8DC")
        t1[(0, j)].set_text_props(weight="bold")
    for i, r in enumerate(results, start=1):
        if r["is_control"]:
            t1[(i, 1)].set_facecolor("#F8F9F9")
            t1[(i, 7)].set_facecolor("#FDEDEC")
        if r["detection_reliable"]:
            t1[(i, 7)].set_facecolor("#D5F5E3")
        if r["resp_snr_db"] >= SNR_THRESHOLD_DB:
            t1[(i, 2)].set_facecolor("#D5F5E3")
        else:
            t1[(i, 2)].set_facecolor("#FDEDEC")
        if r["breath_confidence"] >= CONFIDENCE_THRESHOLD:
            t1[(i, 4)].set_facecolor("#D5F5E3")
        else:
            t1[(i, 4)].set_facecolor("#FDEDEC")
        if r["bpm_error"] is not None:
            t1[(i, 6)].set_facecolor("#D5F5E3" if r["bpm_error"] <= 4 else "#FADBD8")

    # Table 2
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis("off")
    ax2.set_title(
        f"Table 2. {ctx['table2_title']}",
        fontsize=13,
        fontweight="bold",
        loc="left",
        color=TXT,
    )
    headers2 = [
        "Condition",
        f"{ctx['snr_name']}\n(dB)",
        f"{ctx['amp_name']}\n(×)",
        "Detection Rate\n(%)",
        "Directivity Gain\n(dB)",
        "BPM Error\n(bpm)",
        "Reliable?",
    ]
    data2 = []
    for row in eval_rows:
        data2.append(
            [
                row["condition"],
                f"{row['snr_delta_db']:+.1f}",
                f"{row['amp_ratio']:.1f}×",
                row["detection_rate_txt"],
                row["directivity_gain_txt"],
                row["bpm_error_txt"],
                "Yes" if row["reliable"] else "No",
            ]
        )
    if not data2:
        data2 = [["N/A", "—", "—", "—", "—", "—", "—"]]
    t2 = ax2.table(
        cellText=data2,
        colLabels=headers2,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    t2.auto_set_font_size(False)
    t2.set_fontsize(9)
    t2.scale(1.0, 1.65)
    for j in range(len(headers2)):
        t2[(0, j)].set_facecolor("#D5D8DC")
        t2[(0, j)].set_text_props(weight="bold")
    for i, row in enumerate(eval_rows, start=1):
        t2[(i, 1)].set_facecolor("#D5F5E3" if row["snr_delta_db"] > 0 else "#FADBD8")
        t2[(i, 2)].set_facecolor("#D5F5E3" if row["amp_ratio"] > 1 else "#FADBD8")
        if row["bpm_error"] is not None:
            t2[(i, 5)].set_facecolor("#D5F5E3" if row["bpm_error"] <= 4 else "#FADBD8")
        t2[(i, 6)].set_facecolor("#D5F5E3" if row["reliable"] else "#FDEDEC")

    note = (
        "Control mode: deltas are response increases vs no-subject control, not structural gains vs bare phone."
        if mode == "control"
        else "Structure mode: deltas are structural gains vs Bare + breathing baseline."
    )
    fig.text(
        0.05,
        0.035,
        note
        + " Detection Rate needs ≥5 trials per condition; Directivity Gain needs matched 0°/90° recordings.",
        fontsize=10,
        color=TXT,
        ha="left",
    )

    plt.savefig(out_path, dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def make_breathing_detection_figure(
    results, labels, colors, out_path, gt_bpm=None, mode="structure"
):
    nc = len(results)
    fig, axes = plt.subplots(
        nc, 2, figsize=(17.5, 4.4 * nc), facecolor=BG, squeeze=False
    )

    if mode == "control":
        title = "AcousticProbe — Breathing Detection Evidence with No-Subject Control"
    else:
        title = "AcousticProbe — Breathing Detection Evidence"
    fig.suptitle(title, fontsize=18, fontweight="bold", color=TXT, y=0.995)

    for i, (label, r, color) in enumerate(zip(labels, results, colors)):
        # Left: time-domain evidence
        ax = axes[i, 0]
        ax.set_facecolor(BG)
        t = r["t_metric"]
        ax.plot(
            t,
            r["disp_detrended_metric"],
            color="#BDC3C7",
            lw=1.0,
            alpha=0.8,
            label="Detrended displacement",
        )
        ax.plot(
            t,
            r["resp_signal_smooth"],
            color=color,
            lw=2.2,
            label="Filtered breathing band",
        )
        if len(r["peak_indices"]) > 0:
            ax.scatter(
                t[r["peak_indices"]],
                r["resp_signal_smooth"][r["peak_indices"]],
                s=28,
                color="#C0392B",
                zorder=5,
                label="Detected peaks",
            )
            for px in t[r["peak_indices"]]:
                ax.axvline(px, color="#C0392B", alpha=0.10, lw=1.0)
        ax.set_title(f"{label} — time-domain trace", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Displacement (mm)")
        ax.grid(True, alpha=0.22, color=GRID)
        ax.legend(
            frameon=False,
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
        )

        msg = [
            f"Amp = {r['disp_amplitude_mm']:.1f} mm",
            f"Confidence = {r['breath_confidence']:.1f}x",
            f"SNR = {r['resp_snr_db']:.1f} dB",
        ]
        if r["peak_bpm"] is not None:
            msg.append(f"Peak interval rate ≈ {r['peak_bpm']:.1f} bpm")
        if r["is_control"]:
            msg.append("Role = no-subject control")
        ax.text(
            0.01,
            0.97,
            "\n".join(msg),
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                alpha=0.85,
                edgecolor="#D5D8DC",
            ),
        )

        # Right: frequency-domain evidence
        ax = axes[i, 1]
        ax.set_facecolor(BG)
        bpm_axis = r["resp_freqs"] * 60.0
        spec = r["resp_fft"]
        mask = bpm_axis <= 40
        ax.plot(bpm_axis[mask], spec[mask], color=color, lw=2.2)
        ax.axvspan(
            BPM_SEARCH_MIN,
            BPM_SEARCH_MAX,
            color="#5DADE2",
            alpha=0.10,
            label="Search band",
        )
        ax.axvspan(
            BPM_NORMAL_MIN,
            BPM_NORMAL_MAX,
            color="#27AE60",
            alpha=0.08,
            label="Typical resting range",
        )
        if r["resp_rate_bpm"] > 0:
            ypk = np.interp(r["resp_rate_bpm"], bpm_axis[mask], spec[mask])
            ax.scatter(
                [r["resp_rate_bpm"]],
                [ypk],
                color="#C0392B",
                s=38,
                zorder=5,
                label="Chosen spectral peak",
            )
            ax.axvline(r["resp_rate_bpm"], color="#C0392B", linestyle="--", lw=1.5)
        gt_i = gt_for_index(gt_bpm, i)
        if gt_i is not None and not r["is_control"]:
            ax.axvline(
                gt_i,
                color="#8E44AD",
                linestyle=":",
                lw=2.0,
                label=f"Ground truth = {gt_i:.1f} bpm",
            )

        ax.set_title(
            f"{label} — frequency-domain evidence", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Breathing rate (bpm)")
        ax.set_ylabel("Spectrum magnitude")
        if np.any(mask):
            y_max = np.max(spec[mask])
            ax.set_ylim(0, y_max * 1.22)

        ax.grid(True, alpha=0.22, color=GRID)
        ax.legend(frameon=False, fontsize=8, loc="upper right")

        if r["is_control"]:
            summary = (
                f"False-positive candidate = {r['resp_rate_bpm']:.1f} bpm\n"
                f"Reliable = No\n"
                f"BPM error = —"
            )
        else:
            summary = f"Detected BPM = {r['resp_rate_bpm']:.1f}\nReliable = {'Yes' if r['detection_reliable'] else 'No'}"
            if r["bpm_error"] is not None:
                summary += f"\nBPM error = {r['bpm_error']:.1f}"
            else:
                summary += "\nBPM error = —"
        ax.text(
            0.02,
            0.97,
            summary,
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=8.5,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                alpha=0.88,
                edgecolor="#D5D8DC",
            ),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(out_path, dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def make_diagnostic_figure(results, labels, colors, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=BG)
    fig.suptitle(
        "AcousticProbe — Diagnostic Signal Comparison",
        fontsize=16,
        fontweight="bold",
        color=TXT,
    )

    for ax in axes.flat:
        ax.set_facecolor(BG)
        ax.grid(True, alpha=0.18, color=GRID)

    ax = axes[0, 0]
    for label, r, color in zip(labels, results, colors):
        ax.plot(
            r["t_chirps"],
            r["disp_detrended"],
            color=color,
            lw=0.9,
            alpha=0.8,
            label=label,
        )
    ax.set_title("Raw detrended phase displacement", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (mm)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for label, r, color in zip(labels, results, colors):
        ax.plot(
            r["t_metric"],
            r["resp_signal_smooth"],
            color=color,
            lw=1.8,
            alpha=0.9,
            label=f"{label} ({r['disp_amplitude_mm']:.1f} mm)",
        )
    ax.set_title("Filtered breathing-band waveform", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (mm)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    for label, r, color in zip(labels, results, colors):
        bpm_axis = r["resp_freqs"] * 60.0
        mask = bpm_axis <= 40
        ax.plot(
            bpm_axis[mask],
            r["resp_fft"][mask],
            color=color,
            lw=1.8,
            alpha=0.9,
            label=label,
        )
    ax.axvspan(BPM_NORMAL_MIN, BPM_NORMAL_MAX, color="#27AE60", alpha=0.08)
    ax.axvspan(BPM_SEARCH_MIN, BPM_SEARCH_MAX, color="#5DADE2", alpha=0.08)
    ax.set_title("Breathing spectrum", fontweight="bold")
    ax.set_xlabel("Breathing rate (bpm)")
    ax.set_ylabel("Magnitude")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    for label, r, color in zip(labels, results, colors):
        db = 20 * np.log10(r["mean_prof"][: r["max_idx"]] + 1e-9)
        ax.plot(
            r["range_ax"][: r["max_idx"]],
            db,
            color=color,
            lw=1.8,
            alpha=0.9,
            label=label,
        )
        ax.axvline(r["target_m"], color=color, linestyle=":", lw=1.2, alpha=0.5)
    ax.set_title("Range profile", fontweight="bold")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Magnitude (dB)")
    ax.legend(frameon=False, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def convert_for_json(value):
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    return value


def make_structure_capability_summary(
    results,
    labels,
    colors,
    out_path=None,
    gt_bpm=None,
    mode="structure",
    *args,
    **kwargs,
):
    """Nature-style 2x2 summary figure for structure-assisted breathing detection.

    This function is intentionally visualization-only: it does not change the
    upstream detection or reliability logic. It accepts the same calling style
    as main(): results, labels, colors, out_path, gt_bpm, mode.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from pathlib import Path

    # ---------- robust argument compatibility ----------
    if out_path is None:
        out_path = kwargs.get("out_path", kwargs.get("save_path", None))
    if out_path is None:
        out_path = Path("structure_capability_summary.png")
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(results)
    if n == 0:
        return

    labels = (
        list(labels) if labels is not None else [f"Condition {i+1}" for i in range(n)]
    )
    if len(labels) < n:
        labels += [f"Condition {i+1}" for i in range(len(labels), n)]
    labels = labels[:n]

    def _short_label(s):
        t = str(s)
        low = t.lower()
        if "bare" in low or "control" in low or "no-subject" in low:
            return "Bare / control"
        if "5cm" in low or "tube a" in low:
            return "Tube A"
        if "3cm" in low or "tube b" in low:
            return "Tube B"
        return t.replace(" + breathing", "")

    plot_labels = [_short_label(x) for x in labels]

    default_palette = ["#7B8794", "#E74C3C", "#27AE60", "#8E44AD", "#F39C12"]
    palette = []
    for i in range(n):
        try:
            palette.append(colors[i])
        except Exception:
            palette.append(default_palette[i % len(default_palette)])

    def _num(x, default=np.nan):
        try:
            if x is None:
                return default
            v = float(x)
            if not np.isfinite(v):
                return default
            return v
        except Exception:
            return default

    def _get(r, keys, default=np.nan):
        if not isinstance(r, dict):
            return default
        for k in keys:
            if k in r:
                v = _num(r.get(k), default=np.nan)
                if np.isfinite(v):
                    return v
        return default

    snr = np.array(
        [_get(r, ["resp_snr_db", "snr_db", "rsnr_db"]) for r in results], dtype=float
    )
    amp = np.array(
        [_get(r, ["disp_amplitude_mm", "amplitude_mm", "amp_mm"]) for r in results],
        dtype=float,
    )
    conf = np.array(
        [_get(r, ["breath_confidence", "confidence_x", "confidence"]) for r in results],
        dtype=float,
    )
    det_bpm = np.array(
        [
            _get(r, ["resp_rate_bpm", "detected_bpm", "bpm", "peak_bpm"])
            for r in results
        ],
        dtype=float,
    )
    reliable = np.array(
        [
            bool(r.get("detection_reliable", False)) if isinstance(r, dict) else False
            for r in results
        ],
        dtype=bool,
    )

    # Ground truth: per-result > dict/list/scalar passed from GUI.
    def _gt_for(i):
        r = results[i]
        if isinstance(r, dict):
            for k in ["ground_truth_bpm", "gt_bpm", "manual_bpm"]:
                if k in r:
                    v = _num(r.get(k), default=np.nan)
                    if np.isfinite(v):
                        return v
        if isinstance(gt_bpm, dict):
            for key in [labels[i], plot_labels[i], str(i), i]:
                if key in gt_bpm:
                    return _num(gt_bpm[key], default=np.nan)
            return np.nan
        if isinstance(gt_bpm, (list, tuple, np.ndarray)):
            if i < len(gt_bpm):
                return _num(gt_bpm[i], default=np.nan)
            return np.nan
        return _num(gt_bpm, default=np.nan)

    gt_vals = np.array([_gt_for(i) for i in range(n)], dtype=float)
    bpm_err = np.array(
        [_get(r, ["bpm_error", "bpm_err"], default=np.nan) for r in results],
        dtype=float,
    )
    for i in range(n):
        if (
            not np.isfinite(bpm_err[i])
            and np.isfinite(det_bpm[i])
            and np.isfinite(gt_vals[i])
        ):
            bpm_err[i] = abs(det_bpm[i] - gt_vals[i])

    # Baseline = first bare/control condition, otherwise first condition.
    baseline_idx = 0
    for i, lab in enumerate(labels):
        low = str(lab).lower()
        if "bare" in low or "control" in low or "no-subject" in low:
            baseline_idx = i
            break
    baseline_snr = snr[baseline_idx] if np.isfinite(snr[baseline_idx]) else np.nan
    baseline_amp = amp[baseline_idx] if np.isfinite(amp[baseline_idx]) else np.nan
    snr_improve = (
        snr - baseline_snr if np.isfinite(baseline_snr) else np.full(n, np.nan)
    )

    def _norm_metric(values):
        values = np.asarray(values, dtype=float)
        out = np.zeros_like(values, dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            return out
        vmin, vmax = np.nanmin(values[finite]), np.nanmax(values[finite])
        if abs(vmax - vmin) < 1e-12:
            out[finite] = 0.5
        else:
            out[finite] = (values[finite] - vmin) / (vmax - vmin)
        out[~finite] = 0.0
        return out

    def _fmt(v, kind):
        if not np.isfinite(v):
            return "N/A"
        if kind == "snr":
            return f"{v:.1f} dB"
        if kind == "amp":
            return f"{v:.1f} mm"
        if kind == "conf":
            return f"{v:.2f}×"
        if kind == "imp":
            return f"{v:+.1f} dB"
        if kind == "bpm":
            return f"{v:.1f}"
        return f"{v:.2f}"

    # Best performer uses existing reliability first, then BPM error, confidence, SNR.
    candidates = np.where(reliable)[0]
    if len(candidates) > 0:
        # Among reliable ones, prefer lower BPM error if available, then higher confidence.
        sort_scores = []
        for i in candidates:
            err_score = -bpm_err[i] if np.isfinite(bpm_err[i]) else -999
            conf_score = conf[i] if np.isfinite(conf[i]) else -999
            snr_score = snr[i] if np.isfinite(snr[i]) else -999
            sort_scores.append((err_score, conf_score, snr_score))
        best_idx = int(
            candidates[
                int(np.argmax([s[0] * 1e6 + s[1] * 1e3 + s[2] for s in sort_scores]))
            ]
        )
    else:
        fallback = np.nan_to_num(conf, nan=-np.inf) + 0.05 * np.nan_to_num(snr, nan=0)
        best_idx = int(np.argmax(fallback)) if np.isfinite(fallback).any() else 0

    # ---------- visual style ----------
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans", "sans-serif"],
            "axes.titleweight": "bold",
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "hatch.linewidth": 0.7,
        }
    )

    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), dpi=300, facecolor="white")
    fig.subplots_adjust(
        left=0.09,
        right=0.965,
        top=0.845,      # 给总标题/subtitle 和 A/B 标题留空间
        bottom=0.125,
        wspace=0.42,    # A/B 中间留 legend 空间
        hspace=0.58
    )
    fig.text(
        0.08,
        0.965,
        "Structure-assisted acoustic breathing detection",
        fontsize=15.5,
        weight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.08,
        0.925,
        "Compact comparison of signal magnitude, spectral confidence, and agreement with manually counted breathing rate.",
        fontsize=8.8,
        color="#6B6B6B",
        ha="left",
        va="top",
    )

    def _style_ax(ax, grid_axis="y"):
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
        if grid_axis:
            ax.grid(True, axis=grid_axis, color="#D0D0D0", alpha=0.32, linewidth=0.55)

    # ---------- A. horizontal grouped bar ----------
    axA = axs[0, 0]
    metrics = [
        ("Resp SNR", snr, "snr", ""),
        ("Amplitude", amp, "amp", "///"),
        ("Confidence", conf, "conf", "..."),
        ("SNR improvement", snr_improve, "imp", "xxx"),
    ]
    group_y = np.arange(len(metrics))[::-1]
    bar_h = 0.30
    offsets = np.linspace(-bar_h * (n - 1) / 2, bar_h * (n - 1) / 2, n)
    for m_idx, (m_name, vals, kind, hatch) in enumerate(metrics):
        norm = _norm_metric(vals)
        finite = np.isfinite(vals)
        best_candidates = np.where(finite)[0]
        best_m = (
            int(best_candidates[np.nanargmax(vals[finite])])
            if len(best_candidates)
            else None
        )
        for i in range(n):
            w = float(norm[i]) if np.isfinite(norm[i]) else 0.0
            draw_w = w if w > 0 else (0.025 if np.isfinite(vals[i]) else 0.0)
            y = group_y[m_idx] + offsets[i]
            axA.barh(
                y,
                draw_w,
                height=bar_h * 0.86,
                color=palette[i],
                edgecolor="white",
                linewidth=0.55,
                hatch=hatch,
                alpha=0.92,
            )
            if np.isfinite(vals[i]):
                txt = _fmt(vals[i], kind)
                # Keep text readable and avoid overlap: long bars get inside labels, short bars outside.
                if draw_w >= 0.22:
                    axA.text(
                        draw_w - 0.015,
                        y,
                        txt,
                        ha="right",
                        va="center",
                        fontsize=7.7,
                        color="white",
                        weight="bold",
                    )
                    star_x = min(draw_w + 0.010, 1.03)
                else:
                    axA.text(
                        draw_w + 0.014,
                        y,
                        txt,
                        ha="left",
                        va="center",
                        fontsize=7.7,
                        color="#222222",
                    )
                    star_x = min(draw_w + 0.075, 1.03)
                if best_m is not None and i == best_m:
                    axA.text(
                        star_x,
                        y,
                        "★",
                        ha="left",
                        va="center",
                        fontsize=8.4,
                        color="#222222",
                    )
    axA.set_yticks(group_y)
    axA.set_yticklabels([m[0] for m in metrics])
    axA.set_xlim(0, 1.14)
    axA.set_xlabel("Normalized score")
    axA.set_title("A  Per-metric comparison (★ = best)", loc="left", pad=10)
    _style_ax(axA, grid_axis="x")
    handles = [
        Patch(facecolor=palette[i], edgecolor="none", label=plot_labels[i])
        for i in range(n)
    ]

    axA.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.985, 0.50),   # 放在 A 右侧空白，不要太靠近 B
        frameon=True,
        fancybox=False,                 # 更像论文图的小矩形框
        framealpha=0.95,
        edgecolor="#D8D8D8",
        facecolor="white",
        fontsize=7.2,
        ncol=1,
        borderpad=0.22,
        labelspacing=0.28,
        handlelength=1.15,              # 短一点
        handleheight=0.55,              # 瘦一点
        borderaxespad=0.0,
    )

    # ---------- B. Amplitude + SNR ----------
    axB = axs[0, 1]
    x = np.arange(n)
    b_width = 0.32
    bars = axB.bar(
        x,
        np.nan_to_num(amp, nan=0.0),
        width=b_width,
        color=palette,
        edgecolor="white",
        linewidth=0.55,
        alpha=0.92,
    )
    for i, b in enumerate(bars):
        if np.isfinite(amp[i]):
                axB.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() * 0.84,
                    f"{amp[i]:.1f} mm",
                    ha="center",
                    va="center",
                    fontsize=7.8,
                    color="white",
                    weight="bold",
                )
    axB2 = axB.twinx()
    axB2.plot(x, snr, color="#222222", marker="o", linewidth=1.8, markersize=5)
    for i, v in enumerate(snr):
        if np.isfinite(v):
            if i == n - 1:
                offset = (-6, 10)
                ha = "right"
            else:
                offset = (3, 10)
                ha = "left"
            axB2.annotate(
                f"{v:.1f} dB",
                xy=(x[i], v),
                xytext=offset,
                textcoords="offset points",
                fontsize=8.0,
                color="#222222",
                ha=ha,
                va="bottom",
            )
    axB2.axhline(0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.8)
    axB2.text(
        0.98,
        0,
        "0 dB reference",
        transform=axB2.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=7.6,
        color="#777777",
    )
    axB.set_xticks(x)
    axB.set_xticklabels(plot_labels, rotation=0)
    axB.set_ylabel("Amplitude (mm)")
    axB2.set_ylabel("Resp SNR (dB)")
    amp_f = amp[np.isfinite(amp)]
    y_amp_max = max(62, (np.nanmax(amp_f) * 1.32) if len(amp_f) else 62)
    axB.set_ylim(0, y_amp_max * 1.12)  # 给图例和 SNR 标注留顶部空间
    snr_f = snr[np.isfinite(snr)]
    if len(snr_f):
        axB2.set_ylim(min(0, np.nanmin(snr_f) - 1.0), np.nanmax(snr_f) + 1.0)
    else:
        axB2.set_ylim(0, 1)
    axB.set_title(
        "B  Signal strength and respiratory SNR",
        loc="left",
        fontweight="bold",
        fontsize=10.5,
        pad=12,
    )
    _style_ax(axB, grid_axis="y")
    axB2.spines["top"].set_visible(False)

    # --- B legend: put inside axes to avoid title overlap ---
    legend_handles = [
        Patch(facecolor="#7B8794", edgecolor="none", label="Amplitude (mm)"),
        Line2D([0], [0], color="#222222", marker="o", lw=1.8, label="Resp SNR (dB)"),
        Line2D([0], [0], color="#888888", lw=0.8, linestyle="--", label="0 dB reference"),
    ]

    axB.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),   # 在 B 图内部右上角，不出轴
        frameon=True,
        fancybox=False,
        framealpha=0.92,
        edgecolor="#D8D8D8",
        facecolor="white",
        fontsize=7.2,
        ncol=1,
        borderpad=0.24,
        labelspacing=0.25,
        handlelength=1.35,
        handleheight=0.55,
        borderaxespad=0.15,
    )

    # ---------- C. Confidence threshold ----------
    axC = axs[1, 0]
    conf_f = conf[np.isfinite(conf)]
    y_top = max(3.8, (np.nanmax(conf_f) if len(conf_f) else 3.0) + 0.55)
    axC.axhspan(0, min(3.0, y_top), color="#E74C3C", alpha=0.045, zorder=0)
    axC.axhspan(3.0, y_top, color="#27AE60", alpha=0.055, zorder=0)
    axC.axhline(3.0, color="#C0392B", linestyle="--", linewidth=0.9)
    axC.text(
        0.99,
        3.0 + 0.03,
        "reliable threshold (3×)",
        transform=axC.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=7.8,
        color="#C0392B",
    )
    axC.text(
        0.98,
        y_top - 0.12,
        "Reliable zone",
        transform=axC.get_yaxis_transform(),
        ha="right",
        va="top",
        fontsize=7.8,
        color="#18864B",
    )
    axC.text(
        0.98,
        0.12,
        "Unreliable zone",
        transform=axC.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=7.8,
        color="#C0392B",
    )
    barsC = []
    for i in range(n):
        val = conf[i] if np.isfinite(conf[i]) else 0.0
        alpha = 0.95 if val >= 3.0 else 0.40
        barsC.append(
            axC.bar(
                x[i],
                val,
                width=0.38,
                color=palette[i],
                edgecolor="white",
                linewidth=0.55,
                alpha=alpha,
            )[0]
        )
        if np.isfinite(conf[i]):
            ok = conf[i] >= 3.0
            axC.text(
                x[i],
                val + 0.08,
                f"{conf[i]:.2f}× {'✓' if ok else '✗'}",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="#118847" if ok else "#C0392B",
                weight="bold",
            )
    axC.set_xticks(x)
    axC.set_xticklabels(plot_labels, rotation=0)
    axC.set_ylim(0, y_top)
    axC.set_ylabel("Peak / median in search band (×)")
    axC.set_title("C  Breathing detection confidence", loc="left", pad=6)
    _style_ax(axC, grid_axis="y")

    # ---------- D. Breathing-rate accuracy ----------
    axD = axs[1, 1]

    # TODO: when ≥5 trials per condition, switch to boxplot + scatter overlay.
    def _trial_errors(r):
        if not isinstance(r, dict):
            return []
        trials = r.get("trials", None)
        if not isinstance(trials, (list, tuple)):
            return []
        vals = []
        for t in trials:
            if not isinstance(t, dict):
                continue
            err = _num(t.get("bpm_error", None), default=np.nan)
            if not np.isfinite(err):
                d = _num(
                    t.get("detected_bpm", t.get("resp_rate_bpm", None)), default=np.nan
                )
                g = _num(
                    t.get("ground_truth_bpm", t.get("gt_bpm", None)), default=np.nan
                )
                if np.isfinite(d) and np.isfinite(g):
                    err = abs(d - g)
            if np.isfinite(err):
                vals.append(err)
        return vals

    trial_errs = [_trial_errors(r) for r in results]
    has_multiple = any(len(v) >= 3 for v in trial_errs)
    if has_multiple:
        data = [v if len(v) else [np.nan] for v in trial_errs]
        bp = axD.boxplot(
            data,
            positions=x,
            widths=0.42,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="#222222", linewidth=1.1),
        )
        for i, box in enumerate(bp["boxes"]):
            box.set(
                facecolor=palette[i], alpha=0.30, edgecolor=palette[i], linewidth=0.8
            )
        rng = np.random.default_rng(4)
        for i, vals in enumerate(trial_errs):
            if len(vals):
                jitter = rng.normal(0, 0.035, len(vals))
                axD.scatter(
                    np.full(len(vals), x[i]) + jitter,
                    vals,
                    s=30,
                    color=palette[i],
                    alpha=0.70,
                    edgecolor="white",
                    linewidth=0.4,
                )
        axD.set_ylabel("BPM Error (bpm)")
        finite_err = [e for v in trial_errs for e in v if np.isfinite(e)]
        axD.set_ylim(0, max(5, max(finite_err) + 1 if finite_err else 5))
    else:
        axD.axhspan(
            12, 20, color="#27AE60", alpha=0.10, label="Normal resting range", zorder=0
        )
        axD.text(
            0.985,
            19.7,
            "Normal resting range",
            ha="right",
            va="top",
            fontsize=7.8,
            color="#777777",
            transform=axD.get_yaxis_transform(),
        )
        width = 0.26
        for i in range(n):
            d = det_bpm[i]
            g = gt_vals[i]
            if np.isfinite(d):
                axD.bar(
                    x[i] - width / 2,
                    d,
                    width=width,
                    color=palette[i],
                    edgecolor="white",
                    linewidth=0.55,
                    alpha=0.92,
                )
            if np.isfinite(g):
                axD.bar(
                    x[i] + width / 2,
                    g,
                    width=width,
                    facecolor="white",
                    edgecolor=palette[i],
                    linewidth=1.0,
                    hatch="///",
                )
            if np.isfinite(d) and np.isfinite(g):
                axD.plot(
                    [x[i] - width / 2, x[i] + width / 2],
                    [d, g],
                    color="#9A9A9A",
                    linestyle="--",
                    linewidth=0.9,
                )
                err = abs(d - g)
                axD.text(
                    x[i],
                    max(d, g) + 0.85,
                    f"err: {err:.1f} bpm",
                    ha="center",
                    va="bottom",
                    fontsize=8.3,
                    color="#118847" if err < 2.0 else "#C0392B",
                )
        handlesD = [
            Patch(facecolor="#7B8794", edgecolor="none", label="Detected BPM"),
            Patch(
                facecolor="white",
                edgecolor="#7B8794",
                hatch="///",
                label="Ground truth BPM",
            ),
            Patch(
                facecolor="#27AE60",
                edgecolor="none",
                alpha=0.10,
                label="Normal resting range",
            ),
        ]
        axD.legend(handles=handlesD, loc="upper left", ncol=3, frameon=False)
        y_candidates = []
        y_candidates.extend([v for v in det_bpm if np.isfinite(v)])
        y_candidates.extend([v for v in gt_vals if np.isfinite(v)])
        y_max = max([20] + y_candidates) + 3
        axD.set_ylim(0, y_max)
        axD.set_ylabel("Breathing rate (bpm)")
    axD.set_xticks(x)
    axD.set_xticklabels(plot_labels, rotation=0)
    axD.set_title("D  Breathing rate accuracy", loc="left", pad=6)
    _style_ax(axD, grid_axis="y")

    # ---------- bottom conclusion ----------
    best_name = plot_labels[best_idx]
    best_imp = snr_improve[best_idx] if np.isfinite(snr_improve[best_idx]) else np.nan
    amp_ratio = (
        (amp[best_idx] / baseline_amp)
        if np.isfinite(amp[best_idx])
        and np.isfinite(baseline_amp)
        and baseline_amp != 0
        else np.nan
    )
    best_conf = conf[best_idx] if np.isfinite(conf[best_idx]) else np.nan
    line1 = f"Best performer: {best_name} — Resp SNR {_fmt(best_imp, 'imp')}, amplitude {amp_ratio:.1f}× vs bare phone, confidence {best_conf:.1f}×"
    if not np.isfinite(amp_ratio):
        line1 = f"Best performer: {best_name} — Resp SNR {_fmt(best_imp, 'imp')}, confidence {best_conf:.1f}×"
    fig.text(
        0.08,
        0.075,
        line1,
        fontsize=9.5,
        weight="bold",
        ha="left",
        va="bottom",
        color="#222222",
    )
    fig.text(
        0.08,
        0.048,
        "Reliability: spectral confidence ≥3×, respiratory SNR >0 dB, BPM error <2 bpm vs manual count.",
        fontsize=8.5,
        ha="left",
        va="bottom",
        color="#777777",
    )

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(paths, labels=None, gt_bpm=None, mode="structure"):
    gt_bpm = parse_gt_bpm(gt_bpm)
    mode = mode.lower().strip()
    if mode not in {"structure", "control"}:
        raise ValueError("mode must be 'structure' or 'control'")

    if labels is None:
        labels = (
            CONTROL_LABELS_DEFAULT[: len(paths)]
            if mode == "control"
            else LABELS_DEFAULT[: len(paths)]
        )
    colors = COLORS[: len(paths)]

    results = []
    for i, p in enumerate(paths):
        is_control = mode == "control" and i == 0
        print(f"Processing [{labels[i]}]: {Path(p).name} ...")
        results.append(process_one(p, gt_for_index(gt_bpm, i), is_control=is_control))

    def safe_name(s):
        s = str(s).strip()
        s = re.sub(r"[^\w\-]+", "_", s)
        s = re.sub(r"_+", "_", s)
        return s.strip("_")[:40] or "condition"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # folder name = mode + structure names + timestamp
    label_part = "_vs_".join(safe_name(x) for x in labels)
    folder_name = f"{safe_name(mode)}_{label_part}_{ts}"

    # save to current running directory, not WAV folder
    out_dir = Path.cwd() / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dash_path = out_dir / "metrics_dashboard.png"
    breath_path = out_dir / "breathing_detection.png"
    diag_path = out_dir / "signal_diagnostics.png"
    eval_table_path = out_dir / "evaluation_tables.png"
    summary_path = out_dir / "structure_capability_summary.png"
    json_path = out_dir / "metrics.json"
    txt_path = out_dir / "metrics.txt"

    make_metrics_dashboard(results, labels, colors, dash_path, gt_bpm=gt_bpm, mode=mode)
    make_breathing_detection_figure(
        results, labels, colors, breath_path, gt_bpm=gt_bpm, mode=mode
    )
    make_diagnostic_figure(results, labels, colors, diag_path)
    make_structure_capability_summary(
        results,
        labels,
        colors,
        summary_path,
        gt_bpm=gt_bpm,
        mode=mode,
    )

    # ------------------------------------------------------------
    # Extra: real breathing evidence visualization for each breathing condition
    # ------------------------------------------------------------
    real_breathing_paths = []

    try:
        from visualize_real_breathing import make_real_breathing_figure

        for i, (wav_path, label, result) in enumerate(zip(paths, labels, results)):
            # In control mode, skip the first no-subject control file
            if mode == "control" and i == 0:
                continue

            real_path = out_dir / f"real_breathing_evidence_{safe_name(label)}.png"

            gt_i = gt_for_index(gt_bpm, i)

            make_real_breathing_figure(
                wav_path,
                gt_bpm=gt_i,
                label=label,
                out_path=real_path,
            )

            real_breathing_paths.append(real_path)

    except Exception as e:
        import traceback

        print("\n[ERROR] Failed to generate real breathing evidence figures:")
        traceback.print_exc()

    make_evaluation_tables_figure(
        results, labels, colors, eval_table_path, gt_bpm=gt_bpm, mode=mode
    )

    eval_rows, eval_ctx = compute_evaluation_metrics(
        results, labels, gt_bpm=gt_bpm, mode=mode
    )

    metrics_json = {}
    keep_keys = [
        "is_control",
        "resp_snr_db",
        "breath_confidence",
        "resp_rate_bpm",
        "peak_bpm",
        "disp_amplitude_mm",
        "bpm_error",
        "detection_reliable",
        "csr_db",
        "coherence",
        "tracking_jitter_cm",
        "target_m",
    ]
    for label, r in zip(labels, results):
        metrics_json[label] = {k: convert_for_json(r[k]) for k in keep_keys}
    metrics_json["_evaluation_metrics"] = {
        "context": eval_ctx,
        "rows": eval_rows,
        "notes": [
            "Detection Rate is N/A unless there are at least 5 trials for the same condition label.",
            "Directivity Gain is N/A unless matched 0-degree and 90-degree recordings are provided.",
        ],
    }
    metrics_json["_metadata"] = {
        "mode": mode,
        "gt_bpm": gt_to_json(gt_bpm),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "snr_threshold_db": SNR_THRESHOLD_DB,
        "bpm_search_min": BPM_SEARCH_MIN,
        "bpm_search_max": BPM_SEARCH_MAX,
        "trim_seconds": TRIM_SECONDS,
    }
    json_path.write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
    save_text_report(txt_path, results, labels, gt_bpm, mode)

    print("\nSaved outputs:")
    print(f"  {dash_path}")
    print(f"  {breath_path}")
    print(f"  {diag_path}")
    print(f"  {eval_table_path}")
    print(f"  {summary_path}")

    for p in real_breathing_paths:
        print(f"  {p}")

    print(f"  {json_path}")
    print(f"  {txt_path}")

    print("\nQuick interpretation:")
    for label, r in zip(labels, results):
        role = "control" if r["is_control"] else "breathing"
        bpm_err = "—" if r["bpm_error"] is None else f"{r['bpm_error']:.1f}"
        print(
            f"  {label:<22} role={role:<9} amp={r['disp_amplitude_mm']:.1f} mm | "
            f"conf={r['breath_confidence']:.1f}x | SNR={r['resp_snr_db']:.1f} dB | "
            f"BPM={r['resp_rate_bpm']:.1f} | err={bpm_err} | "
            f"reliable={'yes' if r['detection_reliable'] else 'no'}"
        )

    print(format_terminal_tables(results, labels, gt_bpm=gt_bpm, mode=mode))

    for img_path in [
        dash_path,
        breath_path,
        diag_path,
        eval_table_path,
        summary_path,
    ] + real_breathing_paths:
        if img_path.exists():
            img = plt.imread(img_path)
            plt.figure(figsize=(12, 8), facecolor=BG)
            plt.imshow(img)
            plt.axis("off")
            plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AcousticProbe multi-condition comparison"
    )
    parser.add_argument("wavs", nargs="*", help="WAV files")
    parser.add_argument(
        "--bpm",
        type=str,
        default=None,
        help="Ground-truth BPM. Use a single value or comma-separated per-file values, e.g. 15.65,14.19,15.32",
    )
    parser.add_argument(
        "--labels", nargs="*", default=None, help="Custom labels for each condition"
    )
    parser.add_argument(
        "--mode",
        choices=["structure", "control"],
        default="structure",
        help=(
            "structure: first file is Bare + breathing baseline; "
            "control: first file is No-subject control and is excluded from BPM accuracy"
        ),
    )
    args = parser.parse_args()

    if len(args.wavs) >= 2:
        if args.labels is not None and len(args.labels) != len(args.wavs):
            raise ValueError("Number of --labels must match number of wav files")
        main(args.wavs, args.labels, args.bpm, mode=args.mode)
    else:
        dl = Path.home() / "Downloads"
        ws = sorted(dl.glob("fmcw_*.wav"), key=lambda p: p.stat().st_mtime)[-3:]
        if len(ws) < 2:
            print(f"Found {len(ws)} WAVs. Need >=2.")
            print("Usage examples:")
            print(
                "  python compare_conditions.py no_subject.wav tubeA.wav tubeB.wav --mode control --bpm 15"
            )
            print(
                "  python compare_conditions.py bare_breathing.wav tubeA.wav tubeB.wav --mode structure --bpm 15"
            )
            sys.exit(1)
        labels = (
            CONTROL_LABELS_DEFAULT[: len(ws)]
            if args.mode == "control"
            else LABELS_DEFAULT[: len(ws)]
        )
        print("Auto-detected (oldest→newest):")
        for i, w in enumerate(ws):
            print(f"  [{labels[i]}] {w.name}")
        main([str(w) for w in ws], labels, args.bpm, mode=args.mode)
