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
COLORS = ["#C4C4C4", "#5475BC", "#21D4B9", "#F3432C", "#f39b7f", "#4dbbd5", "#00a087", "#e64b35"]
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


def make_structure_capability_summary(results, labels=None, colors=None, out_dir=None, gt_bpm=None, mode="auto"):
    """
    Compact summary figure for structure-assisted acoustic breathing detection.

    Layout:
        A (top-left): 4 stacked small-multiple bar charts
        B (top-right): amplitude bar + respiratory SNR line
        C (bottom-left): confidence bars with reliable/unreliable zones
        D (bottom-right): paired bar chart for detected BPM vs ground truth

    Compatible with existing main() calls.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # =========================================================
    # Helpers
    # =========================================================
    def _first_finite(*vals, default=np.nan):
        for v in vals:
            try:
                if v is None:
                    continue
                x = float(v)
                if np.isfinite(x):
                    return x
            except Exception:
                continue
        return default

    def _get(r, keys, default=np.nan):
        if not isinstance(r, dict):
            return default
        for k in keys:
            if k in r:
                return _first_finite(r.get(k), default=default)
        return default

    def _gt_for(i, r):
        # 1) per-result priority
        val = _get(
            r,
            ["gt_bpm", "ground_truth_bpm", "manual_bpm", "manual_gt_bpm", "true_bpm"],
            np.nan,
        )
        if np.isfinite(val):
            return val

        # 2) global gt_bpm fallback
        try:
            if isinstance(gt_bpm, dict):
                lab = labels[i] if labels is not None and i < len(labels) else None
                for key in (lab, str(i), i):
                    if key in gt_bpm:
                        return _first_finite(gt_bpm[key], default=np.nan)
            elif isinstance(gt_bpm, (list, tuple, np.ndarray)):
                if i < len(gt_bpm):
                    return _first_finite(gt_bpm[i], default=np.nan)
            else:
                return _first_finite(gt_bpm, default=np.nan)
        except Exception:
            pass
        return np.nan

    def _short_label(x):
        if x is None:
            return "Condition"
        s = str(x).strip()
        s = s.replace("No-subject control", "Bare / control")
        s = s.replace("Bare + breathing", "Bare / control")
        s = s.replace("Tube A + breathing", "Tube A")
        s = s.replace("Tube B + breathing", "Tube B")
        return s

    def _safe_best_idx(arr, prefer="max"):
        arr = np.asarray(arr, dtype=float)
        idx = np.where(np.isfinite(arr))[0]
        if len(idx) == 0:
            return 0
        if prefer == "min":
            return int(idx[np.argmin(arr[idx])])
        return int(idx[np.argmax(arr[idx])])

    def _resolve_save_path(out_dir_like):
        """
        Allow caller to pass either:
            - a directory
            - or a full png path
        """
        if out_dir_like is None:
            save_dir = Path(".")
            save_dir.mkdir(parents=True, exist_ok=True)
            return save_dir / "structure_capability_summary.png"

        if isinstance(out_dir_like, (list, tuple, np.ndarray)):
            out_dir_like = out_dir_like[-1] if len(out_dir_like) else "."

        p = Path(out_dir_like)
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        else:
            p.mkdir(parents=True, exist_ok=True)
            return p / "structure_capability_summary.png"

    # =========================================================
    # Save path / early exit
    # =========================================================
    save_path = _resolve_save_path(out_dir)

    n = len(results)
    if n == 0:
        return save_path

    plot_labels = [
        _short_label(labels[i] if labels is not None and i < len(labels) else f"Condition {i+1}")
        for i in range(n)
    ]

    default_palette = ["#BABABA","#3c5488", "#08B79D",
                       "#F3432C"]

    if colors is not None and len(colors) >= n:
        palette = list(colors)[:n]
    else:
        palette = [default_palette[i % len(default_palette)] for i in range(n)]

    # =========================================================
    # Extract metrics
    # =========================================================
    resp_snr = np.array(
        [_get(r, ["resp_snr_db", "breathing_snr_db", "snr_db", "respiratory_snr_db"]) for r in results],
        dtype=float,
    )
    amp = np.array(
        [_get(r, ["disp_amplitude_mm", "amplitude_mm", "amp_mm", "motion_amplitude_mm", "breathing_amp_mm"]) for r in results],
        dtype=float,
    )
    conf = np.array(
        [_get(r, ["breath_confidence", "confidence", "confidence_ratio", "breathing_confidence", "spectral_confidence", "peak_median_ratio"]) for r in results],
        dtype=float,
    )
    det_bpm = np.array(
        [_get(r, ["resp_rate_bpm", "detected_bpm", "bpm", "estimated_bpm", "dominant_bpm"]) for r in results],
        dtype=float,
    )
    gt_vals = np.array([_gt_for(i, r) for i, r in enumerate(results)], dtype=float)

    baseline_snr = resp_snr[0] if len(resp_snr) and np.isfinite(resp_snr[0]) else np.nan
    snr_improve = resp_snr - baseline_snr if np.isfinite(baseline_snr) else np.full(n, np.nan)

    baseline_amp = amp[0] if len(amp) and np.isfinite(amp[0]) and amp[0] != 0 else np.nan
    amp_ratio = amp / baseline_amp if np.isfinite(baseline_amp) else np.full(n, np.nan)

    bpm_err = np.abs(det_bpm - gt_vals)

    # Reliable / weak logic preserved
    reliable = []
    for i in range(n):
        r = results[i]
        if isinstance(r, dict) and "detection_reliable" in r:
            reliable.append(bool(r["detection_reliable"]))
        else:
            rel = (
                np.isfinite(conf[i]) and conf[i] >= 3.0 and
                np.isfinite(resp_snr[i]) and resp_snr[i] > 0 and
                (not np.isfinite(bpm_err[i]) or bpm_err[i] < 2.0)
            )
            reliable.append(bool(rel))
    reliable = np.array(reliable, dtype=bool)

    # Pick best performer
    score = np.zeros(n, dtype=float)
    for i in range(n):
        score[i] += 10 if reliable[i] else 0
        score[i] += conf[i] if np.isfinite(conf[i]) else 0
        score[i] += 0.2 * resp_snr[i] if np.isfinite(resp_snr[i]) else 0
        if np.isfinite(bpm_err[i]):
            score[i] += max(0, 3 - bpm_err[i])
    best_idx = _safe_best_idx(score, prefer="max")

    # =========================================================
    # Global style (Nature-refined)
    # =========================================================
    VALUE_SIZE = 6.2
    LEGEND_SIZE = 6.0
    NOTE_SIZE = 6.8
    TEXT_COLOR = "#1F1F1F"
    MUTED_COLOR = "#6F6F6F"
    GRID_COLOR = "#D9D9D9"
    GOOD_COLOR = "#4F7A63"
    BAD_COLOR = "#A44C42"
    CONF_THR = 3.0
    BPM_ERR_TARGET = 2.0

    plt.rcParams.update({
        "font.family": "Arial", "font.size": 6.5,
        "axes.titlesize": 8.0, "axes.labelsize": 6.8,
        "xtick.labelsize": 6.2, "ytick.labelsize": 6.2,
        "axes.linewidth": 0.45, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    def _polish(ax, grid_axis="y"):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis=grid_axis, color=GRID_COLOR, alpha=0.22, lw=0.35)
        ax.tick_params(length=2.2, width=0.45, color="#888888")

    def _fmax(arr, default=1.0):
        arr = np.asarray(arr, dtype=float)
        return float(np.nanmax(arr[np.isfinite(arr)])) if np.any(np.isfinite(arr)) else default

    fig = plt.figure(figsize=(13.0, 8.6), dpi=300, facecolor="white")
    gs = fig.add_gridspec(2, 2, left=0.065, right=0.985, top=0.875,
                          bottom=0.10, wspace=0.34, hspace=0.42)

    fig.text(0.065, 0.96, "Structure-assisted acoustic breathing detection",
             ha="left", va="top", fontsize=13, fontweight="bold", color=TEXT_COLOR)
    fig.text(0.065, 0.932,
             "Four-panel summary: metric profile, signal strength, "
             "confidence threshold, and breathing-rate accuracy.",
             ha="left", va="top", fontsize=7.5, color=MUTED_COLOR)

    x = np.arange(n)

    # =========================================================
    # A: Per-metric comparison — 2×2 sub-grid of horizontal bar charts
    # =========================================================
    gs_A = gs[0, 0].subgridspec(2, 2, hspace=0.50, wspace=0.55)

    metric_names_A = ["Resp SNR", "Amplitude", "Confidence", "SNR \u0394"]
    metric_units_A = ["dB", "mm", "\u00d7", "dB"]
    metric_keys_A = [resp_snr, amp, conf, snr_improve]
    fmts_A = ["{:.1f}", "{:.1f}", "{:.1f}", "{:+.1f}"]
    higher_better = [True, True, True, True]

    for j, (mname, munit, mvals, fmt, hb) in enumerate(
            zip(metric_names_A, metric_units_A, metric_keys_A, fmts_A, higher_better)):
        row_j, col_j = divmod(j, 2)
        ax_j = fig.add_subplot(gs_A[row_j, col_j])

        vals = np.array(mvals, dtype=float)
        finite_mask = np.isfinite(vals)

        # Determine bar positions (one bar per condition, vertical stacking)
        y_pos = np.arange(n)
        bar_h_j = max(0.45, 0.80)

        for i in range(n):
            v = vals[i] if finite_mask[i] else 0.0
            ax_j.barh(y_pos[i], v, height=bar_h_j, color=palette[i],
                      alpha=1, edgecolor="white", linewidth=0.5)
            if finite_mask[i]:
                lbl = fmt.format(vals[i])
                # Place label outside bar end
                ha_lbl = "left" if v >= 0 else "right"
                x_lbl = v + abs(v) * 0.03 + 0.2 if v >= 0 else v - 0.2
                ax_j.text(x_lbl, y_pos[i], lbl,
                          ha=ha_lbl, va="center", fontsize=6.0, color=palette[i],
                          fontweight="bold")

        # Star on best
        if finite_mask.any():
            fi = np.where(finite_mask)[0]
            best_i = fi[int(np.nanargmax(vals[fi]))] if hb else fi[int(np.nanargmin(vals[fi]))]
            v_best = vals[best_i]
            x_star = v_best + abs(v_best) * 0.03 + 1.2 if v_best >= 0 else v_best - 1.2
            ax_j.text(x_star, y_pos[best_i], "\u2605",
                      ha="center", va="center", fontsize=8, color=palette[best_i])

        ax_j.set_title(f"{mname} ({munit})", fontsize=8, fontweight="bold", loc="left", pad=3)
        ax_j.set_yticks(y_pos)
        ax_j.set_yticklabels(plot_labels, fontsize=5.5)
        ax_j.invert_yaxis()
        ax_j.spines["top"].set_visible(False)
        ax_j.spines["right"].set_visible(False)
        ax_j.tick_params(axis="both", labelsize=5.5, length=2)
        ax_j.grid(axis="x", color="#CCCCCC", alpha=0.3, lw=0.4)

        # Handle negative values in SNR Δ
        if not hb or (finite_mask.any() and np.nanmin(vals[finite_mask]) < 0):
            ax_j.axvline(0, color="#999999", lw=0.5, zorder=0)

    # Panel A shared title
    fig.text(gs[0, 0].get_position(fig).x0, gs[0, 0].get_position(fig).y1 + 0.01,
             "A  Per-metric comparison (\u2605 = best)",
             ha="left", va="bottom", fontsize=9.5, fontweight="bold", color=TEXT_COLOR)

    # =========================================================
    # B: Signal strength and respiratory SNR
    # =========================================================
    axB = fig.add_subplot(gs[0, 1])
    axB.set_title("B  Signal strength and respiratory SNR",
                  loc="left", pad=6, fontweight="bold", fontsize=9.5)

    amp_plot = np.where(np.isfinite(amp), amp, 0.0)
    axB.bar(x, amp_plot, width=0.30, color=palette, alpha=1,
            edgecolor=palette, linewidth=0.7)
    for i in range(n):
        if np.isfinite(amp[i]):
            axB.text(i, amp[i] + 1.1, f"{amp[i]:.1f}",
                     ha="center", va="bottom", fontsize=5.8, color=TEXT_COLOR)

    axB2 = axB.twinx()
    snr_plot = np.where(np.isfinite(resp_snr), resp_snr, 0.0)
    axB2.plot(x, snr_plot, color=TEXT_COLOR, marker="o", lw=1.2, ms=3.5, zorder=4)
    axB2.axhline(0, color="#8A8A8A", lw=0.5, ls="--", alpha=0.65)
    for i in range(n):
        if np.isfinite(resp_snr[i]):
            axB2.text(i + 0.05, resp_snr[i] + 0.18, f"{resp_snr[i]:.1f}",
                      fontsize=5.8, color=TEXT_COLOR)

    axB.set_ylabel("Amplitude (mm)")
    axB2.set_ylabel("Resp SNR (dB)")
    axB.set_xticks(x)
    axB.set_xticklabels(plot_labels)
    axB.set_ylim(0, _fmax(amp) * 1.30)
    fs_snr = resp_snr[np.isfinite(resp_snr)] if np.any(np.isfinite(resp_snr)) else np.array([0.0])
    axB2.set_ylim(0, float(np.nanmax(fs_snr)) * 1.28)
    _polish(axB)
    axB2.spines["top"].set_visible(False)
    axB.legend(handles=[
        Patch(facecolor="#BBBBBB", edgecolor="#999999", alpha=0.30, label="Amplitude"),
        Line2D([0], [0], color=TEXT_COLOR, marker="o", lw=1.2, ms=3, label="Resp SNR"),
    ], loc="upper left", frameon=False, fontsize=5.8, handlelength=1.2)

    # =========================================================
    # C: Breathing detection confidence — lollipop
    # =========================================================
    axC = fig.add_subplot(gs[1, 0])
    axC.set_title("C  Breathing detection confidence",
                  loc="left", pad=6, fontweight="bold", fontsize=9.5)

    cmax = _fmax(conf, 3.0) + 0.65
    axC.axhspan(CONF_THR, cmax, color=GOOD_COLOR, alpha=0.04, zorder=0)
    axC.axhspan(0, CONF_THR, color=BAD_COLOR, alpha=0.025, zorder=0)
    axC.axhline(CONF_THR, color=BAD_COLOR, lw=0.6, ls="--", alpha=0.85)

    bar_w_C = 0.55 if n <= 4 else max(0.35, 0.80 / n * 2)
    for i in range(n):
        v = conf[i] if np.isfinite(conf[i]) else 0.0
        ok = v >= CONF_THR
        alp = 1 if ok else 0.30
        axC.bar(i, v, width=bar_w_C, color=palette[i], alpha=alp,
                edgecolor=palette[i], linewidth=0.7, zorder=2)
        mark = "\u2713" if ok else "\u2717"
        axC.text(i, v + 0.10, f"{v:.2f}\u00d7 {mark}",
                 ha="center", va="bottom", fontsize=5.6, fontweight="bold",
                 color=GOOD_COLOR if ok else BAD_COLOR)

    axC.text(n - 0.05, CONF_THR + 0.04, "3\u00d7 threshold",
             ha="right", va="bottom", fontsize=5.8, color=BAD_COLOR)
    axC.set_xticks(x)
    axC.set_xticklabels(plot_labels)
    axC.set_ylabel("Confidence (\u00d7)")
    axC.set_ylim(0, cmax)
    _polish(axC)

    # =========================================================
    # D: Breathing rate accuracy — lollipop or boxplot
    # =========================================================
    axD = fig.add_subplot(gs[1, 1])
    axD.set_title("D  Breathing rate accuracy",
                  loc="left", pad=6, fontweight="bold", fontsize=9.5)

    # Check for multi-trial data
    def _get_trials_D(r):
        if not isinstance(r, dict):
            return []
        trials = r.get("_trial_results", None) or r.get("trials", None)
        if isinstance(trials, (list, tuple)) and len(trials) > 0:
            return [t for t in trials if isinstance(t, dict)]
        return []

    trial_errs_D = []
    for i_r, r in enumerate(results):
        trials = _get_trials_D(r)
        if len(trials) >= 3:
            errs = []
            for t in trials:
                ev = t.get("bpm_error", None)
                if ev is not None and np.isfinite(ev):
                    errs.append(float(ev))
                else:
                    dv = t.get("resp_rate_bpm", t.get("peak_bpm", None))
                    gv = t.get("ground_truth_bpm", t.get("gt_bpm", None))
                    if dv is not None and gv is not None and np.isfinite(dv) and np.isfinite(gv):
                        errs.append(abs(float(dv) - float(gv)))
            trial_errs_D.append(errs)
        else:
            trial_errs_D.append([])

    has_boxplot_D = any(len(v) >= 3 for v in trial_errs_D)

    if has_boxplot_D:
        # ── Multi-trial: boxplot + scatter ────────────────────
        data_bp = [v if len(v) else [np.nan] for v in trial_errs_D]
        bp = axD.boxplot(
            data_bp, positions=x, widths=0.36, patch_artist=True,
            showfliers=False,
            medianprops=dict(color=TEXT_COLOR, linewidth=0.8),
            boxprops=dict(linewidth=0.5),
            whiskerprops=dict(linewidth=0.5),
            capprops=dict(linewidth=0.5),
        )
        for ib, box in enumerate(bp["boxes"]):
            box.set_facecolor(palette[ib])
            box.set_alpha(0.20)
            box.set_edgecolor(palette[ib])

        rng = np.random.default_rng(3)
        for isc, errs in enumerate(trial_errs_D):
            if len(errs):
                jit = rng.normal(0, 0.035, len(errs))
                axD.scatter(
                    np.full(len(errs), x[isc]) + jit, errs,
                    s=14, color=palette[isc], alpha=0.65,
                    edgecolor="white", linewidth=0.3, zorder=3,
                )

        axD.axhspan(0, BPM_ERR_TARGET, color=GOOD_COLOR, alpha=0.04, zorder=0)
        axD.axhline(BPM_ERR_TARGET, color=BAD_COLOR, lw=0.6, ls="--", alpha=0.8)
        axD.text(n - 0.05, BPM_ERR_TARGET + 0.08, "2 bpm target",
                 ha="right", va="bottom", fontsize=5.8, color=BAD_COLOR)
        axD.set_ylabel("BPM error, |detected \u2212 GT|")
        fe = [e for v in trial_errs_D for e in v if np.isfinite(e)]
        axD.set_ylim(0, max(6.0, max(fe) + 0.8 if fe else 6.0))
        legend_D = [
            Patch(facecolor="#999999", alpha=0.20, edgecolor="#999999",
                  label="Error distribution"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="#777777", markersize=3, label="Trial"),
            Line2D([0], [0], color=BAD_COLOR, lw=0.6, ls="--",
                   label="2 bpm threshold"),
        ]
    else:
        # ── Single-trial: error lollipop ──────────────────────
        axD.axhspan(0, BPM_ERR_TARGET, color=GOOD_COLOR, alpha=0.04, zorder=0)
        axD.axhline(BPM_ERR_TARGET, color=BAD_COLOR, lw=0.6, ls="--", alpha=0.8)

        for i in range(n):
            e = bpm_err[i] if np.isfinite(bpm_err[i]) else 0.0
            ok_e = e < BPM_ERR_TARGET
            col_e = GOOD_COLOR if ok_e else BAD_COLOR
            axD.vlines(i, 0, e, color=palette[i], lw=1.5, alpha=0.65, zorder=2)
            axD.scatter(i, e, s=32, color=palette[i],
                        edgecolor="white", linewidth=0.5, zorder=3)
            if np.isfinite(bpm_err[i]):
                axD.text(i, e + 0.18, f"{e:.1f}",
                         ha="center", va="bottom", fontsize=5.6,
                         color=col_e, fontweight="bold")

        axD.text(n - 0.05, BPM_ERR_TARGET + 0.06, "2 bpm target",
                 ha="right", va="bottom", fontsize=5.8, color=BAD_COLOR)
        axD.text(n - 0.05, 0.12, "acceptable zone",
                 ha="right", va="bottom", fontsize=5.6, color=GOOD_COLOR)
        axD.set_ylabel("BPM error, |detected \u2212 GT|")
        axD.set_ylim(0, max(6.0, _fmax(bpm_err, 1.0) + 0.9))
        legend_D = [
            Patch(facecolor=GOOD_COLOR, alpha=0.10, edgecolor="none",
                  label="< 2 bpm target zone"),
            Line2D([0], [0], color=BAD_COLOR, lw=0.6, ls="--",
                   label="2 bpm threshold"),
        ]

    axD.set_xticks(x)
    axD.set_xticklabels(plot_labels)
    _polish(axD)
    axD.legend(
        handles=legend_D, loc="upper center",
        bbox_to_anchor=(0.5, -0.16), frameon=False,
        ncol=min(len(legend_D), 3), fontsize=5.8,
        handlelength=1.2, columnspacing=0.8,
    )

    # =========================================================
    # Bottom conclusion text
    # =========================================================
    best_name = plot_labels[best_idx]
    snr_plus = snr_improve[best_idx] if np.isfinite(snr_improve[best_idx]) else np.nan
    amp_ratio_best = amp_ratio[best_idx] if np.isfinite(amp_ratio[best_idx]) else np.nan
    conf_best = conf[best_idx] if np.isfinite(conf[best_idx]) else np.nan
    snr_txt = (f"+{snr_plus:.1f} dB" if np.isfinite(snr_plus) and snr_plus >= 0
               else (f"{snr_plus:.1f} dB" if np.isfinite(snr_plus) else "N/A"))
    amp_txt = f"{amp_ratio_best:.1f}\u00d7" if np.isfinite(amp_ratio_best) else "N/A"
    conf_txt = f"{conf_best:.1f}\u00d7" if np.isfinite(conf_best) else "N/A"

    fig.text(
        0.065, 0.067,
        f"Best performer: {best_name} \u2014 Resp SNR {snr_txt}, "
        f"amplitude {amp_txt} vs bare phone, confidence {conf_txt}",
        fontsize=7.5, fontweight="bold", ha="left", color=TEXT_COLOR,
    )
    fig.text(
        0.065, 0.042,
        "Reliability criteria: confidence \u22653\u00d7, respiratory SNR >0 dB, "
        "and BPM error close to manual count.",
        fontsize=6.5, ha="left", color=MUTED_COLOR,
    )

    # =========================================================
    # Save / close
    # =========================================================
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return save_path


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

    # ── Process every file individually ───────────────────────────────────
    all_results = []
    all_gt = []
    for i, p in enumerate(paths):
        is_control = mode == "control" and i == 0
        g = gt_for_index(gt_bpm, i)
        print(f"Processing [{labels[i]}]: {Path(p).name} ...")
        all_results.append(process_one(p, g, is_control=is_control))
        all_gt.append(g)

    # ── Group by label ────────────────────────────────────────────────────
    from collections import OrderedDict
    groups = OrderedDict()
    for i, (label, r, g) in enumerate(zip(labels, all_results, all_gt)):
        if label not in groups:
            groups[label] = {"results": [], "gt_bpms": [], "paths": []}
        groups[label]["results"].append(r)
        groups[label]["gt_bpms"].append(g)
        groups[label]["paths"].append(paths[i])

    has_repeats = any(len(v["results"]) > 1 for v in groups.values())

    if has_repeats:
        print(f"\n{'═'*70}")
        print("  Detected repeated trials — grouping by label")
        print(f"{'═'*70}")
        for label, v in groups.items():
            print(f"  {label}: {len(v['results'])} trials")
        print()

    # ── Build aggregated results (one per unique label) ───────────────────
    SCALAR_KEYS = [
        "resp_snr_db", "csr_db", "coherence", "breath_confidence",
        "resp_rate_bpm", "peak_bpm", "disp_amplitude_mm",
        "tracking_jitter_cm", "bpm_error", "target_m",
    ]

    unique_labels = list(groups.keys())
    results = []
    agg_gt_bpm = []

    for label in unique_labels:
        trial_results = groups[label]["results"]
        trial_gts = groups[label]["gt_bpms"]
        n_trials = len(trial_results)

        if n_trials == 1:
            agg = dict(trial_results[0])
            agg["_n_trials"] = 1
            agg["_trial_results"] = trial_results
            results.append(agg)
            agg_gt_bpm.append(trial_gts[0])
        else:
            agg = {}
            for key in SCALAR_KEYS:
                vals = [r.get(key) for r in trial_results if r.get(key) is not None]
                if vals:
                    agg[key] = float(np.nanmean(vals))
                    agg[key + "_std"] = float(np.nanstd(vals))
                    agg[key + "_all"] = vals
                else:
                    agg[key] = None
                    agg[key + "_std"] = None
                    agg[key + "_all"] = []

            n_detected = sum(
                1 for r in trial_results if r.get("detection_reliable", False)
            )
            agg["detection_rate"] = 100.0 * n_detected / n_trials
            agg["detection_reliable"] = n_detected > n_trials / 2

            # Copy waveform data from best trial (highest confidence)
            best_idx = 0
            best_conf = -1
            for ti, r in enumerate(trial_results):
                c = r.get("breath_confidence", 0) or 0
                if c > best_conf:
                    best_conf = c
                    best_idx = ti
            best_trial = trial_results[best_idx]
            for key in best_trial:
                if key not in agg:
                    agg[key] = best_trial[key]

            agg["is_control"] = trial_results[0].get("is_control", False)
            agg["_n_trials"] = n_trials
            agg["_trial_results"] = trial_results

            # Use mean of available GTs
            valid_gts = [g for g in trial_gts if g is not None]
            agg_gt_bpm.append(float(np.mean(valid_gts)) if valid_gts else None)

            print(f"  {label}: n={n_trials}, "
                  f"SNR={agg.get('resp_snr_db', 0):.1f}±{agg.get('resp_snr_db_std', 0):.1f} dB, "
                  f"Amp={agg.get('disp_amplitude_mm', 0):.2f}±{agg.get('disp_amplitude_mm_std', 0):.2f} mm, "
                  f"Conf={agg.get('breath_confidence', 0):.1f}±{agg.get('breath_confidence_std', 0):.1f}×, "
                  f"DetRate={agg.get('detection_rate', 0):.0f}%")

            results.append(agg)

    labels = unique_labels
    gt_bpm = agg_gt_bpm
    colors = COLORS[: len(labels)]

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
    # Uses the best trial's WAV path from each group
    # ------------------------------------------------------------
    real_breathing_paths = []

    try:
        from visualize_real_breathing import make_real_breathing_figure

        for i, (label, result) in enumerate(zip(labels, results)):
            if mode == "control" and i == 0:
                continue

            # Get the best trial's wav path from grouped data
            trial_results = result.get("_trial_results", [result])
            best_ti = 0
            best_c = -1
            for ti, tr in enumerate(trial_results):
                c = tr.get("breath_confidence", 0) or 0
                if c > best_c:
                    best_c = c
                    best_ti = ti
            
            # Find the corresponding wav path
            if label in groups:
                wav_path = groups[label]["paths"][best_ti]
            else:
                continue

            real_path = out_dir / f"real_breathing_evidence_{safe_name(label)}.png"
            gt_i = gt_for_index(gt_bpm, i)
            make_real_breathing_figure(
                wav_path, gt_bpm=gt_i, label=label, out_path=real_path,
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
        entry = {}
        for k in keep_keys:
            entry[k] = convert_for_json(r.get(k))
        # Add grouping info
        n_trials = r.get("_n_trials", 1)
        entry["n_trials"] = n_trials
        if n_trials > 1:
            entry["detection_rate"] = r.get("detection_rate")
            for sk in SCALAR_KEYS:
                if sk + "_std" in r:
                    entry[sk + "_std"] = convert_for_json(r[sk + "_std"])
        metrics_json[label] = entry
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
    try:
        plt.show()
    except Exception:
        pass  # Non-interactive or non-main-thread — skip


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