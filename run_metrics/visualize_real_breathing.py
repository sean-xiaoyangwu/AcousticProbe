"""
AcousticProbe — Real Breathing Evidence Visualizer

Goal:
    Generate a clean figure that makes the detected breathing visually obvious.

What it shows:
    A) Time-domain breathing waveform + fitted periodic breathing template
    B) Cycle-folded waveform: each detected breathing cycle overlaid together
    C) Frequency-domain evidence: detected BPM vs optional ground-truth BPM

Usage:
    python visualize_real_breathing.py tubeA.wav --bpm 15 --label "Tube A + breathing"

Optional:
    python visualize_real_breathing.py tubeA.wav --bpm 15 --label "Tube A" --out tubeA_real_breathing.png

Requirement:
    Put this file in the same folder as compare_conditions.py.
    compare_conditions.py should be the newer version that defines process_one().
"""

import argparse
import importlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig


BG = "#FAFAFA"
TXT = "#2C3E50"
GRID = "#D0D7DE"
MAIN = "#E74C3C"
AUX = "#6C7A89"
GOOD = "#27AE60"
PURPLE = "#8E44AD"


def moving_average(x, w):
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def fit_sinusoid(t, y, bpm):
    """
    Fit y(t) = a*sin(wt) + b*cos(wt) + c at the detected breathing BPM.
    Returns fitted signal, R^2, and correlation.
    """
    f = bpm / 60.0
    w = 2 * np.pi * f
    X = np.column_stack([
        np.sin(w * t),
        np.cos(w * t),
        np.ones_like(t),
    ])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coef

    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    if np.std(y) < 1e-12 or np.std(y_hat) < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(y, y_hat)[0, 1])

    return y_hat, float(r2), corr


def fold_cycles(t, y, bpm, trim_first_last=True, points_per_cycle=160):
    """
    Fold the waveform into repeated breathing cycles using detected BPM.
    This is visually useful: true breathing should show repeated cycle shapes.
    """
    period = 60.0 / bpm
    t0 = t[0]
    t_rel = t - t0
    n_cycles = int(np.floor(t_rel[-1] / period))

    cycles = []
    phase_grid = np.linspace(0, 1, points_per_cycle)

    start_k = 1 if trim_first_last and n_cycles > 3 else 0
    end_k = n_cycles - 1 if trim_first_last and n_cycles > 3 else n_cycles

    for k in range(start_k, end_k):
        start = k * period
        end = (k + 1) * period
        m = (t_rel >= start) & (t_rel < end)
        if np.sum(m) < 8:
            continue
        phase = (t_rel[m] - start) / period
        yy = y[m]
        yy = yy - np.mean(yy)
        amp = np.percentile(yy, 95) - np.percentile(yy, 5)
        if amp > 1e-9:
            yy = yy / amp
        interp = np.interp(phase_grid, phase, yy)
        cycles.append(interp)

    if len(cycles) == 0:
        return phase_grid, np.empty((0, points_per_cycle))

    return phase_grid, np.vstack(cycles)


def estimate_cycle_stability(cycles):
    """
    Measure how repeatable folded breathing cycles are.
    Returns mean pairwise correlation with the average cycle.
    """
    if cycles.shape[0] < 2:
        return 0.0
    mean_cycle = np.mean(cycles, axis=0)
    vals = []
    for c in cycles:
        if np.std(c) < 1e-9 or np.std(mean_cycle) < 1e-9:
            continue
        vals.append(np.corrcoef(c, mean_cycle)[0, 1])
    if not vals:
        return 0.0
    return float(np.mean(vals))


def classify_real_breathing(result, template_corr, cycle_stability, gt_bpm=None):
    """
    A conservative visual-evidence gate.
    This is not a medical classifier; it is for experiment visualization.
    """
    bpm = result["resp_rate_bpm"]
    bpm_ok = 8 <= bpm <= 30
    conf_ok = result["breath_confidence"] >= 3.0
    snr_ok = result["resp_snr_db"] >= 0.0
    amp_ok = result["disp_amplitude_mm"] >= 3.0
    periodic_ok = template_corr >= 0.35 or cycle_stability >= 0.30

    gt_ok = True
    if gt_bpm is not None:
        gt_ok = abs(bpm - gt_bpm) <= 4.0

    passed = bpm_ok and conf_ok and snr_ok and amp_ok and periodic_ok and gt_ok
    return passed, {
        "BPM in breathing range": bpm_ok,
        "confidence ≥ 3×": conf_ok,
        "SNR ≥ 0 dB": snr_ok,
        "amplitude ≥ 3 mm": amp_ok,
        "periodicity visible": periodic_ok,
        "close to ground truth": gt_ok,
    }


def make_real_breathing_figure(wav_path, gt_bpm=None, label=None, out_path=None):
    import compare_conditions

    label = label or Path(wav_path).stem
    result = compare_conditions.process_one(wav_path, gt_bpm=gt_bpm, is_control=False)

    t = np.asarray(result["t_metric"])
    y = np.asarray(result["resp_signal_smooth"])
    y_raw = np.asarray(result["resp_signal_metric"])

    bpm = float(result["resp_rate_bpm"])
    period = 60.0 / bpm if bpm > 0 else np.nan

    y_template, r2, corr = fit_sinusoid(t, y, bpm)
    phase, cycles = fold_cycles(t, y, bpm)
    cycle_stability = estimate_cycle_stability(cycles)
    passed, checks = classify_real_breathing(result, corr, cycle_stability, gt_bpm)

    if out_path is None:
        out_path = Path(wav_path).with_name(f"{Path(wav_path).stem}_real_breathing_evidence.png")
    else:
        out_path = Path(out_path)

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], width_ratios=[1.3, 1.0], hspace=0.35, wspace=0.25)

    verdict = "REAL BREATHING-LIKE PERIODIC MOTION" if passed else "WEAK / NOT RELIABLE"
    verdict_color = GOOD if passed else "#C0392B"
    fig.suptitle(
        f"AcousticProbe — Real Breathing Evidence: {label}",
        fontsize=18,
        fontweight="bold",
        color=TXT,
        y=0.98,
    )

    # ------------------------------------------------------------------
    # A. Time-domain waveform + periodic template
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(BG)
    ax1.plot(t, y_raw, color="#BDC3C7", lw=1.0, alpha=0.60, label="Filtered breathing-band signal")
    ax1.plot(t, y, color=MAIN, lw=2.0, label="Smoothed breathing waveform")
    ax1.plot(t, y_template, color=TXT, lw=2.2, linestyle="--", label=f"Best-fit periodic template at {bpm:.1f} bpm")

    # Draw expected cycle boundaries from detected BPM
    if np.isfinite(period) and period > 0:
        k = 0
        while t[0] + k * period <= t[-1]:
            ax1.axvline(t[0] + k * period, color=GOOD, alpha=0.12, lw=1.0)
            k += 1

    ax1.set_title("A. Time-domain evidence: repeated low-frequency motion", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Phase-derived displacement (mm)")
    ax1.grid(True, alpha=0.22, color=GRID)
    ax1.legend(frameon=False, fontsize=9, loc="upper right")

    summary = (
        f"Detected BPM = {bpm:.1f}\n"
        f"Period ≈ {period:.2f} s/cycle\n"
        f"Template correlation = {corr:.2f}\n"
        f"Template R² = {r2:.2f}\n"
        f"Verdict: {verdict}"
    )
    ax1.text(
        0.015, 0.965, summary,
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color=TXT,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=verdict_color, linewidth=1.8, alpha=0.92),
    )

    # ------------------------------------------------------------------
    # B. Cycle-folded evidence
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(BG)

    if cycles.shape[0] > 0:
        for c in cycles:
            ax2.plot(phase, c, color=MAIN, lw=0.8, alpha=0.18)
        mean_cycle = np.mean(cycles, axis=0)
        ax2.plot(phase, mean_cycle, color=TXT, lw=3.0, label="Average folded cycle")
        ax2.text(
            0.02, 0.95,
            f"Folded cycles = {cycles.shape[0]}\nCycle stability = {cycle_stability:.2f}",
            transform=ax2.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#D5D8DC", alpha=0.90),
        )
    else:
        ax2.text(0.5, 0.5, "Not enough cycles to fold", ha="center", va="center", transform=ax2.transAxes)

    ax2.set_title("B. Cycle-folded evidence: cycles should repeat", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Normalized breathing cycle phase")
    ax2.set_ylabel("Normalized displacement")
    ax2.grid(True, alpha=0.22, color=GRID)
    ax2.legend(frameon=False, fontsize=9, loc="upper right")

    # ------------------------------------------------------------------
    # C. Frequency-domain evidence + reliability checks
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(BG)

    bpm_axis = np.asarray(result["resp_freqs"]) * 60.0
    spec = np.asarray(result["resp_fft"])
    mask = bpm_axis <= 40
    ax3.plot(bpm_axis[mask], spec[mask], color=MAIN, lw=2.2, label="Breathing spectrum")
    ax3.axvspan(8, 30, color="#5DADE2", alpha=0.10, label="Search band: 8–30 bpm")
    ax3.axvspan(12, 20, color=GOOD, alpha=0.08, label="Typical resting range")
    ax3.axvline(bpm, color="#C0392B", linestyle="--", lw=2.0, label=f"Detected = {bpm:.1f} bpm")
    if gt_bpm is not None:
        ax3.axvline(gt_bpm, color=PURPLE, linestyle=":", lw=2.2, label=f"Ground truth = {gt_bpm:.1f} bpm")

    if np.any(mask):
        ax3.set_ylim(0, np.max(spec[mask]) * 1.18)

    ax3.set_title("C. Frequency-domain evidence: dominant breathing-rate peak", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Breathing rate (bpm)")
    ax3.set_ylabel("Spectrum magnitude")
    ax3.grid(True, alpha=0.22, color=GRID)
    ax3.legend(frameon=False, fontsize=8, loc="upper right")

    bpm_error_txt = "—" if gt_bpm is None else f"{abs(bpm - gt_bpm):.1f} bpm"
    check_lines = [
        f"Amp = {result['disp_amplitude_mm']:.1f} mm",
        f"SNR = {result['resp_snr_db']:.1f} dB",
        f"Confidence = {result['breath_confidence']:.1f}×",
        f"BPM error = {bpm_error_txt}",
        "",
        "Reliability checks:",
    ]
    for name, ok in checks.items():
        check_lines.append(f"{'✓' if ok else '✗'} {name}")

    ax3.text(
        0.02, 0.97,
        "\n".join(check_lines),
        transform=ax3.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=TXT,
        bbox=dict(boxstyle="round,pad=0.40", facecolor="white", edgecolor="#D5D8DC", alpha=0.92),
    )

    plt.savefig(out_path, dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")
    print("Summary:")
    print(f"  Detected BPM: {bpm:.2f}")
    print(f"  Amp: {result['disp_amplitude_mm']:.2f} mm")
    print(f"  SNR: {result['resp_snr_db']:.2f} dB")
    print(f"  Confidence: {result['breath_confidence']:.2f}x")
    print(f"  Template corr: {corr:.2f}")
    print(f"  Cycle stability: {cycle_stability:.2f}")
    print(f"  Verdict: {verdict}")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Visualize real breathing evidence from one FMCW WAV file")
    parser.add_argument("wav", help="Input WAV file")
    parser.add_argument("--bpm", type=float, default=None, help="Optional ground-truth BPM")
    parser.add_argument("--label", type=str, default=None, help="Condition label shown in the title")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    make_real_breathing_figure(args.wav, gt_bpm=args.bpm, label=args.label, out_path=args.out)


if __name__ == "__main__":
    main()
