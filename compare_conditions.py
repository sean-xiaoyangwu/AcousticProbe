"""
AcousticProbe — Multi-condition comparison with proper metrics & polished visuals

Usage:
    python compare_conditions.py <bare.wav> <tube3mm.wav> <tube4mm.wav>
    python compare_conditions.py   (auto-detects newest WAVs in ~/Downloads)
"""

import sys, json, numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy import signal as sig
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors

# ── Config ────────────────────────────────────────────────────────────────────

COLORS = ["#6C7A89", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD"]
BG     = "#FAFAFA"

# ── DSP ───────────────────────────────────────────────────────────────────────

def load_params(p):
    j = p.with_suffix(".json")
    if j.exists(): return json.loads(j.read_text())
    return {"sample_rate_hz":48000,"f_start_hz":18000,"f_end_hz":22000,"sweep_duration_s":0.020}

def make_chirp(p):
    fs,f0,f1,T = p["sample_rate_hz"],p["f_start_hz"],p["f_end_hz"],p["sweep_duration_s"]
    N=int(fs*T); B=f1-f0; t=np.arange(N)/fs
    return np.sin(2*np.pi*(f0*t+(B/(2*T))*t**2)).astype(np.float32)

def bp(x,lo,hi,fs,o=5):
    return sig.sosfilt(sig.butter(o,[lo,hi],btype='band',fs=fs,output='sos'),x)
def lp(x,c,fs,o=5):
    return sig.sosfilt(sig.butter(o,c,btype='low',fs=fs,output='sos'),x)
def hp_zp(x,c,fs,o=2):
    return sig.sosfiltfilt(sig.butter(o,c,btype='high',fs=fs,output='sos'),x)
def bp_zp(x,lo,hi,fs,o=2):
    return sig.sosfiltfilt(sig.butter(o,[lo,hi],btype='band',fs=fs,output='sos'),x)
def ma(x,w):
    return np.convolve(x,np.ones(w),'valid')/w

# ── Process ───────────────────────────────────────────────────────────────────

def process_one(wav_path):
    wav_path = Path(wav_path)
    p = load_params(wav_path)
    _,data = wavfile.read(str(wav_path))
    if data.dtype!=np.float32: data=data.astype(np.float32)/np.iinfo(data.dtype).max
    if data.ndim>1: data=data[:,0]

    fs,f0,f1,T = p["sample_rate_hz"],p["f_start_hz"],p["f_end_hz"],p["sweep_duration_s"]
    N=int(fs*T); B=f1-f0; c=343.0; fc=(f0+f1)/2
    drop=int(1.0/T); fs_c=1.0/T

    rx=bp(data,f0-500,f1+500,fs)
    n=(len(rx)//N)*N; rx=rx[:n].reshape(-1,N)
    tx=np.tile(make_chirp(p),(rx.shape[0],1))
    rx,tx=rx[drop:],tx[drop:]; nc=rx.shape[0]

    mixed=np.apply_along_axis(lambda x:lp(x,5000,fs),1,rx*tx)
    NFFT=N*4
    ffts=np.fft.rfft(mixed,n=NFFT,axis=1)
    mag=np.abs(ffts)
    fax=np.fft.rfftfreq(NFFT,d=1/fs)
    rax=fax*c*T/(2*B)
    mi=np.searchsorted(rax,2.0)
    tc=np.arange(nc)*T

    bg=np.abs(np.diff(mag[:,:mi],axis=0))
    mbg=bg.mean(axis=0)
    mn=np.searchsorted(rax,0.25)
    tb=int(np.argmax(mbg[mn:]))+mn
    tm=rax[tb]
    mp=np.mean(mag,axis=0)

    pi2=np.argmax(bg,axis=1)
    sm=sig.medfilt(pi2.astype(float),7); sm=ma(sm,5)
    pd=rax[sm.astype(int)]

    pr=np.angle(ffts[:,tb])
    pu=np.unwrap(pr)
    dm=(pu-pu[0])*c/(4*np.pi*fc)*1000
    dd=sig.detrend(dm); dd=hp_zp(dd,0.05,fs_c)
    rs=bp_zp(dd,0.1,3.0,fs_c)

    # Metrics
    nr=dd-rs
    rsnr=10*np.log10(np.var(rs)/(np.var(nr)+1e-12))

    tp=mp[tb]
    cb=np.concatenate([mp[mn:max(mn,tb-10)],mp[min(tb+10,mi):mi]])
    cp=np.mean(cb) if len(cb)>0 else 1e-9
    csr=20*np.log10(tp/(cp+1e-9))

    coh=np.abs(np.mean(np.exp(1j*np.diff(pr))))

    rf=np.abs(np.fft.rfft(rs))
    rfq=np.fft.rfftfreq(len(rs),d=T)
    rmk=(rfq>=0.1)&(rfq<=1.0)
    bc=np.max(rf[rmk])/(np.median(rf[rmk])+1e-9) if rmk.any() else 0
    rr=0.0
    if rmk.any() and bc>2: rr=rfq[rmk][np.argmax(rf[rmk])]*60
    da=np.percentile(rs,97.5)-np.percentile(rs,2.5)
    jt=np.std(pd)*100

    # Motion signal (wider band, 0.1–10 Hz for hand/body movement)
    nyq_motion = min(10.0, fs_c/2 - 0.5)
    ms = bp_zp(dd, 0.1, nyq_motion, fs_c) if fs_c > 21 else rs
    motion_amp = np.percentile(ms, 97.5) - np.percentile(ms, 2.5)

    # ── New motion metrics ──────────────────────────────────────────────
    # Velocity RMS (mm/s): first derivative of displacement, then RMS
    vel = np.diff(ms) * fs_c  # mm/s
    velocity_rms = np.sqrt(np.mean(vel**2))

    # Motion dynamic range (mm): full peak-to-peak of wideband displacement
    motion_dyn_range = np.max(ms) - np.min(ms)

    # Motion bandwidth (Hz): -3dB bandwidth of motion signal power spectrum
    ms_fft = np.abs(np.fft.rfft(ms))**2
    ms_freqs = np.fft.rfftfreq(len(ms), d=T)
    ms_fft_norm = ms_fft / (np.max(ms_fft) + 1e-12)
    above_3db = ms_freqs[ms_fft_norm >= 0.5]  # -3dB = half power
    motion_bw = (above_3db[-1] - above_3db[0]) if len(above_3db) >= 2 else 0.0

    # Motion SNR (dB): wideband motion signal vs residual noise
    ms_noise = dd - ms
    motion_snr = 10*np.log10(np.var(ms)/(np.var(ms_noise)+1e-12))

    # Activity type detection: check if dominant energy is in breathing band vs motion band
    full_fft = np.abs(np.fft.rfft(dd))
    full_freqs = np.fft.rfftfreq(len(dd), d=T)
    breath_mask = (full_freqs >= 0.15) & (full_freqs <= 0.5)
    motion_mask = (full_freqs >= 0.5) & (full_freqs <= 8.0)
    breath_energy = np.sum(full_fft[breath_mask]**2) if breath_mask.any() else 0
    motion_energy = np.sum(full_fft[motion_mask]**2) if motion_mask.any() else 0
    activity_ratio = motion_energy / (breath_energy + 1e-12)

    return {"resp_snr_db":rsnr,"csr_db":csr,"coherence":coh,
            "breath_confidence":bc,"resp_rate":rr,"disp_amplitude_mm":da,
            "tracking_jitter_cm":jt,"target_m":tm,"range_ax":rax,"max_idx":mi,
            "mean_prof":mp,"resp_signal":rs,"disp_detrended":dd,
            "t_chirps":tc,"resp_fft":rf,"resp_freqs":rfq,
            "motion_signal":ms,"motion_amplitude_mm":motion_amp,
            "activity_ratio":activity_ratio,
            "velocity_rms":velocity_rms,"motion_dyn_range_mm":motion_dyn_range,
            "motion_bw_hz":motion_bw,"motion_snr_db":motion_snr}


# ── Visualization ─────────────────────────────────────────────────────────────

def make_radar(ax, labels, results, colors, cond_labels, is_motion=False):
    """Radar/spider chart — normalized, shows who wins at each metric."""
    if is_motion:
        metrics = ["motion_snr_db","csr_db","coherence","motion_dyn_range_mm","velocity_rms"]
        display = ["Motion\nSNR","CSR","Phase\nCoherence","Dynamic\nRange","Velocity\nRMS"]
    else:
        metrics = ["resp_snr_db","csr_db","coherence","breath_confidence","disp_amplitude_mm"]
        display = ["Resp\nSNR","CSR","Phase\nCoherence","Breathing\nConfidence","Signal\nAmplitude"]
    
    n_met = len(metrics)
    angles = np.linspace(0, 2*np.pi, n_met, endpoint=False).tolist()
    angles += angles[:1]

    # Normalize to 0–1, with min at 0.15 for visibility
    raw = {m: [r[m] for r in results] for m in metrics}
    norm = {}
    for m in metrics:
        lo, hi = min(raw[m]), max(raw[m])
        span = hi - lo if hi > lo else 1.0
        norm[m] = [(v - lo) / span * 0.7 + 0.25 for v in raw[m]]

    ax.set_facecolor(BG)
    for i, r in enumerate(results):
        vals = [norm[m][i] for m in metrics] + [norm[metrics[0]][i]]
        ax.fill(angles, vals, alpha=0.15, color=colors[i])
        ax.plot(angles, vals, lw=2.5, color=colors[i], label=cond_labels[i],
                marker='o', markersize=6, markeredgecolor='white', markeredgewidth=1)

    ax.set_thetagrids(np.degrees(angles[:-1]), display, fontsize=9, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", "Best"], fontsize=7, color='#999')
    ax.grid(True, alpha=0.3, color='#CCC')
    ax.legend(loc='lower left', bbox_to_anchor=(-0.15, -0.15), fontsize=9,
              framealpha=0.9, edgecolor='#DDD', ncol=len(cond_labels))
    ax.set_title("Overall Performance", fontsize=13, fontweight='bold', pad=20)


def make_horizontal_bars(ax, labels, results, colors, is_motion=False):
    """Horizontal grouped bars showing improvement vs bare phone."""
    if is_motion:
        metrics = [
            ("motion_snr_db",       "Motion SNR",        "dB",    True),
            ("csr_db",              "CSR",               "dB",    True),
            ("motion_dyn_range_mm", "Dynamic Range",     "mm",    True),
            ("velocity_rms",        "Velocity RMS",      "mm/s",  True),
            ("coherence",           "Phase Coherence",   "",      True),
        ]
    else:
        metrics = [
            ("resp_snr_db",       "Resp SNR",          "dB",  True),
            ("csr_db",            "CSR",               "dB",  True),
            ("disp_amplitude_mm", "Signal Amplitude",  "mm",  True),
            ("breath_confidence", "Detection Confidence", "x", True),
            ("coherence",         "Phase Coherence",   "",    True),
        ]

    n_met = len(metrics)
    n_cond = len(results)
    y_pos = np.arange(n_met)
    bar_h = 0.7 / n_cond
    offsets = (np.arange(n_cond) - (n_cond-1)/2) * bar_h

    # Draw bars — one barh call per condition so legend picks up correct color
    for i in range(n_cond):
        vals = [results[i][key] for key, _, _, _ in metrics]
        ax.barh(y_pos + offsets[i], vals, bar_h * 0.85,
                color=colors[i], alpha=0.85,
                edgecolor='white', linewidth=0.5,
                label=labels[i])
        
        # Value labels
        for j, (key, name, unit, hb) in enumerate(metrics):
            val = results[i][key]
            all_vals = [r[key] for r in results]
            max_abs = max(abs(v) for v in all_vals) if all_vals else 1
            fmt = f"{val:.1f}" if abs(val) >= 1 else f"{val:.3f}"
            if unit: fmt += f" {unit}"
            x_pos = val + 0.03 * max_abs if val >= 0 else val - 0.03 * max_abs
            ha = 'left' if val >= 0 else 'right'
            ax.text(x_pos, y_pos[j] + offsets[i], fmt,
                   va='center', ha=ha, fontsize=7.5, fontweight='bold', color=colors[i])

    # Best indicator: star placed well past the value label to avoid overlap
    for j, (key, name, unit, hb) in enumerate(metrics):
        all_vals = [r[key] for r in results]
        best_idx = np.argmax(all_vals) if hb else np.argmin(all_vals)
        best_val = all_vals[best_idx]
        max_abs = max(abs(v) for v in all_vals) if all_vals else 1
        # Format the value label text to estimate its width
        fmt_val = f"{best_val:.1f}" if abs(best_val) >= 1 else f"{best_val:.3f}"
        if unit: fmt_val += f" {unit}"
        # Place star after the value label: value_end + generous gap
        char_width = 0.018 * max_abs  # approximate width per character in data coords
        label_width = len(fmt_val) * char_width
        if best_val >= 0:
            star_x = best_val + 0.04 * max_abs + label_width + 0.06 * max_abs
            ha_star = 'left'
        else:
            star_x = best_val - 0.04 * max_abs - label_width - 0.06 * max_abs
            ha_star = 'right'
        ax.text(star_x, y_pos[j] + offsets[best_idx], "★best",
               va='center', ha=ha_star,
               fontsize=7.5, color=colors[best_idx], fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([m[1] for m in metrics], fontsize=10, fontweight='bold')
    ax.set_xlabel("Value", fontsize=9)
    ax.set_title("Metric Comparison", fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.2)
    ax.set_facecolor(BG)
    ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
    # Expand x-axis to prevent label clipping
    xlim = ax.get_xlim()
    x_range = xlim[1] - xlim[0]
    ax.set_xlim(xlim[0] - 0.05 * x_range, xlim[1] + 0.35 * x_range)


def make_improvement_table(ax, labels, results, colors, is_motion=False):
    """Visual improvement table vs baseline (first condition)."""
    ax.axis('off')
    ax.set_facecolor(BG)
    
    if is_motion:
        metrics = [
            ("motion_snr_db",       "Motion SNR",    "dB",    True),
            ("motion_dyn_range_mm", "Dyn Range",     "mm",    True),
            ("velocity_rms",        "Vel RMS",       "mm/s",  True),
            ("csr_db",              "CSR",           "dB",    True),
        ]
    else:
        metrics = [
            ("resp_snr_db",       "Resp SNR",    "dB",  True),
            ("disp_amplitude_mm", "Amplitude",   "mm",  True),
            ("breath_confidence", "Confidence",  "x",   True),
            ("csr_db",            "CSR",         "dB",  True),
        ]
    
    n_cond = len(results)
    if n_cond < 2:
        ax.text(0.5, 0.5, "Need 2+ conditions", ha='center', va='center', fontsize=12)
        return
    
    # Header
    col_labels = ["Metric"] + [f"vs {labels[0]}" if i > 0 else labels[0] for i in range(n_cond)]
    
    table_data = []
    cell_colors = []
    
    for key, name, unit, hb in metrics:
        row = [name]
        row_colors = ['#F0F0F0']
        
        base_val = results[0][key]
        row.append(f"{base_val:.2f} {unit}")
        row_colors.append('#F7F7F7')
        
        for i in range(1, n_cond):
            val = results[i][key]
            diff = val - base_val
            
            if key == "disp_amplitude_mm":
                ratio = val / (base_val + 1e-9)
                txt = f"{ratio:.1f}×"
                good = ratio > 1.2
            else:
                txt = f"{diff:+.1f} {unit}"
                good = diff > 0 if hb else diff < 0
            
            row.append(txt)
            row_colors.append('#E8F5E9' if good else '#FFEBEE')
        
        table_data.append(row)
        cell_colors.append(row_colors)
    
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellColours=cell_colors,
                     colColours=['#E0E0E0'] * (n_cond + 1),
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # Style header
    for j in range(n_cond + 1):
        table[0, j].set_text_props(fontweight='bold')
    
    ax.set_title("Improvement vs Baseline", fontsize=13, fontweight='bold', pad=15)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(paths, labels=None):
    if labels is None: labels = [Path(p).stem for p in paths]
    colors = COLORS[:len(paths)]
    nc = len(paths)

    results = []
    for i, p in enumerate(paths):
        print(f"Processing [{labels[i]}]: {Path(p).name} ...")
        results.append(process_one(p))

    # ══════════════════════════════════════════════════════════════════════
    # Figure 1: Dashboard — Radar + Bars + Improvement Table
    # ══════════════════════════════════════════════════════════════════════

    # Detect activity type
    avg_ratio = np.mean([r["activity_ratio"] for r in results[1:]]) if nc > 1 else results[0]["activity_ratio"]
    is_motion = avg_ratio > 3.0
    activity_label = "Hand Movement" if is_motion else "Breathing"

    fig = plt.figure(figsize=(20, 11), facecolor=BG)
    fig.suptitle(f"AcousticProbe -- Performance Dashboard ({activity_label})",
                 fontsize=18, fontweight='bold', color='#2C3E50', y=0.98)

    # Layout: Radar (left, tall) | Bars (top-right) | Table (mid-right) | Highlight (bot-right)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35,
                           left=0.06, right=0.96, top=0.91, bottom=0.10,
                           height_ratios=[1.2, 0.8, 1.0],
                           width_ratios=[1, 1.3])

    # Radar chart (left, spans all three rows)
    ax_radar = fig.add_subplot(gs[:, 0], polar=True)
    make_radar(ax_radar, labels, results, colors, labels, is_motion=is_motion)

    # Horizontal bars (top-right)
    ax_bars = fig.add_subplot(gs[0, 1])
    make_horizontal_bars(ax_bars, labels, results, colors, is_motion=is_motion)

    # Improvement table (mid-right)
    ax_table = fig.add_subplot(gs[1, 1])
    make_improvement_table(ax_table, labels, results, colors, is_motion=is_motion)

    # BPM + Amplitude highlight (bottom-right only)
    ax_hl = fig.add_subplot(gs[2, 1])
    ax_hl.set_facecolor(BG)
    
    x = np.arange(nc)
    w = 0.25

    if is_motion:
        # Motion mode: Dynamic Range + Velocity RMS
        vals_left = [r["motion_dyn_range_mm"] for r in results]
        vals_right = [r["velocity_rms"] for r in results]
        label_left, unit_left = "Dynamic Range", "mm"
        label_right, unit_right = "Velocity RMS", "mm/s"
    else:
        # Breathing mode: Amplitude + BPM
        vals_left = [r["disp_amplitude_mm"] for r in results]
        vals_right = [r["resp_rate"] for r in results]
        label_left, unit_left = "Amplitude", "mm"
        label_right, unit_right = "Breathing rate", "bpm"

    bars1 = ax_hl.bar(x - w/2, vals_left, w,
                      color=colors[:nc], alpha=0.85, edgecolor='white', linewidth=0.5)
    
    ax2 = ax_hl.twinx()
    bars2 = ax2.bar(x + w/2, vals_right, w,
                    color=colors[:nc], alpha=0.3, hatch='///', edgecolor='#666', linewidth=0.5)
    
    # Value labels on bars
    for bar, val in zip(bars1, vals_left):
        ax_hl.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                  f'{val:.2f}{unit_left}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, vals_right):
        txt = f'{val:.1f} {unit_right}' if val > 0 else 'N/A'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                txt, ha='center', va='bottom', fontsize=9, fontweight='bold', color='#666')
    
    ax_hl.set_xticks(x)
    ax_hl.set_xticklabels(labels, fontsize=9)
    ax_hl.set_ylabel(f"{label_left} ({unit_left})", fontsize=9)
    ax_hl.set_xlim(-0.6, nc - 0.4)
    ax2.set_ylabel(f"{label_right} ({unit_right})", fontsize=9, color='#999')
    if not is_motion:
        ax2.axhspan(12, 20, alpha=0.06, color='green')
    ax_hl.set_title(f"{label_left} & {label_right}", fontsize=11, fontweight='bold')
    ax_hl.grid(axis='y', alpha=0.2)
    
    # Legend explaining the two bar types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#888', alpha=0.85, edgecolor='white', label=f'{label_left} ({unit_left})'),
        Patch(facecolor='#888', alpha=0.3, hatch='///', edgecolor='#666', label=f'{label_right} ({unit_right})'),
    ]
    ax_hl.legend(handles=legend_elements, fontsize=7, loc='upper left', framealpha=0.9)

    # ── Conclusion text box ───────────────────────────────────────────────
    # Use activity-appropriate metrics for winner calculation
    if is_motion:
        win_metrics = [("motion_snr_db",True),("csr_db",True),("coherence",True),
                       ("motion_dyn_range_mm",True),("velocity_rms",True)]
        snr_key, amp_key = "motion_snr_db", "motion_dyn_range_mm"
    else:
        win_metrics = [("resp_snr_db",True),("csr_db",True),("coherence",True),
                       ("breath_confidence",True),("disp_amplitude_mm",True)]
        snr_key, amp_key = "resp_snr_db", "disp_amplitude_mm"

    best_snr_idx = np.argmax([r[snr_key] for r in results])
    best_amp_idx = np.argmax([r[amp_key] for r in results])
    
    # Overall winner: who wins the most metrics
    wins = [0] * nc
    for key, hb in win_metrics:
        vals = [r[key] for r in results]
        winner = np.argmax(vals) if hb else np.argmin(vals)
        wins[winner] += 1
    overall_best = np.argmax(wins)
    
    # Build conclusion lines
    snr_gain = results[best_snr_idx][snr_key] - results[0][snr_key]
    amp_ratio = results[best_amp_idx][amp_key] / (results[0][amp_key] + 1e-9)
    
    conclusion_lines = []
    if nc > 1 and overall_best > 0:
        conclusion_lines.append(f"[BEST]  Best overall: {labels[overall_best]} (wins {wins[overall_best]}/{sum(wins)} metrics)")
        conclusion_lines.append(f"[+]  SNR improvement: +{snr_gain:.1f} dB ({labels[best_snr_idx]})")
        if is_motion:
            conclusion_lines.append(f"[+]  Dynamic range: {amp_ratio:.1f}x wider ({labels[best_amp_idx]})")
            best_vel_idx = np.argmax([r["velocity_rms"] for r in results])
            conclusion_lines.append(f"[+]  Velocity RMS: {results[best_vel_idx]['velocity_rms']:.1f} mm/s ({labels[best_vel_idx]})")
        else:
            conclusion_lines.append(f"[+]  Signal amplitude: {amp_ratio:.1f}x stronger ({labels[best_amp_idx]})")
            best_conf_idx = np.argmax([r["breath_confidence"] for r in results])
            best_conf = results[best_conf_idx]["breath_confidence"]
            if best_conf > 3:
                conclusion_lines.append(f"[OK]  Reliable detection (confidence {best_conf:.1f}x)")
            else:
                conclusion_lines.append(f"[!]  Detection confidence below threshold ({best_conf:.1f}x < 3x)")
    else:
        conclusion_lines.append("[i]  Baseline only -- add structure recordings to compare")
    
    conclusion_text = "\n".join(conclusion_lines)
    
    fig.text(0.06, 0.01, conclusion_text,
             ha='left', va='bottom', fontsize=10, fontweight='bold',
             color='#2C3E50', family='monospace', linespacing=1.5,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F6E8' if overall_best > 0 else '#FFF9C4',
                       edgecolor='#27AE60' if overall_best > 0 else '#F39C12',
                       alpha=0.95, linewidth=2))

    # Adjust layout to make room for conclusion box
    gs.update(bottom=0.14)

    out1 = Path(paths[0]).parent / "metrics_comparison.png"
    plt.savefig(str(out1), dpi=150, facecolor=BG, bbox_inches='tight')
    print(f"Saved → {out1}")

    # ══════════════════════════════════════════════════════════════════════
    # Figure 2: Signal Comparison (2×2)
    # ══════════════════════════════════════════════════════════════════════

    # Detect dominant activity type across conditions
    # (is_motion and activity_label already computed above)
    if is_motion:
        filt_label = "Motion Waveform (filtered 0.1-10 Hz)"
        spectrum_label = "Motion Spectrum"
        spectrum_unit = "movements/min"
        amp_label = "Motion Amplitude"
    else:
        filt_label = "Breathing Waveform (filtered 0.1-3 Hz)"
        spectrum_label = "Breathing Spectrum"
        spectrum_unit = "breaths/min"
        amp_label = "Resp Amplitude"

    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 9), facecolor=BG)
    fig2.suptitle(f"AcousticProbe -- Signal Comparison ({activity_label})",
                  fontsize=16, fontweight='bold', color='#2C3E50')

    for ax in axes2.flat:
        ax.set_facecolor(BG)

    # Raw displacement
    ax = axes2[0, 0]
    for i, r in enumerate(results):
        ax.plot(r["t_chirps"], r["disp_detrended"],
                color=colors[i], lw=0.4, alpha=0.5, label=labels[i])
    ax.set_title("Raw Phase Displacement", fontweight='bold')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("mm")
    ax.legend(fontsize=8, framealpha=0.8); ax.grid(True, alpha=0.15)

    # Filtered waveform (breathing or motion)
    ax = axes2[0, 1]
    for i, r in enumerate(results):
        waveform = r["motion_signal"] if is_motion else r["resp_signal"]
        amp = r["motion_amplitude_mm"] if is_motion else r["disp_amplitude_mm"]
        t = r["t_chirps"]
        # Trim to match waveform length if needed
        minlen = min(len(t), len(waveform))
        ax.plot(t[:minlen], waveform[:minlen], color=colors[i], lw=2, alpha=0.85,
                label=f'{labels[i]}  +/-{amp:.2f}mm')
    ax.set_title(filt_label, fontweight='bold')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("mm")
    ax.legend(fontsize=8, framealpha=0.8); ax.grid(True, alpha=0.15)

    # Spectrum
    ax = axes2[1, 0]
    for i, r in enumerate(results):
        mk = (r["resp_freqs"]>=0.05)&(r["resp_freqs"]<=2.0)
        ax.plot(r["resp_freqs"][mk]*60, r["resp_fft"][mk],
                color=colors[i], lw=2, alpha=0.85, label=labels[i])
    if not is_motion:
        ax.axvspan(12, 20, alpha=0.08, color='green', label='Normal range')
    ax.set_title(spectrum_label, fontweight='bold')
    ax.set_xlabel(spectrum_unit); ax.set_ylabel("Magnitude")
    ax.legend(fontsize=8, framealpha=0.8); ax.grid(True, alpha=0.15)

    # Range profile
    ax = axes2[1, 1]
    for i, r in enumerate(results):
        db = 20*np.log10(r["mean_prof"][:r["max_idx"]]+1e-9)
        ax.plot(r["range_ax"][:r["max_idx"]], db,
                color=colors[i], lw=2, alpha=0.85, label=labels[i])
        ax.axvline(r["target_m"], color=colors[i], ls=':', lw=1.5, alpha=0.4)
    ax.set_title("Range Profile (FMCW echo vs distance)", fontweight='bold')
    ax.set_xlabel("Distance (m)"); ax.set_ylabel("dB")
    ax.legend(fontsize=8, framealpha=0.8); ax.grid(True, alpha=0.15)

    # Annotations
    best_amp_idx2 = np.argmax([r["disp_amplitude_mm"] for r in results])
    best_amp_val = results[best_amp_idx2]["disp_amplitude_mm"]
    worst_amp_val = results[0]["disp_amplitude_mm"]
    ratio = best_amp_val / (worst_amp_val + 1e-9)

    if nc > 1:
        axes2[0, 1].annotate(
            f"Best: {labels[best_amp_idx2]}\n+/-{best_amp_val:.2f}mm ({ratio:.1f}x vs bare)",
            xy=(0.98, 0.95), xycoords='axes fraction',
            ha='right', va='top', fontsize=9, fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=colors[best_amp_idx2],
                      alpha=0.85, edgecolor='white', linewidth=1.5))

    best_rate_idx = -1
    best_conf_val = 0
    for i, r in enumerate(results):
        if r["breath_confidence"] > best_conf_val:
            best_conf_val = r["breath_confidence"]
            best_rate_idx = i

    if best_rate_idx >= 0 and results[best_rate_idx]["resp_rate"] > 0:
        rate = results[best_rate_idx]["resp_rate"]
        rate_label = f"{rate:.0f} bpm" if not is_motion else f"{rate:.1f} Hz"
        axes2[1, 0].annotate(
            f"Detected: {rate_label}\n({labels[best_rate_idx]}, conf {best_conf_val:.1f}x)",
            xy=(0.98, 0.95), xycoords='axes fraction',
            ha='right', va='top', fontsize=9, fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=colors[best_rate_idx],
                      alpha=0.85, edgecolor='white', linewidth=1.5))

    plt.tight_layout()
    out2 = Path(paths[0]).parent / "signal_comparison.png"
    plt.savefig(str(out2), dpi=150, facecolor=BG)
    print(f"Saved -> {out2}")

    # ══════════════════════════════════════════════════════════════════════
    # Figure 3: Metrics Table (saved as image)
    # ══════════════════════════════════════════════════════════════════════
    if is_motion:
        col_headers = ["Condition", "MotionSNR\n(dB)", "CSR\n(dB)", "Coherence\n(0-1)",
                       "DynRange\n(mm)", "VelRMS\n(mm/s)", "MotionBW\n(Hz)", "Jitter\n(cm)"]
    else:
        col_headers = ["Condition", "RespSNR\n(dB)", "CSR\n(dB)", "Coherence\n(0-1)",
                       "Confidence\n(x)", "Amplitude\n(mm)", "BPM", "Jitter\n(cm)"]

    table_rows = []
    row_colors_list = []
    for i, r in enumerate(results):
        if is_motion:
            row = [labels[i],
                   f"{r['motion_snr_db']:.1f}",
                   f"{r['csr_db']:.1f}",
                   f"{r['coherence']:.3f}",
                   f"{r['motion_dyn_range_mm']:.2f}",
                   f"{r['velocity_rms']:.1f}",
                   f"{r['motion_bw_hz']:.2f}",
                   f"{r['tracking_jitter_cm']:.1f}"]
        else:
            cs = f"{r['breath_confidence']:.1f}x"
            bs = f"{r['resp_rate']:.1f}" if r['resp_rate'] > 0 else "N/A"
            row = [labels[i],
                   f"{r['resp_snr_db']:.1f}",
                   f"{r['csr_db']:.1f}",
                   f"{r['coherence']:.3f}",
                   cs,
                   f"{r['disp_amplitude_mm']:.2f}",
                   bs,
                   f"{r['tracking_jitter_cm']:.1f}"]
        table_rows.append(row)
        row_colors_list.append([colors[i] + '18'] * len(col_headers))  # light tint

    # Improvement rows
    if nc > 1:
        table_rows.append([""] * len(col_headers))  # separator
        row_colors_list.append(['#FFFFFF'] * len(col_headers))

        for i in range(1, nc):
            dc = results[i]["csr_db"] - results[0]["csr_db"]
            dc_coh = results[i]["coherence"] - results[0]["coherence"]
            if is_motion:
                ds = results[i]["motion_snr_db"] - results[0]["motion_snr_db"]
                dr = results[i]["motion_dyn_range_mm"] / (results[0]["motion_dyn_range_mm"] + 1e-9)
                dv = results[i]["velocity_rms"] / (results[0]["velocity_rms"] + 1e-9)
                db = results[i]["motion_bw_hz"] - results[0]["motion_bw_hz"]
                row = [f"vs {labels[0]}",
                       f"{ds:+.1f}",
                       f"{dc:+.1f}",
                       f"{dc_coh:+.3f}",
                       f"{dr:.1f}x",
                       f"{dv:.1f}x",
                       f"{db:+.2f}",
                       "--"]
            else:
                ds = results[i]["resp_snr_db"] - results[0]["resp_snr_db"]
                ar = results[i]["disp_amplitude_mm"] / (results[0]["disp_amplitude_mm"] + 1e-9)
                dc_conf = results[i]["breath_confidence"] - results[0]["breath_confidence"]
                row = [f"vs {labels[0]}",
                       f"{ds:+.1f}",
                       f"{dc:+.1f}",
                       f"{dc_coh:+.3f}",
                       f"{dc_conf:+.1f}x",
                       f"{ar:.1f}x",
                       "--",
                       "--"]
            table_rows.append(row)
            good_color = '#E8F5E9'
            row_colors_list.append([good_color] * len(col_headers))

    # Create figure
    n_rows = len(table_rows)
    fig3_h = max(3.0, 1.2 + n_rows * 0.5)
    fig3, ax3 = plt.subplots(figsize=(14, fig3_h), facecolor=BG)
    ax3.axis('off')
    ax3.set_title(f"AcousticProbe -- Metrics Summary ({activity_label})",
                  fontsize=15, fontweight='bold', color='#2C3E50', pad=15)

    tbl = ax3.table(cellText=table_rows, colLabels=col_headers,
                    cellColours=row_colors_list,
                    colColours=['#D5D8DC'] * len(col_headers),
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.0)

    # Bold headers
    for j in range(len(col_headers)):
        tbl[0, j].set_text_props(fontweight='bold', fontsize=9)

    # Bold condition names
    for i in range(len(table_rows)):
        tbl[i + 1, 0].set_text_props(fontweight='bold')

    # Winner banner
    if nc > 1:
        wins = [0] * nc
        for key, hb in win_metrics:
            vals = [r[key] for r in results]
            winner = np.argmax(vals) if hb else np.argmin(vals)
            wins[winner] += 1
        overall_best = np.argmax(wins)
        best_snr = max(range(nc), key=lambda i: results[i][snr_key])
        best_amp = max(range(nc), key=lambda i: results[i][amp_key])
        amp_val = results[best_amp][amp_key]
        amp_ratio2 = amp_val / (results[0][amp_key] + 1e-9)

        amp_name = "Dynamic Range" if is_motion else "Amplitude"
        summary = (f"BEST OVERALL: {labels[overall_best]} "
                   f"(wins {wins[overall_best]}/5)   |   "
                   f"Best SNR: {labels[best_snr]} ({results[best_snr][snr_key]:+.1f} dB)   |   "
                   f"Best {amp_name}: {labels[best_amp]} ({amp_val:.2f} mm, {amp_ratio2:.1f}x)")
        fig3.text(0.5, 0.02, summary, ha='center', va='bottom',
                  fontsize=10, fontweight='bold', color='#2C3E50',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F6E8',
                            edgecolor='#27AE60', alpha=0.9, linewidth=1.5))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out3 = Path(paths[0]).parent / "metrics_table.png"
    plt.savefig(str(out3), dpi=150, facecolor=BG, bbox_inches='tight')
    print(f"Saved -> {out3}")

    # ══════════════════════════════════════════════════════════════════════
    # Terminal table
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*100)
    print(f"  AcousticProbe -- Performance Metrics ({activity_label})")
    print("="*100)
    if is_motion:
        print(f"{'Condition':<18} {'MotSNR':>9} {'CSR':>8} {'Coherence':>10} "
              f"{'DynRange':>10} {'VelRMS':>10} {'MotBW':>8} {'Jitter':>9}")
        print(f"{'':18} {'(dB)':>9} {'(dB)':>8} {'(0-1)':>10} "
              f"{'(mm)':>10} {'(mm/s)':>10} {'(Hz)':>8} {'(cm)':>9}")
    else:
        print(f"{'Condition':<18} {'RespSNR':>9} {'CSR':>8} {'Coherence':>10} "
              f"{'Confidence':>11} {'Amplitude':>10} {'BPM':>6} {'Jitter':>9}")
        print(f"{'':18} {'(dB)':>9} {'(dB)':>8} {'(0-1)':>10} "
              f"{'(x)':>11} {'(mm)':>10} {'':>6} {'(cm)':>9}")
    print("-"*100)
    for i,r in enumerate(results):
        if is_motion:
            print(f"{labels[i]:<18} {r['motion_snr_db']:>9.1f} {r['csr_db']:>8.1f} "
                  f"{r['coherence']:>10.3f} {r['motion_dyn_range_mm']:>10.2f} "
                  f"{r['velocity_rms']:>10.1f} {r['motion_bw_hz']:>8.2f} "
                  f"{r['tracking_jitter_cm']:>9.1f}")
        else:
            cs = f"{r['breath_confidence']:.1f}x"
            bs = f"{r['resp_rate']:.1f}" if r['resp_rate']>0 else "N/A"
            print(f"{labels[i]:<18} {r['resp_snr_db']:>9.1f} {r['csr_db']:>8.1f} "
                  f"{r['coherence']:>10.3f} {cs:>11} "
                  f"{r['disp_amplitude_mm']:>10.2f} {bs:>6} "
                  f"{r['tracking_jitter_cm']:>9.1f}")
    print("-"*100)

    if nc>1:
        print(f"\n  Improvement vs {labels[0]}:")
        if is_motion:
            print(f"  {'':18} {'dSNR':>9} {'dCSR':>8} {'DynRng':>10} {'VelRMS':>10}")
            print(f"  {'-'*60}")
            for i in range(1,nc):
                ds=results[i]["motion_snr_db"]-results[0]["motion_snr_db"]
                dc=results[i]["csr_db"]-results[0]["csr_db"]
                dr=results[i]["motion_dyn_range_mm"]/(results[0]["motion_dyn_range_mm"]+1e-9)
                vr=results[i]["velocity_rms"]/(results[0]["velocity_rms"]+1e-9)
                print(f"  {labels[i]:<18} {ds:>+9.1f} dB {dc:>+8.1f} dB {dr:>10.1f}x {vr:>10.1f}x")
        else:
            print(f"  {'':18} {'dSNR':>9} {'dCSR':>8} {'Amp':>10}")
            print(f"  {'-'*50}")
            for i in range(1,nc):
                ds=results[i]["resp_snr_db"]-results[0]["resp_snr_db"]
                dc=results[i]["csr_db"]-results[0]["csr_db"]
                ar=results[i]["disp_amplitude_mm"]/(results[0]["disp_amplitude_mm"]+1e-9)
                print(f"  {labels[i]:<18} {ds:>+9.1f} dB {dc:>+8.1f} dB {ar:>10.1f}x")

        wins = [0] * nc
        for key, hb in win_metrics:
            vals = [r[key] for r in results]
            winner = np.argmax(vals) if hb else np.argmin(vals)
            wins[winner] += 1
        overall_best = np.argmax(wins)

        print(f"\n  >>> BEST OVERALL: {labels[overall_best]} (wins {wins[overall_best]}/5 metrics)")
        best_snr_t = max(range(nc), key=lambda i: results[i][snr_key])
        best_amp_t = max(range(nc), key=lambda i: results[i][amp_key])
        snr_val = results[best_snr_t][snr_key]
        amp_val = results[best_amp_t][amp_key]
        amp_ratio = amp_val / (results[0][amp_key] + 1e-9)
        amp_name = "Dynamic Range" if is_motion else "Amplitude"
        print(f"     Best SNR:       {labels[best_snr_t]} ({snr_val:+.1f} dB)")
        print(f"     Best {amp_name}: {labels[best_amp_t]} ({amp_val:.2f} mm, {amp_ratio:.1f}x vs bare)")
    print("="*100)

    plt.show()


if __name__ == "__main__":
    n = len(sys.argv) - 1
    if n >= 2:
        paths = sys.argv[1:]
        lb = [Path(p).stem for p in paths]
        main(paths, lb)
    else:
        dl = Path.home()/"Downloads"
        ws = sorted(dl.glob("fmcw_*.wav"), key=lambda p:p.stat().st_mtime)[-3:]
        if len(ws)<2:
            print(f"Found {len(ws)} WAVs. Need ≥2.")
            print("Usage: python compare_conditions.py <bare.wav> <tubeA.wav> [...]")
            sys.exit(1)
        lb = [p.stem for p in ws]
        print("Auto-detected (oldest→newest):")
        for i,w in enumerate(ws): print(f"  [{lb[i]}] {w.name}")
        main([str(w) for w in ws], lb)
        