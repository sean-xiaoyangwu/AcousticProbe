"""
AcousticProbe -- Multi-condition comparison with BOTH breathing & motion metrics

Usage:
    python compare_conditions.py <bare.wav> <tube3mm.wav> <tube4mm.wav>
    python compare_conditions.py   (auto-detects newest WAVs in ~/Downloads)

NOTE: The FIRST file is always treated as the baseline (bare phone).
      All improvements are computed relative to this baseline.
"""

import sys, json, numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy import signal as sig
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# -- Config --

LABELS = ["Bare iPhone", "Tube A", "Tube B", "Tube C", "Tube D"]
COLORS = ["#6C7A89", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD"]
BG     = "#FAFAFA"

# -- DSP helpers --

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

# -- Process one WAV --

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

    # Cross-correlation alignment (from analyze_fmcw.py / KevinsWang):
    # Find precise chirp start to compensate audio buffer timing jitter.
    chirp_ref = make_chirp(p)
    search_len = min(3 * N, len(rx))
    corr = np.abs(np.correlate(rx[:search_len], chirp_ref, mode='valid'))
    offset = int(np.argmax(corr))
    rx = rx[offset:]

    n=(len(rx)//N)*N; rx=rx[:n].reshape(-1,N)
    tx=np.tile(chirp_ref,(rx.shape[0],1))

    # Drop head only (speaker startup transient), matching analyze_fmcw.py
    rx,tx = rx[drop:], tx[drop:]
    nc=rx.shape[0]

    mixed=np.apply_along_axis(lambda x:lp(x,5000,fs),1,rx*tx)
    NFFT=N*4
    # Hanning window (from analyze_fmcw.py): suppresses sidelobes -13dB -> -31dB
    hann_win = np.hanning(N)
    ffts=np.fft.rfft(mixed * hann_win, n=NFFT, axis=1)
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

    # -- Breathing metrics (0.1-3 Hz) --
    rs=bp_zp(dd,0.1,3.0,fs_c)
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

    # -- Motion metrics (0.1-10 Hz) --
    nyq_motion = min(10.0, fs_c/2 - 0.5)
    ms = bp_zp(dd, 0.1, nyq_motion, fs_c) if fs_c > 21 else rs
    motion_amp = np.percentile(ms, 97.5) - np.percentile(ms, 2.5)
    vel = np.diff(ms) * fs_c
    velocity_rms = np.sqrt(np.mean(vel**2))
    motion_dyn_range = np.max(ms) - np.min(ms)

    ms_fft = np.abs(np.fft.rfft(ms))**2
    ms_freqs = np.fft.rfftfreq(len(ms), d=T)
    ms_fft_norm = ms_fft / (np.max(ms_fft) + 1e-12)
    above_3db = ms_freqs[ms_fft_norm >= 0.5]
    motion_bw = (above_3db[-1] - above_3db[0]) if len(above_3db) >= 2 else 0.0
    ms_noise = dd - ms
    motion_snr = 10*np.log10(np.var(ms)/(np.var(ms_noise)+1e-12))

    return {
        "resp_snr_db":rsnr, "csr_db":csr, "coherence":coh,
        "breath_confidence":bc, "resp_rate":rr, "disp_amplitude_mm":da,
        "tracking_jitter_cm":jt,
        "motion_snr_db":motion_snr, "velocity_rms":velocity_rms,
        "motion_dyn_range_mm":motion_dyn_range, "motion_bw_hz":motion_bw,
        "motion_amplitude_mm":motion_amp,
        "target_m":tm, "range_ax":rax, "max_idx":mi,
        "mean_prof":mp, "resp_signal":rs, "motion_signal":ms,
        "disp_detrended":dd, "t_chirps":tc, "resp_fft":rf, "resp_freqs":rfq,
    }


# -- Visualization helpers --

def make_radar(ax, results, colors, cond_labels, metrics, display, title):
    n_met = len(metrics)
    angles = np.linspace(0, 2*np.pi, n_met, endpoint=False).tolist()
    angles += angles[:1]
    raw = {m: [r[m] for r in results] for m in metrics}
    norm = {}
    for m in metrics:
        lo, hi = min(raw[m]), max(raw[m])
        span = hi - lo if hi > lo else 1.0
        norm[m] = [(v - lo) / span * 0.7 + 0.25 for v in raw[m]]
    ax.set_facecolor(BG)
    for i in range(len(results)):
        vals = [norm[m][i] for m in metrics] + [norm[metrics[0]][i]]
        ax.fill(angles, vals, alpha=0.15, color=colors[i])
        ax.plot(angles, vals, lw=2.5, color=colors[i], label=cond_labels[i],
                marker='o', markersize=5, markeredgecolor='white', markeredgewidth=1)
    ax.set_thetagrids(np.degrees(angles[:-1]), display, fontsize=8, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", "Best"], fontsize=6, color='#999')
    ax.grid(True, alpha=0.3, color='#CCC')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=15)


def make_bars(ax, labels, results, colors, metrics_list, title):
    n_met = len(metrics_list)
    n_cond = len(results)
    y_pos = np.arange(n_met)
    bar_h = 0.7 / n_cond
    offsets = (np.arange(n_cond) - (n_cond-1)/2) * bar_h
    for i in range(n_cond):
        vals = [results[i][key] for key, _, _, _ in metrics_list]
        ax.barh(y_pos + offsets[i], vals, bar_h * 0.85,
                color=colors[i], alpha=0.85, edgecolor='white', linewidth=0.5, label=labels[i])
        for j, (key, name, unit, hb) in enumerate(metrics_list):
            val = results[i][key]
            all_vals = [r[key] for r in results]
            max_abs = max(abs(v) for v in all_vals) if all_vals else 1
            fmt = f"{val:.1f}" if abs(val) >= 1 else f"{val:.3f}"
            if unit: fmt += f" {unit}"
            x_pos = val + 0.03 * max_abs if val >= 0 else val - 0.03 * max_abs
            ax.text(x_pos, y_pos[j] + offsets[i], fmt,
                   va='center', ha='left' if val >= 0 else 'right',
                   fontsize=6.5, fontweight='bold', color=colors[i])
    for j, (key, name, unit, hb) in enumerate(metrics_list):
        all_vals = [r[key] for r in results]
        best_idx = np.argmax(all_vals) if hb else np.argmin(all_vals)
        best_val = all_vals[best_idx]
        max_abs = max(abs(v) for v in all_vals) if all_vals else 1
        fmt_val = f"{best_val:.1f}" if abs(best_val) >= 1 else f"{best_val:.3f}"
        if unit: fmt_val += f" {unit}"
        cw = 0.018 * max_abs
        lw = len(fmt_val) * cw
        star_x = best_val + 0.04*max_abs + lw + 0.06*max_abs if best_val >= 0 \
                 else best_val - 0.04*max_abs - lw - 0.06*max_abs
        ax.text(star_x, y_pos[j] + offsets[best_idx], "*best",
               va='center', ha='left' if best_val >= 0 else 'right',
               fontsize=6.5, color=colors[best_idx], fontweight='bold')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m[1] for m in metrics_list], fontsize=8, fontweight='bold')
    ax.set_xlabel("Value", fontsize=8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.2); ax.set_facecolor(BG)
    ax.legend(fontsize=7, loc='lower right', framealpha=0.9)
    xlim = ax.get_xlim()
    xr = xlim[1] - xlim[0]
    ax.set_xlim(xlim[0] - 0.05*xr, xlim[1] + 0.35*xr)


# -- Metric sets --

BREATH_RADAR_K = ["resp_snr_db","csr_db","coherence","breath_confidence","disp_amplitude_mm"]
BREATH_RADAR_L = ["Resp\nSNR","CSR","Phase\nCoherence","Breathing\nConfidence","Signal\nAmplitude"]
MOTION_RADAR_K = ["motion_snr_db","csr_db","coherence","motion_dyn_range_mm","velocity_rms"]
MOTION_RADAR_L = ["Motion\nSNR","CSR","Phase\nCoherence","Dynamic\nRange","Velocity\nRMS"]

BREATH_BARS = [
    ("resp_snr_db",       "Resp SNR",          "dB",  True),
    ("csr_db",            "CSR",               "dB",  True),
    ("disp_amplitude_mm", "Resp Amplitude",    "mm",  True),
    ("breath_confidence", "Breath Confidence", "x",   True),
]
MOTION_BARS = [
    ("motion_snr_db",       "Motion SNR",    "dB",    True),
    ("motion_dyn_range_mm", "Dynamic Range", "mm",    True),
    ("velocity_rms",        "Velocity RMS",  "mm/s",  True),
    ("motion_bw_hz",        "Motion BW",     "Hz",    True),
]

ALL_WIN = [
    ("resp_snr_db",True), ("csr_db",True), ("coherence",True),
    ("disp_amplitude_mm",True), ("motion_snr_db",True),
    ("motion_dyn_range_mm",True), ("velocity_rms",True),
]


# -- Main --

def main(paths, labels=None):
    if labels is None: labels = LABELS[:len(paths)]
    colors = COLORS[:len(paths)]
    nc = len(paths)

    # Create timestamped output folder next to first WAV file
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(paths[0]).parent / f"analysis_{ts}"
    out_dir.mkdir(exist_ok=True)

    print(f"\n  Baseline (first file): {labels[0]} -> {Path(paths[0]).name}")
    print(f"  Output folder: {out_dir}")
    print(f"  All improvements are computed vs this baseline.\n")

    results = []
    for i, p in enumerate(paths):
        print(f"  Processing [{labels[i]}]: {Path(p).name} ...")
        results.append(process_one(p))

    # == Figure 1: Dashboard -- 2 radars + 2 bar charts ==
    fig = plt.figure(figsize=(22, 12), facecolor=BG)
    fig.suptitle("AcousticProbe -- Performance Dashboard",
                 fontsize=18, fontweight='bold', color='#2C3E50', y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30,
                           left=0.05, right=0.97, top=0.91, bottom=0.08)

    ax_r1 = fig.add_subplot(gs[0, 0], polar=True)
    make_radar(ax_r1, results, colors, labels,
               BREATH_RADAR_K, BREATH_RADAR_L, "Breathing Performance")

    ax_r2 = fig.add_subplot(gs[0, 1], polar=True)
    make_radar(ax_r2, results, colors, labels,
               MOTION_RADAR_K, MOTION_RADAR_L, "Motion Performance")

    handles = [plt.Line2D([0],[0], color=colors[i], lw=2.5, marker='o', markersize=5,
               markeredgecolor='white', label=labels[i]) for i in range(nc)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.935),
               ncol=nc, fontsize=10, framealpha=0.9, edgecolor='#DDD')

    ax_b1 = fig.add_subplot(gs[1, 0])
    make_bars(ax_b1, labels, results, colors, BREATH_BARS, "Breathing Metrics")

    ax_b2 = fig.add_subplot(gs[1, 1])
    make_bars(ax_b2, labels, results, colors, MOTION_BARS, "Motion Metrics")

    # Conclusion box
    if nc > 1:
        wins = [0] * nc
        for key, hb in ALL_WIN:
            vals = [r[key] for r in results]
            wins[np.argmax(vals) if hb else np.argmin(vals)] += 1
        ob = np.argmax(wins)
        nt = len(ALL_WIN)
        br = max(range(nc), key=lambda i: results[i]["resp_snr_db"])
        bm = max(range(nc), key=lambda i: results[i]["motion_snr_db"])
        ba = max(range(nc), key=lambda i: results[i]["disp_amplitude_mm"])
        bd = max(range(nc), key=lambda i: results[i]["motion_dyn_range_mm"])
        lines = []
        if ob > 0:
            lines.append(f"[BEST] {labels[ob]} (wins {wins[ob]}/{nt} metrics)")
            rg = results[br]["resp_snr_db"] - results[0]["resp_snr_db"]
            mg = results[bm]["motion_snr_db"] - results[0]["motion_snr_db"]
            ar = results[ba]["disp_amplitude_mm"] / (results[0]["disp_amplitude_mm"]+1e-9)
            dr = results[bd]["motion_dyn_range_mm"] / (results[0]["motion_dyn_range_mm"]+1e-9)
            lines.append(f"  RespSNR: +{rg:.1f}dB ({labels[br]})  |  MotSNR: +{mg:.1f}dB ({labels[bm]})")
            lines.append(f"  Amp: {ar:.1f}x ({labels[ba]})  |  DynRange: {dr:.1f}x ({labels[bd]})")
        else:
            lines.append("[i] Baseline wins most metrics")
        fig.text(0.05, 0.01, "\n".join(lines), ha='left', va='bottom',
                 fontsize=9, fontweight='bold', color='#2C3E50', family='monospace',
                 linespacing=1.4,
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='#E8F6E8' if ob>0 else '#FFF9C4',
                           edgecolor='#27AE60' if ob>0 else '#F39C12',
                           alpha=0.95, linewidth=2))

    out1 = out_dir / "metrics_comparison.png"
    plt.savefig(str(out1), dpi=150, facecolor=BG, bbox_inches='tight')
    print(f"\nSaved -> {out1}")

    # == Figure 2: Signal Comparison (2x3) ==
    fig2, axes2 = plt.subplots(2, 3, figsize=(20, 9), facecolor=BG)
    fig2.suptitle("AcousticProbe -- Signal Comparison",
                  fontsize=16, fontweight='bold', color='#2C3E50')
    for ax in axes2.flat: ax.set_facecolor(BG)

    # Row 0: Raw | Breathing waveform | Motion waveform
    ax = axes2[0, 0]
    for i, r in enumerate(results):
        ax.plot(r["t_chirps"], r["disp_detrended"], color=colors[i], lw=0.4, alpha=0.5, label=labels[i])
    ax.set_title("Raw Phase Displacement", fontweight='bold')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("mm")
    ax.legend(fontsize=7, framealpha=0.8); ax.grid(True, alpha=0.15)

    ax = axes2[0, 1]
    for i, r in enumerate(results):
        ax.plot(r["t_chirps"], r["resp_signal"], color=colors[i], lw=2, alpha=0.85,
                label=f'{labels[i]}  +/-{r["disp_amplitude_mm"]:.2f}mm')
    ax.set_title("Breathing Waveform (0.1-3 Hz)", fontweight='bold')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("mm")
    ax.legend(fontsize=7, framealpha=0.8); ax.grid(True, alpha=0.15)

    ax = axes2[0, 2]
    for i, r in enumerate(results):
        t,w = r["t_chirps"], r["motion_signal"]
        ml = min(len(t), len(w))
        ax.plot(t[:ml], w[:ml], color=colors[i], lw=1.5, alpha=0.85,
                label=f'{labels[i]}  +/-{r["motion_amplitude_mm"]:.2f}mm')
    ax.set_title("Motion Waveform (0.1-10 Hz)", fontweight='bold')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("mm")
    ax.legend(fontsize=7, framealpha=0.8); ax.grid(True, alpha=0.15)

    # Row 1: Breathing spectrum | Motion spectrum | Range profile
    ax = axes2[1, 0]
    for i, r in enumerate(results):
        mk = (r["resp_freqs"]>=0.05)&(r["resp_freqs"]<=2.0)
        ax.plot(r["resp_freqs"][mk]*60, r["resp_fft"][mk], color=colors[i], lw=2, alpha=0.85, label=labels[i])
    ax.axvspan(12, 20, alpha=0.08, color='green', label='Normal range')
    ax.set_title("Breathing Spectrum", fontweight='bold')
    ax.set_xlabel("breaths/min"); ax.set_ylabel("Magnitude")
    ax.legend(fontsize=7, framealpha=0.8); ax.grid(True, alpha=0.15)
    bri = max(range(nc), key=lambda i: results[i]["breath_confidence"])
    if results[bri]["resp_rate"] > 0:
        ax.annotate(f"Detected: {results[bri]['resp_rate']:.0f} bpm\n({labels[bri]}, conf {results[bri]['breath_confidence']:.1f}x)",
                    xy=(0.98,0.95), xycoords='axes fraction', ha='right', va='top',
                    fontsize=8, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[bri], alpha=0.85, edgecolor='white'))

    ax = axes2[1, 1]
    for i, r in enumerate(results):
        mf = np.abs(np.fft.rfft(r["motion_signal"]))
        mfq = np.fft.rfftfreq(len(r["motion_signal"]), d=0.020)
        mk = (mfq >= 0.1) & (mfq <= 10.0)
        ax.plot(mfq[mk], mf[mk], color=colors[i], lw=2, alpha=0.85, label=labels[i])
    ax.set_title("Motion Spectrum (0.1-10 Hz)", fontweight='bold')
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Magnitude")
    ax.legend(fontsize=7, framealpha=0.8); ax.grid(True, alpha=0.15)

    ax = axes2[1, 2]
    for i, r in enumerate(results):
        db = 20*np.log10(r["mean_prof"][:r["max_idx"]]+1e-9)
        ax.plot(r["range_ax"][:r["max_idx"]], db, color=colors[i], lw=2, alpha=0.85, label=labels[i])
        ax.axvline(r["target_m"], color=colors[i], ls=':', lw=1.5, alpha=0.4)
    ax.set_title("Range Profile (FMCW echo vs distance)", fontweight='bold')
    ax.set_xlabel("Distance (m)"); ax.set_ylabel("dB")
    ax.legend(fontsize=7, framealpha=0.8); ax.grid(True, alpha=0.15)

    if nc > 1:
        bi = np.argmax([r["disp_amplitude_mm"] for r in results])
        bv = results[bi]["disp_amplitude_mm"]
        rt = bv / (results[0]["disp_amplitude_mm"]+1e-9)
        axes2[0,1].annotate(f"Best: {labels[bi]}\n+/-{bv:.2f}mm ({rt:.1f}x)",
            xy=(0.98,0.95), xycoords='axes fraction', ha='right', va='top',
            fontsize=8, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[bi], alpha=0.85, edgecolor='white'))
        di = np.argmax([r["motion_dyn_range_mm"] for r in results])
        dv = results[di]["motion_dyn_range_mm"]
        dr = dv / (results[0]["motion_dyn_range_mm"]+1e-9)
        axes2[0,2].annotate(f"Best: {labels[di]}\n{dv:.2f}mm ({dr:.1f}x)",
            xy=(0.98,0.95), xycoords='axes fraction', ha='right', va='top',
            fontsize=8, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[di], alpha=0.85, edgecolor='white'))

    plt.tight_layout()
    out2 = out_dir / "signal_comparison.png"
    plt.savefig(str(out2), dpi=150, facecolor=BG)
    print(f"Saved -> {out2}")

    # == Figure 3: Full metrics table ==
    col_h = ["Condition",
             "RespSNR\n(dB)","CSR\n(dB)","Coher.\n(0-1)","BrConf\n(x)","RspAmp\n(mm)","BPM",
             "MotSNR\n(dB)","DynRng\n(mm)","VelRMS\n(mm/s)","MotBW\n(Hz)","Jitter\n(cm)"]
    ncol = len(col_h)
    rows, rcols = [], []
    for i, r in enumerate(results):
        bs = f"{r['resp_rate']:.1f}" if r['resp_rate']>0 else "N/A"
        rows.append([labels[i],
            f"{r['resp_snr_db']:.1f}", f"{r['csr_db']:.1f}", f"{r['coherence']:.3f}",
            f"{r['breath_confidence']:.1f}x", f"{r['disp_amplitude_mm']:.2f}", bs,
            f"{r['motion_snr_db']:.1f}", f"{r['motion_dyn_range_mm']:.2f}",
            f"{r['velocity_rms']:.1f}", f"{r['motion_bw_hz']:.2f}",
            f"{r['tracking_jitter_cm']:.1f}"])
        rcols.append([colors[i]+'18']*ncol)
    if nc > 1:
        rows.append([""]*ncol); rcols.append(['#FFF']*ncol)
        for i in range(1, nc):
            r0, ri = results[0], results[i]
            rows.append([f"vs {labels[0]}",
                f"{ri['resp_snr_db']-r0['resp_snr_db']:+.1f}",
                f"{ri['csr_db']-r0['csr_db']:+.1f}",
                f"{ri['coherence']-r0['coherence']:+.3f}",
                f"{ri['breath_confidence']-r0['breath_confidence']:+.1f}x",
                f"{ri['disp_amplitude_mm']/(r0['disp_amplitude_mm']+1e-9):.1f}x", "--",
                f"{ri['motion_snr_db']-r0['motion_snr_db']:+.1f}",
                f"{ri['motion_dyn_range_mm']/(r0['motion_dyn_range_mm']+1e-9):.1f}x",
                f"{ri['velocity_rms']/(r0['velocity_rms']+1e-9):.1f}x",
                f"{ri['motion_bw_hz']-r0['motion_bw_hz']:+.2f}", "--"])
            rcols.append(['#E8F5E9']*ncol)

    fig3_h = max(3.5, 1.5 + len(rows)*0.55)
    fig3, ax3 = plt.subplots(figsize=(18, fig3_h), facecolor=BG)
    ax3.axis('off')
    ax3.set_title("AcousticProbe -- Full Metrics Summary", fontsize=15, fontweight='bold', color='#2C3E50', pad=15)
    tbl = ax3.table(cellText=rows, colLabels=col_h, cellColours=rcols,
                    colColours=['#D5D8DC']*ncol, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.9)
    for j in range(ncol): tbl[0,j].set_text_props(fontweight='bold', fontsize=7.5)
    for i in range(len(rows)): tbl[i+1,0].set_text_props(fontweight='bold')

    fig3.text(0.32, 0.97, "--- Breathing ---", ha='center', fontsize=9, color='#2980B9', fontweight='bold', style='italic')
    fig3.text(0.72, 0.97, "--- Motion ---", ha='center', fontsize=9, color='#E67E22', fontweight='bold', style='italic')

    if nc > 1:
        wins = [0]*nc
        for key,hb in ALL_WIN:
            vals = [r[key] for r in results]
            wins[np.argmax(vals) if hb else np.argmin(vals)] += 1
        ob = np.argmax(wins)
        fig3.text(0.5, 0.02, f"BEST OVERALL: {labels[ob]} (wins {wins[ob]}/{len(ALL_WIN)} across breathing + motion)",
                  ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2C3E50',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F6E8', edgecolor='#27AE60', alpha=0.9, linewidth=1.5))

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    out3 = out_dir / "metrics_table.png"
    plt.savefig(str(out3), dpi=150, facecolor=BG, bbox_inches='tight')
    print(f"Saved -> {out3}")

    # == Terminal table ==
    W = 135
    print("\n" + "="*W)
    print(f"  AcousticProbe -- Full Performance Metrics   (Baseline: {labels[0]})")
    print("="*W)
    print(f"{'Condition':<16}"
          f"{'RspSNR':>8} {'CSR':>7} {'Coher':>7} {'BrConf':>7} {'RspAmp':>8} {'BPM':>6}"
          f"  |"
          f"{'MotSNR':>8} {'DynRng':>8} {'VelRMS':>8} {'MotBW':>7} {'Jitter':>8}")
    print("-"*W)
    for i,r in enumerate(results):
        bs = f"{r['resp_rate']:.1f}" if r['resp_rate']>0 else "N/A"
        print(f"{labels[i]:<16}"
              f"{r['resp_snr_db']:>8.1f} {r['csr_db']:>7.1f} {r['coherence']:>7.3f} "
              f"{r['breath_confidence']:>7.1f} {r['disp_amplitude_mm']:>8.2f} {bs:>6}"
              f"  |"
              f"{r['motion_snr_db']:>8.1f} {r['motion_dyn_range_mm']:>8.2f} "
              f"{r['velocity_rms']:>8.1f} {r['motion_bw_hz']:>7.2f} "
              f"{r['tracking_jitter_cm']:>8.1f}")
    print("-"*W)
    if nc > 1:
        print(f"\n  Improvement vs {labels[0]}:")
        print(f"  {'':16} {'dRSNR':>8} {'dCSR':>7} {'dAmp':>8}   | {'dMSNR':>8} {'dDyn':>8} {'dVel':>8}")
        print(f"  {'-'*70}")
        for i in range(1,nc):
            r0,ri = results[0],results[i]
            print(f"  {labels[i]:<16} "
                  f"{ri['resp_snr_db']-r0['resp_snr_db']:>+8.1f} dB "
                  f"{ri['csr_db']-r0['csr_db']:>+7.1f} dB "
                  f"{ri['disp_amplitude_mm']/(r0['disp_amplitude_mm']+1e-9):>8.1f}x"
                  f"   | "
                  f"{ri['motion_snr_db']-r0['motion_snr_db']:>+8.1f} dB "
                  f"{ri['motion_dyn_range_mm']/(r0['motion_dyn_range_mm']+1e-9):>8.1f}x "
                  f"{ri['velocity_rms']/(r0['velocity_rms']+1e-9):>8.1f}x")
        wins = [0]*nc
        for key,hb in ALL_WIN:
            vals = [r[key] for r in results]
            wins[np.argmax(vals) if hb else np.argmin(vals)] += 1
        ob = np.argmax(wins)
        print(f"\n  >>> BEST OVERALL: {labels[ob]} (wins {wins[ob]}/{len(ALL_WIN)} across breathing + motion)")
    print("="*W)

    # Save terminal output as text log
    log_lines = []
    log_lines.append(f"AcousticProbe Analysis - {ts}")
    log_lines.append(f"Baseline: {labels[0]} ({Path(paths[0]).name})")
    log_lines.append(f"Files: {[Path(p).name for p in paths]}")
    log_lines.append("")
    for i, r in enumerate(results):
        log_lines.append(f"{labels[i]}:")
        for k in ["resp_snr_db","csr_db","coherence","breath_confidence",
                   "disp_amplitude_mm","resp_rate","motion_snr_db",
                   "motion_dyn_range_mm","velocity_rms","motion_bw_hz","tracking_jitter_cm"]:
            log_lines.append(f"  {k}: {r[k]:.4f}")
    (out_dir / "metrics.txt").write_text("\n".join(log_lines))

    # Save metrics as JSON for programmatic access
    json_data = {}
    for i, r in enumerate(results):
        json_data[labels[i]] = {k: float(r[k]) for k in
            ["resp_snr_db","csr_db","coherence","breath_confidence",
             "disp_amplitude_mm","resp_rate","motion_snr_db",
             "motion_dyn_range_mm","velocity_rms","motion_bw_hz","tracking_jitter_cm"]}
    json_data["_meta"] = {"timestamp": ts, "baseline": labels[0],
                          "files": [Path(p).name for p in paths]}
    (out_dir / "metrics.json").write_text(json.dumps(json_data, indent=2))

    print(f"\n  All outputs saved to: {out_dir}/")
    plt.show()


if __name__ == "__main__":
    n = len(sys.argv) - 1
    if n >= 2:
        paths = sys.argv[1:]
        lb = LABELS[:n] if n<=len(LABELS) else [Path(p).stem for p in paths]
        main(paths, lb)
    else:
        dl = Path.home()/"Downloads"
        ws = sorted(dl.glob("fmcw_*.wav"), key=lambda p:p.stat().st_mtime)[-3:]
        if len(ws)<2:
            print(f"Found {len(ws)} WAVs. Need >=2.")
            print("Usage: python compare_conditions.py <bare.wav> <tubeA.wav> [...]")
            print("\nNOTE: First file = baseline (bare phone)")
            sys.exit(1)
        lb = LABELS[:len(ws)]
        print("Auto-detected (oldest->newest):")
        for i,w in enumerate(ws): print(f"  [{lb[i]}] {w.name}")
        main([str(w) for w in ws], lb)