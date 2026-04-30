"""
AcousticProbe — GUI launcher
Double-click AcousticProbe Analyzer.app, or run:  python run_analysis.py

Select one or more WAV files:
  - Each file is analyzed individually and its PNG opened immediately.
  - If 2+ files are selected, a comparison figure is generated at the end.
"""

import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def run_batch(paths: list[str], target: float | None, thresh: float,
              status_var: tk.StringVar, btn: tk.Button):
    import importlib, subprocess
    import matplotlib.pyplot as plt
    import analyze_fmcw
    import compare_conditions
    importlib.reload(analyze_fmcw)
    importlib.reload(compare_conditions)

    total = len(paths)
    failed = []

    # ── Step 1: individual analysis ──────────────────────────────────────────
    for i, wav_path in enumerate(paths, 1):
        name = Path(wav_path).name
        status_var.set(f"[{i}/{total}] Analyzing {name}…")
        try:
            analyze_fmcw.process(wav_path, target_dist=target,
                                 motion_threshold=thresh)
            out = Path(wav_path).with_name(Path(wav_path).stem + "_analyzed.png")
            subprocess.Popen(["open", str(out)])
        except Exception as e:
            failed.append(f"{name}: {e}")

    # ── Step 2: comparison figures (only when 2+ files) ──────────────────────
    if len(paths) >= 2:
        status_var.set("Generating comparison figures…")
        try:
            labels = [Path(p).stem for p in paths]

            # Suppress plt.show() — running in a background thread
            _orig_show = plt.show
            plt.show = lambda: None
            compare_conditions.main(paths, labels)
            plt.show = _orig_show
            plt.close("all")

            first_dir = Path(paths[0]).parent
            for fname in ["metrics_comparison.png",
                          "signal_comparison.png",
                          "metrics_table.png"]:
                out = first_dir / fname
                if out.exists():
                    subprocess.Popen(["open", str(out)])
        except Exception as e:
            failed.append(f"Comparison: {e}")

    # ── Done ─────────────────────────────────────────────────────────────────
    if failed:
        messagebox.showerror("Errors", "\n".join(failed))
        status_var.set(f"Done with {len(failed)} error(s).")
    else:
        suffix = " + comparison" if len(paths) >= 2 else ""
        status_var.set(f"Done — {total} file(s) analyzed{suffix}.")
    btn.config(state="normal")


def pick_and_run():
    paths = filedialog.askopenfilenames(
        title="Select FMCW recording(s)",
        filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
    )
    if not paths:
        return

    raw = target_var.get().strip()
    target = None
    if raw:
        try:
            target = float(raw)
        except ValueError:
            messagebox.showerror("Error", f"Invalid target distance: '{raw}'")
            return

    thresh = thresh_var.get()
    n = len(paths)
    status_var.set(f"{n} file(s) queued…")
    btn.config(state="disabled")
    threading.Thread(
        target=run_batch,
        args=(list(paths), target, thresh, status_var, btn),
        daemon=True,
    ).start()


# ── UI ────────────────────────────────────────────────────────────────────────

root = tk.Tk()
root.title("AcousticProbe Analyzer")
root.resizable(False, False)

frame = tk.Frame(root, padx=30, pady=24)
frame.pack()

tk.Label(frame, text="AcousticProbe",
         font=("Helvetica", 16, "bold")).pack(pady=(0, 4))
tk.Label(frame,
         text="Select one or more WAV recordings.\n"
              "Each gets its own analysis PNG; 2+ files also produce a comparison.",
         font=("Helvetica", 11), fg="#555", justify="center").pack(pady=(0, 16))

# Target distance row
dist_row = tk.Frame(frame)
dist_row.pack(pady=(0, 14))
tk.Label(dist_row, text="Target distance (m, optional):",
         font=("Helvetica", 11)).pack(side="left", padx=(0, 8))
target_var = tk.StringVar()
tk.Entry(dist_row, textvariable=target_var, width=6,
         font=("Helvetica", 12)).pack(side="left")
tk.Label(dist_row, text="e.g. 0.5",
         font=("Helvetica", 10), fg="#999").pack(side="left", padx=(6, 0))

# Motion threshold slider row
thresh_row = tk.Frame(frame)
thresh_row.pack(pady=(0, 14))
tk.Label(thresh_row, text="Motion threshold (× noise floor):",
         font=("Helvetica", 11)).pack(side="left", padx=(0, 8))
thresh_var = tk.DoubleVar(value=3.0)
tk.Scale(thresh_row, variable=thresh_var,
         from_=1.0, to=5.0, resolution=0.5,
         orient="horizontal", length=160,
         font=("Helvetica", 10)).pack(side="left")
tk.Label(thresh_row, text="low=sensitive  high=stable",
         font=("Helvetica", 9), fg="#999").pack(side="left", padx=(8, 0))

btn = tk.Button(frame, text="Open WAV file(s)…", command=pick_and_run,
                font=("Helvetica", 13), padx=16, pady=8,
                bg="#0A84FF", fg="white", relief="flat", cursor="hand2")
btn.pack()

status_var = tk.StringVar(value="No file loaded.")
tk.Label(frame, textvariable=status_var,
         font=("Helvetica", 10), fg="#777").pack(pady=(14, 0))

root.mainloop()
