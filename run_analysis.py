"""
AcousticProbe — GUI launcher
Double-click AcousticProbe Analyzer.app, or run:  python run_analysis.py
"""

import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def run_analysis(wav_path: str, target: float | None, thresh: float,
                 status_var: tk.StringVar, btn: tk.Button):
    try:
        status_var.set("Analyzing…")
        import analyze_fmcw
        import importlib
        importlib.reload(analyze_fmcw)          # pick up any code changes
        analyze_fmcw.process(wav_path, target_dist=target, motion_threshold=thresh)
        out = Path(wav_path).with_name(Path(wav_path).stem + "_analyzed.png")
        status_var.set(f"Done — {out.name}")
        import subprocess
        subprocess.Popen(["open", str(out)])
    except Exception as e:
        messagebox.showerror("Error", str(e))
        status_var.set("Failed.")
    finally:
        btn.config(state="normal")


def pick_and_run():
    path = filedialog.askopenfilename(
        title="Select FMCW recording",
        filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
    )
    if not path:
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
    status_var.set(f"Loaded: {Path(path).name}")
    btn.config(state="disabled")
    threading.Thread(
        target=run_analysis,
        args=(path, target, thresh, status_var, btn),
        daemon=True,
    ).start()


root = tk.Tk()
root.title("AcousticProbe Analyzer")
root.resizable(False, False)

frame = tk.Frame(root, padx=30, pady=24)
frame.pack()

tk.Label(frame, text="AcousticProbe", font=("Helvetica", 16, "bold")).pack(pady=(0, 4))
tk.Label(frame, text="Select a WAV recording to run the full FMCW analysis.",
         font=("Helvetica", 11), fg="#555").pack(pady=(0, 16))

# Target distance row
dist_row = tk.Frame(frame)
dist_row.pack(pady=(0, 14))
tk.Label(dist_row, text="Target distance (m, optional):",
         font=("Helvetica", 11)).pack(side="left", padx=(0, 8))
target_var = tk.StringVar()
tk.Entry(dist_row, textvariable=target_var, width=6,
         font=("Helvetica", 12)).pack(side="left")
tk.Label(dist_row, text="e.g. 0.5", font=("Helvetica", 10), fg="#999").pack(side="left", padx=(6, 0))

# Motion threshold slider row
thresh_row = tk.Frame(frame)
thresh_row.pack(pady=(0, 14))
tk.Label(thresh_row, text="Motion threshold (× noise floor):",
         font=("Helvetica", 11)).pack(side="left", padx=(0, 8))
thresh_var = tk.DoubleVar(value=3.0)
thresh_slider = tk.Scale(thresh_row, variable=thresh_var,
                         from_=1.0, to=5.0, resolution=0.5,
                         orient="horizontal", length=160,
                         font=("Helvetica", 10))
thresh_slider.pack(side="left")
tk.Label(thresh_row, text="low=sensitive  high=stable",
         font=("Helvetica", 9), fg="#999").pack(side="left", padx=(8, 0))

btn = tk.Button(frame, text="Open WAV file…", command=pick_and_run,
                font=("Helvetica", 13), padx=16, pady=8,
                bg="#0A84FF", fg="white", relief="flat", cursor="hand2")
btn.pack()

status_var = tk.StringVar(value="No file loaded.")
tk.Label(frame, textvariable=status_var, font=("Helvetica", 10), fg="#777").pack(pady=(14, 0))

root.mainloop()
