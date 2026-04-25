"""
AcousticProbe — GUI launcher
Run:  python run_analysis.py
"""

import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from pathlib import Path


def run_analysis(wav_path: str, status_var: tk.StringVar, btn: tk.Button):
    try:
        status_var.set("Analyzing...")
        import analyze_fmcw
        analyze_fmcw.process(wav_path)
        out = Path(wav_path).with_name(Path(wav_path).stem + "_analyzed.png")
        status_var.set(f"Done — saved to {out.name}")
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
    status_var.set(f"Loaded: {Path(path).name}")
    btn.config(state="disabled")
    threading.Thread(target=run_analysis, args=(path, status_var, btn), daemon=True).start()


root = tk.Tk()
root.title("AcousticProbe Analyzer")
root.resizable(False, False)

frame = tk.Frame(root, padx=30, pady=24)
frame.pack()

tk.Label(frame, text="AcousticProbe", font=("Helvetica", 16, "bold")).pack(pady=(0, 4))
tk.Label(frame, text="Select a WAV recording to run the full FMCW analysis.",
         font=("Helvetica", 11), fg="#555").pack(pady=(0, 16))

btn = tk.Button(frame, text="Open WAV file…", command=pick_and_run,
                font=("Helvetica", 13), padx=16, pady=8,
                bg="#0A84FF", fg="white", relief="flat", cursor="hand2")
btn.pack()

status_var = tk.StringVar(value="No file loaded.")
tk.Label(frame, textvariable=status_var, font=("Helvetica", 10), fg="#777").pack(pady=(14, 0))

root.mainloop()
