"""
AcousticProbe — Comparison GUI (auto duration)

What this version does:
- Select multiple WAV files
- Automatically reads each WAV duration
- You only type the manual ground-truth breath count for each recording
- GT BPM = breath_count / auto_duration_seconds * 60
- Calls compare_conditions.main(paths, labels, gt_bpm, mode=mode)

Usage:
    python run_comparison.py
"""

import sys
import threading
import importlib
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

sys.path.insert(0, str(Path(__file__).parent))

MAX_FILES = 6

MODE_OPTIONS = {
    "No-subject control": "control",
    "Structure comparison": "structure",
}

DEFAULT_LABELS = {
    "control": [
        "No-subject control",
        "Tube A + breathing",
        "Tube B + breathing",
        "Tube C + breathing",
        "Tube D + breathing",
        "Tube E + breathing",
    ],
    "structure": [
        "Bare + breathing",
        "Tube A + breathing",
        "Tube B + breathing",
        "Tube C + breathing",
        "Tube D + breathing",
        "Tube E + breathing",
    ],
}


def get_wav_duration_seconds(path: str) -> float:
    """
    Read WAV duration without relying on Python's wave module.

    Some iPhone / macOS exported WAV files use WAVE_FORMAT_EXTENSIBLE,
    which can make wave.open() throw:
        unknown extended format: 00000003-0000-0010-8000-00aa00389b71

    For duration we only need the WAV fmt chunk byte_rate and data chunk size:
        duration = data_size / byte_rate
    This works for PCM, float WAV, and extensible WAV.
    """
    import struct

    with open(path, "rb") as f:
        riff = f.read(12)
        if len(riff) < 12 or riff[0:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise ValueError("Not a valid RIFF/WAVE file")

        byte_rate = None
        data_size = None

        while True:
            header = f.read(8)
            if len(header) < 8:
                break

            chunk_id, chunk_size = struct.unpack("<4sI", header)
            chunk_start = f.tell()

            if chunk_id == b"fmt ":
                fmt = f.read(min(chunk_size, 16))
                if len(fmt) < 16:
                    raise ValueError("Invalid WAV fmt chunk")

                _, _, _, byte_rate, _, _ = struct.unpack("<HHIIHH", fmt[:16])

                if byte_rate <= 0:
                    raise ValueError("Invalid WAV byte rate")

            elif chunk_id == b"data":
                data_size = chunk_size

            f.seek(chunk_start + chunk_size + (chunk_size % 2))

            if byte_rate is not None and data_size is not None:
                return data_size / float(byte_rate)

    raise ValueError("Could not find WAV fmt/data chunks")


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    sec = seconds - minutes * 60
    return f"{minutes}:{sec:04.1f}"


def safe_name(text: str) -> str:
    bad = '<>:"/\\|?*'
    for ch in bad:
        text = text.replace(ch, "_")
    return text.strip() or "condition"


class ComparisonApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AcousticProbe — Condition Comparison")
        self.root.resizable(False, False)
        self.root.configure(bg="#F5F6FA")

        # each item: path, label_var, duration_sec, duration_var, breaths_var, bpm_var
        self.files = []
        self._build_ui()

    def _current_mode(self):
        return MODE_OPTIONS[self.mode_label_var.get()]

    def _default_label_for_index(self, idx):
        mode = self._current_mode()
        defaults = DEFAULT_LABELS[mode]
        return defaults[idx] if idx < len(defaults) else f"Condition {idx + 1}"

    def _build_ui(self):
        hdr = tk.Frame(self.root, bg="#F5F6FA", padx=30, pady=16)
        hdr.pack(fill="x")

        tk.Label(hdr, text="AcousticProbe", font=("Helvetica", 18, "bold"), bg="#F5F6FA", fg="#2C3E50").pack()
        tk.Label(hdr, text="Multi-Condition FMCW Comparison — Per-file Ground Truth", font=("Helvetica", 12), bg="#F5F6FA", fg="#7F8C8D").pack()

        mode_frame = tk.LabelFrame(
            self.root, text="Experiment Mode", font=("Helvetica", 11, "bold"),
            bg="#F5F6FA", fg="#2C3E50", padx=14, pady=10
        )
        mode_frame.pack(fill="x", padx=30, pady=(0, 12))

        self.mode_label_var = tk.StringVar(value="Structure comparison")

        tk.Radiobutton(
            mode_frame,
            text="Structure comparison: first file is Bare iPhone + human breathing baseline.",
            variable=self.mode_label_var, value="Structure comparison",
            command=self._on_mode_changed, bg="#F5F6FA", fg="#2C3E50",
            anchor="w", justify="left", font=("Helvetica", 10)
        ).pack(fill="x", anchor="w")

        tk.Radiobutton(
            mode_frame,
            text="No-subject control: first file is empty scene / no human. Use this for false-positive testing.",
            variable=self.mode_label_var, value="No-subject control",
            command=self._on_mode_changed, bg="#F5F6FA", fg="#2C3E50",
            anchor="w", justify="left", font=("Helvetica", 10)
        ).pack(fill="x", anchor="w")

        self.list_frame = tk.Frame(self.root, bg="#F5F6FA", padx=30)
        self.list_frame.pack(fill="x")

        col_hdr = tk.Frame(self.list_frame, bg="#F5F6FA")
        col_hdr.pack(fill="x", pady=(0, 4))
        headers = [
            ("File", 26, "w"),
            ("Label", 20, "w"),
            ("Duration\n(auto)", 10, "center"),
            ("Breaths\n(manual)", 9, "center"),
            ("GT BPM\n(auto)", 8, "center"),
            ("Order", 12, "center"),
        ]
        for text, width, anchor in headers:
            tk.Label(
                col_hdr, text=text, width=width, anchor=anchor,
                font=("Helvetica", 10, "bold"), bg="#F5F6FA", fg="#95A5A6"
            ).pack(side="left", padx=(4, 0))

        self.rows_container = tk.Frame(self.list_frame, bg="#F5F6FA")
        self.rows_container.pack(fill="x")

        self.placeholder = tk.Label(
            self.rows_container,
            text="No files added yet.\nClick 'Add WAV Files' to select 2–6 files.\nNo global GT field: enter breath count per row.",
            font=("Helvetica", 11), bg="#F5F6FA", fg="#BDC3C7", pady=20
        )
        self.placeholder.pack(fill="x")

        btn_frame = tk.Frame(self.root, bg="#F5F6FA", padx=30, pady=16)
        btn_frame.pack(fill="x")

        self.add_btn = tk.Button(
            btn_frame, text="Add WAV Files", command=self._add_files,
            font=("Helvetica", 11, "bold"), bg="#3498DB", fg="white",
            relief="flat", padx=16, pady=8, cursor="hand2"
        )
        self.add_btn.pack(side="left")

        self.run_btn = tk.Button(
            btn_frame, text="Run Comparison", command=self._run,
            font=("Helvetica", 11, "bold"), bg="#2ECC71", fg="white",
            relief="flat", padx=16, pady=8, cursor="hand2", state="disabled"
        )
        self.run_btn.pack(side="right")

        self.status_var = tk.StringVar(value="Add at least 2 WAV files to compare.")
        tk.Label(
            self.root, textvariable=self.status_var, font=("Helvetica", 10),
            bg="#F5F6FA", fg="#7F8C8D", wraplength=760, justify="left"
        ).pack(fill="x", padx=30, pady=(0, 16))

    def _on_mode_changed(self):
        for i, item in enumerate(self.files):
            item["label_var"].set(self._default_label_for_index(i))
        self._redraw_rows()
        self._update_state()

    def _update_bpm_for_item(self, item):
        raw = item["breaths_var"].get().strip()
        if not raw:
            item["bpm_var"].set("")
            return
        try:
            breaths = float(raw)
            if breaths < 0:
                raise ValueError
            bpm = breaths / item["duration_sec"] * 60.0
            item["bpm_var"].set(f"{bpm:.2f}")
        except ValueError:
            item["bpm_var"].set("ERR")

    def _bind_bpm_updates(self, item):
        def _callback(*_):
            self._update_bpm_for_item(item)
        item["breaths_var"].trace_add("write", _callback)

    def _add_files(self):
        remaining = MAX_FILES - len(self.files)
        if remaining <= 0:
            messagebox.showwarning("Limit", f"Maximum {MAX_FILES} files.")
            return

        paths = filedialog.askopenfilenames(
            title="Select FMCW WAV recordings",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if not paths:
            return

        existing = {item["path"] for item in self.files}
        added = 0
        for path in paths:
            if added >= remaining:
                break
            if path in existing:
                continue
            try:
                duration_sec = get_wav_duration_seconds(path)
            except Exception as e:
                messagebox.showerror("Could not read WAV duration", f"{Path(path).name}\n\n{e}")
                continue

            idx = len(self.files)
            item = {
                "path": path,
                "label_var": tk.StringVar(value=self._default_label_for_index(idx)),
                "duration_sec": duration_sec,
                "duration_var": tk.StringVar(value=format_duration(duration_sec)),
                "breaths_var": tk.StringVar(value=""),
                "bpm_var": tk.StringVar(value=""),
            }
            self._bind_bpm_updates(item)
            self.files.append(item)
            added += 1

        self._redraw_rows()
        self._update_state()

    def _remove_file_at(self, idx):
        if 0 <= idx < len(self.files):
            self.files.pop(idx)
            self._redraw_rows()
            self._update_state()

    def _move_file(self, idx, delta):
        j = idx + delta
        if 0 <= idx < len(self.files) and 0 <= j < len(self.files):
            self.files[idx], self.files[j] = self.files[j], self.files[idx]
            self._redraw_rows()
            self._update_state()

    def _redraw_rows(self):
        for child in self.rows_container.winfo_children():
            child.destroy()

        if not self.files:
            self.placeholder = tk.Label(
                self.rows_container,
                text="No files added yet.\nClick 'Add WAV Files' to select 2–6 files.\nNo global GT field: enter breath count per row.",
                font=("Helvetica", 11), bg="#F5F6FA", fg="#BDC3C7", pady=20
            )
            self.placeholder.pack(fill="x")
            return

        for idx, item in enumerate(self.files):
            row = tk.Frame(self.rows_container, bg="white", pady=6, padx=6)
            row.pack(fill="x", pady=3)

            fname = Path(item["path"]).name
            if len(fname) > 24:
                fname = fname[:21] + "..."

            tk.Label(row, text=f"{idx + 1}. {fname}", width=26, anchor="w", font=("Helvetica", 10), bg="white", fg="#34495E").pack(side="left", padx=(4, 0))
            tk.Entry(row, textvariable=item["label_var"], width=20, font=("Helvetica", 11), relief="solid", bd=1).pack(side="left", padx=(4, 0))
            tk.Label(row, textvariable=item["duration_var"], width=10, font=("Helvetica", 10, "bold"), bg="white", fg="#2C3E50").pack(side="left", padx=(4, 0))
            tk.Entry(row, textvariable=item["breaths_var"], width=9, font=("Helvetica", 11), justify="center", relief="solid", bd=1).pack(side="left", padx=(4, 0))
            tk.Label(row, textvariable=item["bpm_var"], width=8, font=("Helvetica", 10, "bold"), bg="white", fg="#2C3E50").pack(side="left", padx=(4, 0))

            order_frame = tk.Frame(row, bg="white")
            order_frame.pack(side="left", padx=(6, 0))
            tk.Button(order_frame, text="↑", command=lambda i=idx: self._move_file(i, -1), width=2, bg="#ECF0F1", relief="flat", state="normal" if idx > 0 else "disabled").pack(side="left", padx=(0, 2))
            tk.Button(order_frame, text="↓", command=lambda i=idx: self._move_file(i, 1), width=2, bg="#ECF0F1", relief="flat", state="normal" if idx < len(self.files) - 1 else "disabled").pack(side="left", padx=(0, 2))
            tk.Button(row, text="✕", command=lambda i=idx: self._remove_file_at(i), fg="#E74C3C", bg="white", relief="flat", cursor="hand2", padx=4).pack(side="right")

    def _update_state(self):
        n = len(self.files)
        mode = self._current_mode()
        if n >= 2:
            self.run_btn.config(state="normal")
            if mode == "control":
                self.status_var.set(f"{n} files loaded. First file will be treated as no-subject control. Duration is read automatically; enter breaths only for human-breathing files.")
            else:
                self.status_var.set(f"{n} files loaded. First file will be treated as Bare + breathing baseline. Duration is read automatically; enter breath counts only.")
        elif n == 1:
            self.run_btn.config(state="disabled")
            self.status_var.set("Add at least 1 more file. Need ≥2 files to compare.")
        else:
            self.run_btn.config(state="disabled")
            self.status_var.set("Add at least 2 WAV files to compare.")
        self.add_btn.config(state="disabled" if n >= MAX_FILES else "normal")

    def _parse_gt_bpms(self):
        values = []
        for i, item in enumerate(self.files):
            raw = item["breaths_var"].get().strip()
            if not raw:
                values.append(None)
                continue
            bpm_txt = item["bpm_var"].get().strip()
            if bpm_txt == "ERR" or not bpm_txt:
                messagebox.showerror("Invalid ground truth input", f"Row {i + 1}: enter a valid breath count, e.g. 22 or 24.")
                return "ERROR"
            values.append(float(bpm_txt))
        return values

    def _run(self):
        if len(self.files) < 2:
            messagebox.showwarning("Need files", "Please add at least 2 WAV files.")
            return

        gt_bpm = self._parse_gt_bpms()
        if gt_bpm == "ERROR":
            return

        mode = self._current_mode()
        paths = [item["path"] for item in self.files]
        labels = [item["label_var"].get().strip() or f"Condition {i + 1}" for i, item in enumerate(self.files)]

        folder_hint = f"{mode}_{safe_name(labels[0])}"
        self.run_btn.config(state="disabled")
        self.status_var.set(f"Running comparison... Expected output folder in current running directory: {folder_hint}")

        def do_run():
            try:
                import compare_conditions
                importlib.reload(compare_conditions)
                compare_conditions.main(paths, labels, gt_bpm, mode=mode)
                self.status_var.set("Done. Figures and metrics saved to the current running directory / output folder.")
            except Exception as e:
                import traceback
                traceback.print_exc()
                messagebox.showerror("Error", str(e))
                self.status_var.set("Failed. See terminal for traceback.")
            finally:
                self.run_btn.config(state="normal")

        threading.Thread(target=do_run, daemon=True).start()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    ComparisonApp().run()
