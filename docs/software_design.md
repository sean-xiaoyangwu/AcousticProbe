# AcousticProbe — Software Design (Slides 参考)

两页 slides 用于 6.808 presentation 的软件设计部分。
项目主线：**呼吸感知（primary）+ motion-aware 有效性判据（supporting）**。

---

## Slide 1 — System Pipeline

**目标：30 秒内让人看懂数据流和处理流水线。**

### 数据采集 (iOS)

```
[iPhone speaker] ──18-22 kHz FMCW──> [3D-printed tube] ──> [target]
[iPhone mic]    <──reflected echo──────────────────────────
        │
        ▼
   WAV file + JSON sidecar (params)
```

iOS 端**只做采集**，DSP 全在离线 Python 完成。
理由：算法迭代快、参数完全可重现、便于 batch 对比实验。

### Python 处理流水线

```
WAV + JSON
   │
   ▼
Bandpass [17.5–22.5 kHz]
   │
   ▼
Cross-correlation align     ←  补偿 AVAudioEngine buffer jitter (~2000 samples)
   │
   ▼
Reshape → (chirps × N samples)
   │
   ▼
Mix (RX × TX template) → LP 5 kHz       ←  de-chirping, beat freq ↔ range
   │
   ▼
Hanning-windowed FFT                    ←  sidelobes −13 → −31 dB
   │
   ▼
Range profile (chirps × range_bins)
   │
   ▼
Frame-diff (MTI background subtraction)
   │
   ▼
Target-bin selection (mean of frame-diff, ≥0.25 m)
   │
   ├──── |FFT|  ───────> Motion channel (validity gate)
   │
   └──── ∠FFT  ────────> Breathing channel (primary)
```

### 关键参数

| 参数 | 值 | 含义 |
|---|---|---|
| `f0–f1` | 18–22 kHz | 不可闻 + iPhone 喇叭/麦响应良好 |
| `B` | 4 kHz | 带宽 → 真实距离分辨率 |
| `T` | 20 ms | chirp 周期 → phase 采样 50 Hz |
| `fs` | 48 kHz | iPhone 标准采样率 |
| `Δr = c/(2B)` | **4.3 cm** | 振幅通道分辨率（硬限制） |
| `λ/2` | **8.6 mm** | 相位通道单 chirp wrap 上限 |

### 文件结构（角标）

```
iOS App/FMCWEngine.swift   — capture only, AVAudioEngine
analyze_fmcw.py            — single-file pipeline
compare_conditions.py      — multi-condition benchmark
run_analysis.py            — tkinter GUI launcher
```

---

## Slide 2 — Breathing Extraction with Motion-Aware Gating

**目标：突出"呼吸是产品，motion 是使呼吸可信的判据"。**

### 核心论点（slide 顶部一行）

> Amplitude resolves WHERE; phase resolves how MUCH.
> Motion channel gates breathing channel's validity.

### 双通道分工

| | Motion channel | Breathing channel (primary) |
|---|---|---|
| 数据源 | `\|FFT\|` + frame-diff | `∠FFT` + phase unwrap |
| 分辨率 | 4.3 cm | <1 mm |
| 量程 | 任意大位移 | 单 chirp <λ/2 ≈ 8.6 mm |
| 角色 | **有效性判据 + 管子方向性验证** | **主输出（呼吸波形 + 频率）** |

### 为什么这样分（不能合并）

- 呼吸位移 ~mm 级 → 振幅通道**看不见**（远低于 4.3 cm bin）
- 大手势 ~50 cm → 相位 unwrap **失败**（要展 ~63 圈，累积误差爆炸）
- 同一个 `target_bin` 同时供两路处理 — 测的是同一个人，只是不同尺度

### Motion channel 在呼吸项目中的具体作用

1. **有效性闸门**：`win_max > MOTION_THRESH` 的帧标为 active，
   这些段呼吸率读数标 N/A 或低置信度（避免 unwrap 误差污染输出）
2. **管子方向性的物理证据**：横向挥手实验激活 0.8% vs 前倾 9.3%，
   这是"管子真的提供径向选择"的最强数据（呼吸通道得不到此结论）
3. **proposal scope 保留**：原 proposal 包含 HAR，砍掉 motion = scope shrink

### 支撑技术（小列表）

- **Zero-phase filtering** (`sosfiltfilt`)：保留呼吸波形形状（吸/呼不对称信息）
- **Detrend → HP 0.05 Hz → BP 0.1–3 Hz**：三层去趋势消除累积漂移
- **Noise-gated argmax + hold-last-value**：安静期不随机跳，
  曲线呈"平台 + 跳跃"结构匹配离散手势
- **Frame-diff (MTI)**：静态杂波（墙、管壁）相邻帧相减归零

### 推荐配图

- 主图：长管录音 (`fmcw_1777232317`) 的呼吸波形 — 平稳 ~6 bpm 周期
- 配图：上方 motion activity bar，红色段标 "INVALID"
- 角图：wave 实验的 motion trace（几乎全平，证明径向选择性）

---

## 取舍参考

### 必须保留

| 项 | 原因 |
|---|---|
| Motion channel 的追踪曲线 | wave 实验可视化证据 |
| `compare_conditions.py` 全部 motion 指标 | 管长对比基准 |
| Cross-corr alignment | 不做就没法稳定 mix |
| Zero-phase 滤波 | 呼吸波形形状关键 |

### Slide 上可以略过的细节

| 项 | 原因 |
|---|---|
| AVAudioSession 配置 | 工程细节，slide 一句话带过 |
| GUI / 批处理脚本 | 是工具不是设计 |
| 全部 metric 名字 (CSR, coherence, …) | 压成一句"quantitative comparison metrics" |
| Phase integration overlay (红色线) | 04-30 已证明对大动作失效，不必展示 |
| `analyze_fmcw.py` 第 3 plot 的 raw argmax 灰线 | 调试用 |

---

## 演讲节奏建议

- **Slide 1 (60–90 s)**：强调"为什么 DSP 离线 Python，不在 iOS"
  → 答案：算法迭代速度 + 实验可重现 + batch 对比
- **Slide 2 (60–90 s)**：先发制人讲"为什么不能只用一个通道"
  → 这是评委最常问的问题
- 留出 30 s 展示真实输出（呼吸波形 + 管长对比图）

---

## 一句话总结

> **This is a respiration sensor.**
> Phase channel gives sub-mm breathing waveform.
> Motion channel gates its validity AND validates the tube's directional claim.
> Both share one FMCW front-end — same target, two scales.
