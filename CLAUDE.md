# AcousticProbe — Project Context

MIT 6.808 课程项目。用 iPhone 内置扬声器/麦克风发射 18–22 kHz FMCW 超声信号，
通过 3D 打印管状结构增强特定方向的人体活动检测（HAR）能力。

## 当前推进方向

**单一方案：管状导波结构（directional tube waveguide）**
不再做 proposal 中的三个 prototype（spiral / Helmholtz / quarter-wave），
改为 3D 打印一个简单管子，通过几何遮蔽提升 FMCW 方向性。

## 关键参数

- 信号：18–22 kHz，带宽 B=4 kHz，chirp 周期 T=20 ms，采样率 48 kHz
- 中心频率 20 kHz，λ = 17.2 mm
- 管子接受角公式：`θ = arctan(D / 2L)`（θ 为半角）
- 推荐尺寸：内径 15–20 mm，管长 3–5 cm（对应 ±8–14°）
- 管长引入固定 range offset（分析时需校准）

## 文件结构

```
AcousticProbe/
├── AcousticProbe/          ← Xcode iOS 项目（Swift/SwiftUI）
│   └── AcousticProbe/
│       ├── FMCWEngine.swift    chirp 生成 + AVAudioEngine 录音
│       ├── ContentView.swift   UI
│       └── AcousticProbeApp.swift
├── analyze_fmcw.py         主 DSP 流水线（4 张图输出）
├── run_analysis.py         tkinter GUI 启动器（运行即弹出文件选择）
├── test_signal.py          快速验证（3 张图）
├── compare_conditions.py   多管长条件对比
└── notes/
    └── 2026-04-25_session.md   详细设计讨论与实验分析记录
```

## 已知设计决策

### 算法层（analyze_fmcw.py）

**相位漂移修复（2026-04-25）：**
原代码用固定 `median_bin` 提取相位，目标移动导致累积漂移。
修复方案：`signal.detrend` → 高通 0.05 Hz → 带通 0.1–3 Hz（零相位 `sosfiltfilt`）。
相关函数：`highpass_zp`、`bandpass_zp`。

**呼吸率提取：** 0.1–1 Hz 内找 FFT 峰值（6–60 次/分）。

### 实验结论（三组管长对比，2026-04-25）

| 管长 | Range profile 峰值 | 相对 SNR | 呼吸波形 |
|------|--------------------|----------|----------|
| 短管 | 0.0015 | 1× | 不可靠 |
| 中管 | 0.003  | 2× | 有扰动 |
| 长管 | 0.025  | **17×** | **清晰，~13 次/分** |

**结论：管越长 SNR 越高，方向性增强效果显著。**

## 使用方法

```bash
# GUI 分析（推荐）
python run_analysis.py

# 命令行分析
python analyze_fmcw.py fmcw_<timestamp>.wav

# 快速验证
python test_signal.py fmcw_<timestamp>.wav
```

## 待办

- [ ] 3D 打印不同管长（建议 3 cm / 5 cm / 8 cm）做控制变量实验
- [ ] 固定距离 1 m，人静止，量化三种管长的 SNR 差异
- [ ] 用 compare_conditions.py 做正式 benchmarking
- [ ] 考虑加呼吸率置信度评估（频谱峰值显著性）
