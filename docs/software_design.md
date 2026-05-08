# AcousticProbe — Software Design & 原理详解

本文档是项目软件设计的完整技术参考。
重点解释**呼吸监测原理**——这是项目核心。
末尾附 6.808 slide 准备指南。

> 项目主线：**呼吸感知（primary）+ motion-aware 有效性判据（supporting）**
> 同一 FMCW 前端，振幅 → 动作（cm 级），相位 → 呼吸（亚 mm 级）

---

## 目录

1. [系统总览](#1-系统总览)
2. [信号采集 — iOS 端原理](#2-信号采集--ios-端原理)
3. [FMCW 测距原理（数学推导）](#3-fmcw-测距原理数学推导)
4. [DSP 流水线分步详解](#4-dsp-流水线分步详解)
5. [★ 呼吸通道完整原理](#5--呼吸通道完整原理核心)
6. [动作通道](#6-动作通道)
7. [双通道架构](#7-双通道架构)
8. [实现地图（file:line）](#8-实现地图)
9. [Slide 准备指南](#9-slide-准备指南)

---

## 1. 系统总览

### 1.1 数据流

```
[iPhone speaker] ──18-22 kHz FMCW──> [3D-printed tube] ──> [target]
[iPhone mic]    <──reflected echo─────────────────────────────
        │
        ▼
   WAV file + JSON sidecar (params)
        │
        ▼
   Python offline DSP
        │
        ├──► Motion channel (validity gate)
        │
        └──► Breathing channel (primary)
```

iOS **只做采集**，DSP 全部在 Python 离线完成。理由：
- 算法迭代速度（不需要每次改算法都重编译 app）
- 参数完全可重现（同一份 WAV 跑两次结果一定一致）
- 便于 batch 对比实验（compare_conditions.py 一次跑多份）
- iOS 端不必在低延迟约束下做计算

### 1.2 核心参数（贯穿全文）

| 参数 | 值 | 推导/约束 |
|---|---|---|
| `f_start` | 18 000 Hz | 不可闻下限 + iPhone speaker/mic 仍有响应 |
| `f_end` | 22 000 Hz | 受限于 fs/2 = 24 kHz 的 Nyquist |
| `B` (带宽) | 4 000 Hz | f_end − f_start |
| `T` (chirp 周期) | 20 ms | phase 采样率 fs_c = 1/T = 50 Hz |
| `fs` (音频) | 48 000 Hz | iPhone 标准 |
| `N` (每 chirp 样本) | 960 | fs · T |
| `c` (声速) | 343 m/s | 室温空气 |
| `f_c` (中心频率) | 20 000 Hz | (f_start + f_end) / 2 |
| `λ_c` (中心波长) | **17.15 mm** | c / f_c |
| `Δr = c/(2B)` | **4.29 cm** | 真实距离分辨率（硬限制） |
| `λ_c/2` | **8.6 mm** | 相位单 chirp wrap 间距 |

下面所有推导都会回到这些数。

---

## 2. 信号采集 — iOS 端原理

### 2.1 Chirp 生成 (`FMCWEngine.swift:43-56`)

线性 FM chirp 的数学定义：

$$
s(t) = \sin\big(2\pi \cdot \phi(t)\big), \quad
\phi(t) = f_0 t + \tfrac{1}{2}\mu t^2, \quad
\mu = B/T
$$

瞬时频率：

$$
f_{\text{inst}}(t) = \frac{d\phi}{dt} = f_0 + \mu t
$$

代码实现：
```swift
let phi = 2 * Double.pi * (fStart * t + (B / (2 * sweepDuration)) * t * t)
ptr[i] = Float(sin(phi)) * 0.5
```

每个 chirp 共 `N = 48000 × 0.02 = 960` 个采样。

**为什么幅度 0.5？** −6 dBFS 留出余量，避免任何过冲（speaker 非线性 + 算法内部数值放大都可能削顶）。FMCW 对幅度不敏感，能量损一半换来线性度更划算。

### 2.2 录制配置 (`FMCWEngine.swift:31-39`)

```swift
.setCategory(.playAndRecord, mode: .measurement, options: .defaultToSpeaker)
```

关键点：
- `.measurement` 模式**关闭所有音频处理**：AGC、降噪、回声抑制、语音增强等。这些处理都会在频域引入非线性畸变，破坏 FMCW 线性 chirp 的相位连续性
- `.defaultToSpeaker` 强制走主扬声器（不走听筒），主扬声器 18–22 kHz 频响仍可接受
- `setPreferredSampleRate(48000)` + `setPreferredIOBufferDuration(0.005)` 配合 chirp 周期

### 2.3 发射 + 录制循环 (`FMCWEngine.swift:99-119`)

```swift
engine.inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFmt) { ... }
player.scheduleBuffer(chirpBuffer, at: nil, options: .loops)
```

关键技巧：**单 chirp buffer 用 `.loops` 选项无缝循环播放**。AVAudioEngine 在 buffer 边界几乎无 gap（`< 1 sample`），从而在麦克风端形成连续的 chirp 序列，所有 chirp 起始相位严格对齐。

如果用 `scheduleBuffer` 多次调用，每次都有调度延迟，chirp 间隔不稳定，后续 mix 出来的拍频抖动严重。

### 2.4 JSON sidecar (`FMCWEngine.swift:143-155`)

每条录音附一个同名 `.json`，包含 4 个 chirp 参数。Python 端 `load_params()` 优先读它。这样：
- iOS 端改参数 → 录音自带新参数 → Python 自动适配
- 避免硬编码不一致

---

## 3. FMCW 测距原理（数学推导）

### 3.1 为什么是 FMCW（不是 CW、不是脉冲）

| 方案 | 范围 | 优缺点 |
|---|---|---|
| CW（单频） | 测速度（多普勒），不能直接测距 | 简单，但拿不到 R |
| 脉冲 | 测距（飞行时间） | 需要高时间分辨率（µs 级），手机 mic 做不到 |
| **FMCW** | 同时测距 + 测速 | 把测距转成 FFT，便宜；TX 连续工作，无峰值功率压力 |

FMCW 的核心 trick：**de-chirping**——把"测短时延"变成"测低频"。

### 3.2 拍频与距离的关系

发射信号瞬时频率：

$$
f_{TX}(t) = f_0 + \mu t
$$

目标距离 R 处反射，单程时延 `R/c`，往返时延 `τ = 2R/c`。
回波瞬时频率：

$$
f_{RX}(t) = f_0 + \mu (t - \tau)
$$

混频（接收信号 × 发射模板）后保留低频项，得到**拍频**：

$$
f_{\text{beat}} = f_{TX} - f_{RX} = \mu \tau = \frac{2BR}{cT}
$$

**拍频 ↔ 距离一一映射：**

$$
R = f_{\text{beat}} \cdot \frac{cT}{2B}
$$

代码（`analyze_fmcw.py:124`）：
```python
range_axis = freq_axis * c * T / (2 * B)
```

### 3.3 距离分辨率

FFT 的频率分辨率 = `1/T`（chirp 持续时间的倒数）。代入上式：

$$
\Delta r = \frac{1}{T} \cdot \frac{cT}{2B} = \frac{c}{2B}
$$

**关键：分辨率只和带宽 B 有关，和 fs、T、NFFT 都无关。**

代入数：`Δr = 343 / 8000 = 4.29 cm`。

注意：`analyze_fmcw.py` 用 `NFFT = N×4 = 3840` 做 FFT 零填充，bin 间隔变成 ~10.7 mm，但**这是插值不是新信息**——分辨率仍是 4.3 cm。零填充只是让曲线更平滑、找峰更精确。

### 3.4 最大可测距离

混频后低通截止 5 kHz（`analyze_fmcw.py:113`），对应：

$$
R_{\max} = 5000 \cdot \frac{cT}{2B} = 5000 \cdot \frac{343 \cdot 0.02}{8000} \approx 4.29\,\text{m}
$$

实际分析里 `max_range_idx = searchsorted(range_axis, 3.0)` 取 3 m 上限——室内人体感知够用，再远的回波也大概率不是目标。

---

## 4. DSP 流水线分步详解

下面每一步对应 `analyze_fmcw.py` 里的具体代码段。

### Step 1 — 带通预处理 (`:88`)

```python
rx_filtered = bandpass(data, f0 - 500, f1 + 500, fs)
```

带通 [17.5, 22.5] kHz，保留 chirp 频段 + 500 Hz 余量（多普勒移频不会超过这个量级）。
去除：低频环境噪声、电源工频、人声、突发瞬态等带外干扰。

### Step 2 — Cross-correlation 对齐 (`:96-99`)

```python
search_len = min(3 * N, len(rx_filtered))
corr       = np.abs(np.correlate(rx_filtered[:search_len], tx_chirp, mode='valid'))
offset     = int(np.argmax(corr))
rx_aligned = rx_filtered[offset:]
```

**为什么需要这一步？** AVAudioEngine 在开始播放与开始录制之间存在不确定的 startup latency（实测 ~1900–2000 samples）。如果直接按 N 切片，每个 chirp 的相位起点不对齐——
- 起点错 1 个 sample = 频率错 50 Hz（fs/N · 1/N · ... 实际是 fs/N = 50 Hz 的 bin）
- 错 100 sample = 错 ~5 kHz 的拍频估计 → range 错离谱

互相关用一个完美 chirp 模板与录音的前 3·N samples 求最大相关位置，找到首 chirp 的真实起点，重切。

### Step 3 — 切片 + 丢弃启动瞬态 (`:100-106`)

```python
rx_data = rx_aligned[:num_chirps * N].reshape(num_chirps, N)
drop    = int(1.0 / T)        # 1 秒 = 50 chirps
rx_data = rx_data[drop:]
```

丢前 1 秒：speaker 有冷启动响应（频响在前 ~500 ms 不稳定），mic AGC 即使关了也有热漂移。

### Step 4 — De-chirping（混频 + 低通） (`:111-113`)

```python
mixed    = rx_data * tx_data
mixed_lp = lowpass(mixed, 5000, fs)
```

数学上：

$$
s_{\text{mix}}(t) = \sin(\omega_{TX} t)\sin(\omega_{RX} t) = \tfrac{1}{2}[\cos((\omega_{TX}-\omega_{RX})t) - \cos((\omega_{TX}+\omega_{RX})t)]
$$

低通 5 kHz 把和频 (~40 kHz) 项扔掉，只剩拍频 (~几百 Hz 到几 kHz)。

### Step 5 — Hanning 窗 + FFT (`:120-125`)

```python
hann_win     = np.hanning(N)
complex_ffts = np.fft.rfft(mixed_lp * hann_win, n=NFFT, axis=1)   # NFFT = 4N
range_ffts   = np.abs(complex_ffts)
```

**为什么 Hanning 窗？** 矩形窗（不加窗）的旁瓣 −13 dB。近场强反射（管口直接耦合）的旁瓣会泄漏到目标 bin，污染 SNR。Hanning 旁瓣 −31 dB，对相邻 bin 的污染降低 ~98%。

**为什么 NFFT = 4N（零填充）？** 让 bin 更密（10.7 mm vs 42.9 mm），找峰位置更精确。代价：FFT 运算量 4×，但 N=960 量级很小，无所谓。

`complex_ffts` 同时**保留实部虚部**——后面相位通道要用。`range_ffts` 是振幅给动作通道用。

### Step 6 — Frame-diff（MTI 背景抵消） (`:132`)

```python
diff_bg = np.abs(np.diff(range_ffts[:, :max_range_idx], axis=0))
```

**雷达术语叫 MTI (Moving Target Indication)。** 静态反射体（墙、桌、管壁本身）在相邻两 chirp 上振幅完全不变，相减归零。运动反射体留下残差。

简单粗暴但有效——比"减去时间均值"更稳健（后者假设场景平稳，墙振幅不变；MTI 只假设墙在 20 ms 内不变，这显然成立）。

### Step 7 — Target bin 选择 (`:135-154`)

```python
MIN_RANGE_M     = 0.25
mean_diff_prof  = diff_bg.mean(axis=0)                     # 时间维平均
target_bin      = int(np.argmax(mean_diff_prof[min_idx:])) + min_idx
```

**关键 insight：用 frame-diff 均值的 argmax，不是振幅均值的 argmax。**

近场管漏（管子本身的直接耦合）振幅极大，但**几乎不变化**（恒定路径）→ frame-diff 接近 0。
真人在某距离上反射会随呼吸、姿态微动 → frame-diff 出峰。

所以 frame-diff 均值天然把人凸显，把杂波（包括强但稳定的杂波）压下去。

`MIN_RANGE_M = 0.25` 是兜底——0.25 m 内可能有管子内部多模谐振产生帧差，从这里开始搜索可避开。

支持手动指定 `--target` 参数（在 ±0.20 m 窗口内搜索），用于已知目标距离的实验。

---

## 5. ★ 呼吸通道完整原理（核心）

这一节是项目的灵魂。读完后你应该能回答：
- 为什么 4.3 cm 的距离分辨率能测出 1 mm 的呼吸？
- 为什么 unwrap 对呼吸有效但对手势无效？
- 三层去趋势每层都在做什么？
- 为什么必须用零相位滤波？

### 5.1 相位作为精密干涉仪

到目标的相位累积：信号往返 2R 距离，每经过一个波长 λ_c 累积 2π 相位：

$$
\varphi(t) = \frac{2\pi \cdot 2R(t)}{\lambda_c} = \frac{4\pi R(t)}{\lambda_c}
$$

如果距离微变 ΔR：

$$
\Delta\varphi = \frac{4\pi \cdot \Delta R}{\lambda_c}
$$

反过来：

$$
\Delta R = \frac{\lambda_c}{4\pi} \cdot \Delta\varphi
$$

**代入数（核心常数）：**

$$
\frac{\lambda_c}{4\pi} = \frac{17.15\,\text{mm}}{4\pi} = 1.365\,\text{mm/rad}
$$

每弧度相位变化对应 1.365 mm 的距离变化。这就是**相位干涉仪的灵敏度系数**。

代码（`analyze_fmcw.py:194`）：
```python
disp_mm = (phase_unwrapped - phase_unwrapped[0]) * c / (4 * np.pi * fc) * 1000
#                                                    ^^^^^^^^^^^^^^^^^^^^^^^
#                                                    = λ_c / (4π) [m/rad]
#                                                    乘 1000 转 mm
```

### 5.2 为什么可以达到亚毫米精度？

**相位精度由 SNR 决定。** 对一个复数 FFT 值 `H = A·e^(jφ)`，加上方差 σ_n² 的复噪声：

$$
\sigma_\varphi \approx \frac{1}{\sqrt{2 \cdot \text{SNR}}} \quad \text{(高 SNR 极限)}
$$

其中 SNR = A²/σ_n²（功率比）。

**数值例子：** target_bin SNR = 30 dB（功率比 1000，电压比 ~31.6）
- σ_φ ≈ 1/√2000 ≈ 0.022 rad
- σ_R ≈ 1.365 × 0.022 = **0.030 mm**

亚毫米精度完全可达！这就是为什么虽然距离分辨率（bin 宽度）是 4.3 cm，但**bin 内的相位读数能给出亚毫米的相对位移**。

**关键概念区分：**
- **距离分辨率** = 区分两个独立目标的最小距离（4.3 cm）→ 由带宽 B 决定
- **相位精度** = 同一个目标在 bin 内位置的微变测量精度（< 1 mm）→ 由 SNR 决定

呼吸不是"两个目标之间的事"，是"一个目标的微动"，所以走相位通道。

### 5.3 工作步骤详解

#### Step A — 取目标 bin 的复数 FFT 序列

```python
phase_raw = np.angle(complex_ffts[:, target_bin])
```

每 chirp 一个相位值，序列长度 = 总 chirp 数。采样率 = 1/T = 50 Hz。

#### Step B — Phase unwrap

```python
phase_unwrapped = np.unwrap(phase_raw)
```

`np.angle` 返回 [−π, π]。当真实相位连续变化越过 ±π，`np.angle` 会"跳"到另一边（差 2π）。`np.unwrap` 检测相邻差大于 π 的跳变，加 ±2π 修正，恢复连续相位。

**为什么 unwrap 对呼吸不出错？**

呼吸位移幅度 ΔR ≈ 5 mm（peak-to-peak），对应**相位幅度**：

$$
\Delta\varphi_{\text{breath}} = \frac{4\pi \cdot 5\,\text{mm}}{17.15\,\text{mm}} \approx 3.66\,\text{rad} \approx 210°
$$

完整呼吸周期 ≈ 4 s = 200 chirps（fs_c=50 Hz）。每 chirp 相位变化平均：

$$
\overline{|\Delta\varphi|}_{\text{per chirp}} = \frac{4 \times 3.66}{200} \approx 0.073\,\text{rad} \approx 4°
$$

远小于 π，unwrap 永远不会跳错。

**对比手势：** 50 cm 位移 / 8.6 mm/wrap ≈ 58 wraps。如果手势持续 2 s = 100 chirps，每 chirp 平均：

$$
\overline{|\Delta\varphi|}_{\text{gesture}} = \frac{2 \times 58 \times 2\pi}{100} \approx 7.3\,\text{rad}
$$

接近 2π，叠加相位噪声极易跳错。这就是为什么**phase 不能做大手势 tracking**。

#### Step C — 三层去趋势（这是关键）

```python
disp_mm        = (phase_unwrapped - phase_unwrapped[0]) * c / (4 * np.pi * fc) * 1000
disp_detrended = signal.detrend(disp_mm)                       # 第 1 层
disp_detrended = highpass_zp(disp_detrended, 0.05, 1/T)        # 第 2 层
resp_signal    = bandpass_zp(disp_detrended, 0.1, 3.0, 1/T)    # 第 3 层
```

人体不可能完全静止，会产生多种漂移：

| 漂移源 | 频率范围 | 处理层 |
|---|---|---|
| 呼吸期间慢慢前倾/后仰 | DC（线性） | detrend |
| 体重转移、姿态调整 | 0.01–0.05 Hz | HP 0.05 Hz |
| 真实呼吸 | 0.1–1 Hz | BP 0.1–3 Hz 通过 |

**第 1 层 — `signal.detrend`（线性去趋势）：**
拟合一条直线（最小二乘）减去。处理"匀速漂移"——比如人在录音过程中匀速倒退。
代码就是 `disp - (a·t + b)`。

**第 2 层 — HP 0.05 Hz：**
线性 detrend 之后还残留非线性慢漂（速度变化的姿态调整）。
0.05 Hz = 周期 20 s，比呼吸周期（4 s）长 5×，安全分离。

**第 3 层 — BP 0.1–3 Hz：**
- 下限 0.1 Hz = 6 bpm（生理最低呼吸率，非常深的睡眠）
- 上限 3 Hz = 180 bpm（高于任何真实呼吸率，留余量）

为什么上限 3 Hz 不是 1 Hz？呼吸**波形不是纯正弦**——吸入快、呼出慢（不对称），含 2x、3x 谐波。截止 1 Hz 会削平这些谐波，波形变形。3 Hz 留出谐波空间，同时把心跳（~1.2 Hz 基频，但能量主要在 5–30 Hz 谱内）和高频噪声切掉。

#### Step D — 为什么必须 zero-phase（`sosfiltfilt`）

普通 IIR 滤波器（`sosfilt`）在不同频率引入不同的相位延迟（**phase response**）。
对呼吸波形的影响：
- 0.2 Hz 分量延迟 30 ms
- 0.5 Hz 分量延迟 15 ms
- 1.0 Hz 分量延迟 5 ms

合成后的波形会"扭曲"——吸气段被压缩、呼气段被拉伸（或反过来），破坏吸/呼不对称的形状信息。

`sosfiltfilt` 的做法：**正向滤一次 → 反向滤一次**，正反向相位响应抵消，**总相位响应 = 0**。
代价：
- 不能实时（需要全部数据）→ 离线分析没问题
- 等效阶数翻倍（order=2 等效 order=4 的幅度响应）→ 改用 order=2 即可

代码（`analyze_fmcw.py:44-50`）：
```python
def highpass_zp(data, cutoff, fs, order=2):
    sos = signal.butter(order, cutoff, btype='high', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, data)         # 双向，零相位
```

### 5.4 频谱分析与置信度

```python
resp_fft   = np.abs(np.fft.rfft(resp_signal))
resp_freqs = np.fft.rfftfreq(len(resp_signal), d=T)
resp_mask  = (resp_freqs >= 0.1) & (resp_freqs <= 1.0)       # 6-60 bpm
peak_f     = resp_freqs[resp_mask][np.argmax(resp_fft[resp_mask])]
resp_snr   = peak_power / (noise_power + 1e-9)               # peak/median
```

**为什么频谱分析？** 时域呼吸信号叠加噪声后波形不一定光滑，但频域上呼吸是窄带（带宽 ~0.05 Hz），噪声是宽带——频域 SNR 远高于时域。

**FFT 长度与频率分辨率：**
- 录音 25 s = 1250 chirps
- FFT 频率分辨率 1/25 = 0.04 Hz = 2.4 bpm
- 足以区分 12 / 14.4 / 16.8 / 19.2 bpm 这种相邻档位

**置信度（peak/median）：**
- 用 median 不用 mean → 对异常峰（心跳谐波等）稳健
- median 反映"典型噪声水平"，peak/median 才是真正的"信号有多突出"
- 阈值 3× → 才输出呼吸率，否则报 "no reliable detection"

代码（`analyze_fmcw.py:212-220`）。

### 5.5 完整数值例子（贯穿）

假设场景：人在 0.5 m 处静坐，呼吸 15 bpm，胸壁位移 ±5 mm。

| 阶段 | 量 | 值 |
|---|---|---|
| 呼吸频率 | f_breath | 0.25 Hz |
| 位移幅度 | ΔR | ±5 mm（peak） |
| 相位幅度 | Δφ = 4π·ΔR/λ_c | ±3.66 rad |
| 单 chirp 间相位变化 | Δφ/100 (一个 quarter cycle) | ±0.037 rad/chirp |
| Bin index | R/Δr_bin | ~117 (NFFT bin) |
| 频谱呼吸峰位置 | argmax | 0.25 Hz |
| 报告呼吸率 | × 60 | **15 bpm** ✓ |

恢复出来的 disp 波形：
$$
\text{disp\_mm}(t) \approx 5 \cdot \sin(2\pi \cdot 0.25 \cdot t) \quad \text{[mm]}
$$

完美对应物理。

### 5.6 失效模式

| 情况 | 现象 | 缓解 |
|---|---|---|
| 目标大幅运动（>4 cm/chirp） | unwrap 跳错，disp 出"阶梯" | motion channel 标 INVALID，跳过这段 |
| target_bin 选错 | 相位是杂波相位，无周期性 | 提供 `--target` 手动指定 |
| SNR 太低 | 相位噪声主导，谱无窄峰 | confidence < 3× → 拒绝输出 |
| 走动 / 移动手机 | 大幅 DC 漂移 | 三层去趋势可处理慢漂；快漂走 motion gate |

---

## 6. 动作通道

呼吸是主，动作是辅——这里只讲核心，详细参见 `notes/2026-04-26_28_session.md`。

### 6.1 处理流程

```python
PEAK_WIN  = 80                                              # ±80 zero-padded bins ≈ ±86 cm
win_slice = diff_bg[:, win_lo:win_hi]
win_max   = win_slice.max(axis=1)
local_peak = np.argmax(win_slice, axis=1)

noise_floor   = np.percentile(win_max, 30)
MOTION_THRESH = 3.0 * noise_floor                          # 默认 3×

last_valid = float(target_bin)
for i in range(len(win_max)):
    if win_max[i] > MOTION_THRESH:
        last_valid = float(local_peak[i] + win_lo)         # 有动 → 更新
    global_peak[i] = last_valid                            # 无动 → 保持
```

### 6.2 在呼吸项目中的 3 个作用

1. **有效性闸门：** `win_max > MOTION_THRESH` 的帧标 active。这些段呼吸读数标 N/A 或低置信度（避免 unwrap 误差污染）。
2. **管子方向性物理证据：** 横向挥手 0.8% 激活 vs 前倾 9.3%，最强证明"管子做径向选择"的实验数据。
3. **保留 proposal scope：** 原 proposal 含 HAR；保留 motion channel = 不缩 scope。

输出：`tracked_dist_m`（绝对距离，4.3 cm 分辨率，无 wrap 限制）。

---

## 7. 双通道架构

### 7.1 为什么必须分两路

| | Motion channel | Breathing channel |
|---|---|---|
| 数据源 | `\|FFT\|` + frame-diff | `∠FFT` + phase unwrap |
| 分辨率 | 4.3 cm（由 B 决定） | <1 mm（由 SNR 决定） |
| 量程 | 任意大位移 | 单 chirp <λ/2 ≈ 8.6 mm |
| 适合 | 手势、走动、起立 | 呼吸、心跳、微振 |
| 失效 | < 4 cm 看不见 | > λ/2 unwrap 失败 |

**单通道方案做不到呼吸 + 动作覆盖：**
- 只用 |FFT|（动作）：4.3 cm 分辨率根本看不见 5 mm 呼吸位移
- 只用 ∠FFT（呼吸）：50 cm 手势 unwrap 就崩

### 7.2 两路共享前端

```
Bandpass → Cross-corr → Mix → LP → Hanning FFT
                                    │
                                    ▼
                              complex_ffts        ← 同一 target_bin
                              ├── |·| ──► Motion channel (frame-diff)
                              └── ∠·  ──► Breathing channel (phase)
```

加 motion channel 的代码成本 < 50 行，FFT 共享，几乎免费。**砍掉它没收益，留着获得有效性 gate + 物理证据。**

---

## 8. 实现地图

| 概念 | 文件 | 行号 |
|---|---|---|
| Chirp 生成（iOS） | `iOS App/FMCWEngine.swift` | 43–56 |
| 录制配置 | `iOS App/FMCWEngine.swift` | 31–39 |
| Buffer loop 播放 | `iOS App/FMCWEngine.swift` | 111 |
| JSON sidecar | `iOS App/FMCWEngine.swift` | 143–155 |
| Chirp 模板（Python） | `analyze_fmcw.py` | 55–63 |
| 带通预处理 | `analyze_fmcw.py` | 88 |
| Cross-corr 对齐 | `analyze_fmcw.py` | 96–99 |
| Mix + LP | `analyze_fmcw.py` | 111–113 |
| Hanning + FFT | `analyze_fmcw.py` | 119–125 |
| Frame-diff | `analyze_fmcw.py` | 132 |
| Target bin 选择 | `analyze_fmcw.py` | 135–154 |
| **动作 tracker** | `analyze_fmcw.py` | 156–188 |
| **★ 相位提取** | `analyze_fmcw.py` | 190–194 |
| **★ 三层去趋势** | `analyze_fmcw.py` | 195–197 |
| **★ 频谱 + 置信度** | `analyze_fmcw.py` | 205–220 |
| Zero-phase 滤波 | `analyze_fmcw.py` | 44–50 |
| 呼吸 SNR / 置信度 metric | `compare_conditions.py` | 107–122 |
| CSR / 相位 coherence | `compare_conditions.py` | 111–115 |

---

## 9. Slide 准备指南

下面是把上述内容压成两页 slides 的版本。

### Slide 1 — System Pipeline

```
[iPhone speaker] ──18-22 kHz FMCW──> [3D-printed tube] ──> [target]
[iPhone mic]    <──reflected echo──────────────────────────
        │
        ▼
   WAV file + JSON sidecar (params)
        │
        ▼
Bandpass → Cross-corr align → Reshape (chirps × N)
   ↓
Mix (RX × TX template) → LP 5 kHz → Hanning FFT
   ↓
Range profile (chirps × bins)
   ↓
Frame-diff (MTI) → Target-bin selection
   ↓
   ├──── |FFT|  ──► Motion channel (validity gate)
   └──── ∠FFT   ──► Breathing channel (primary)
```

参数表：
```
fc = 20 kHz   B = 4 kHz   T = 20 ms   fs = 48 kHz
Δr = c/(2B) = 4.3 cm     λc/2 = 8.6 mm
```

要点：
- iOS 只采集，DSP 离线 Python（迭代速度 + 实验可重现）
- Cross-corr 补偿 buffer jitter (~2000 samples)
- Frame-diff = MTI（雷达术语）

### Slide 2 — Breathing Extraction with Motion-Aware Gating

核心论点：
> Amplitude resolves WHERE; phase resolves how MUCH.
> Motion channel gates breathing channel's validity.

双通道分工表（见 §7.1）。

支撑技术：
- **Phase as interferometer**：λ_c/(4π) = 1.365 mm/rad
- **Three-stage detrend**：detrend → HP 0.05 → BP 0.1–3 Hz
- **Zero-phase filtering** (`sosfiltfilt`)：保留呼吸波形形状
- **Frame-diff (MTI)**：静态杂波相邻帧相减归零

推荐配图：
- 主图：长管录音 (`fmcw_1777232317`) 的呼吸波形 ~6 bpm
- 配图：上方 motion activity bar，红色段标 INVALID
- 角图：wave 实验 motion trace（几乎全平，证径向选择）

### 取舍参考

| 必须保留 | 可以略过 |
|---|---|
| Motion channel 追踪曲线（wave 证据） | AVAudioSession 配置细节 |
| compare_conditions 的 motion 指标 | GUI / 批处理 |
| Cross-corr alignment | 全部 metric 名字（压成一句） |
| Zero-phase 滤波 | Phase integration overlay 红线 |

### 演讲节奏

- Slide 1（60–90 s）：强调"为什么 DSP 离线 Python"
- Slide 2（60–90 s）：先发制人讲"为什么不能只用一个通道"
- 留 30 s 展示真实输出（呼吸波形 + 管长对比图）

### 一句话总结

> **This is a respiration sensor.**
> Phase channel gives sub-mm breathing waveform via interferometric phase tracking.
> Motion channel gates its validity AND validates the tube's directional claim.
> Both share one FMCW front-end — same target, two scales.
