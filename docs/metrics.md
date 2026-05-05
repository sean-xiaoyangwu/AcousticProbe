# AcousticProbe — Evaluation Metrics

7 个 metrics 跨 3 层，量化 meta-structure 性能。
全部实现在 `compare_conditions.py:51-152` (`process_one()`)。

---

## Layer 1 — Signal Quality

> "管子有没有改善物理信号？"

### 1. Resp SNR (dB) — `compare_conditions.py:107-109`

```python
rs   = bp_zp(dd, 0.1, 3.0, fs_c)           # 呼吸带通 (0.1-3 Hz)
nr   = dd - rs                              # 带外残余
rsnr = 10 * np.log10(np.var(rs) / (np.var(nr) + 1e-12))
```

**数学：**
$$
\text{RespSNR} = 10\log_{10}\frac{\text{Var}(x_{0.1\text{–}3\,\text{Hz}})}{\text{Var}(x_{\text{rest}})}
$$

**思路：** 把相位位移 `dd` 拆成"呼吸频段"和"其他一切"，比方差就是比能量占比。

**怎么读：**
- > 0 dB：呼吸频段能量大于带外噪声
- > 5 dB：呼吸已经主导信号（可视化波形干净）
- < 0 dB：呼吸淹没在噪声里
- **时域**指标——不要求呼吸有窄峰

### 2. CSR (dB) — Clutter-to-Signal Ratio — `compare_conditions.py:111-114`

```python
tp  = mp[tb]                                # target bin 的振幅
cb  = np.concatenate([mp[mn:max(mn,tb-10)], mp[min(tb+10,mi):mi]])
cp  = np.mean(cb)                           # ±10 bin 邻域以外的杂波平均
csr = 20 * np.log10(tp / (cp + 1e-9))
```

**数学：**
$$
\text{CSR} = 20\log_{10}\frac{|H(\text{target bin})|}{\overline{|H(\text{other bins})|}}
$$

`mp` 是时间维平均的 range profile（每个 bin 的平均回波振幅）。

**思路：** 目标 bin 在 range profile 上"鼓"得有多明显——管子聚焦效果好的话，能量集中到目标 bin，周围 bin 应该是"谷"。

**怎么读：**
- 与呼吸无关的**纯结构指标**——只看 range profile 形状
- 高 CSR 可能配低 Resp SNR（目标清晰但呼吸太弱）
- 高 CSR 是管子方向性的直接证据
- 用 `20·log10` 因为是振幅比（不是能量比）

### 3. Phase Coherence (0–1) — `compare_conditions.py:115`

```python
coh = np.abs(np.mean(np.exp(1j * np.diff(pr))))
```

**数学：**
$$
\text{coh} = \left|\frac{1}{N-1}\sum_{i=1}^{N-1} e^{j(\varphi_{i+1}-\varphi_i)}\right|
$$

`pr = np.angle(complex_ffts[:, target_bin])` 是每 chirp 的原始（未 unwrap）相位。

**思路：** circular statistics 的 "mean resultant length"。
- 把每对相邻相位差 Δφ 写成单位复数 e^(jΔφ)
- 完全随机的 Δφ → 单位复数四面八方 → 平均后趋于 0
- 完全一致的 Δφ → 平均后模长接近 1

**怎么读：**
- 1.0：相位完美锁定（目标静止 + 无相位噪声）
- 0.95–0.99：正常呼吸（相位有小幅缓慢变化）
- < 0.9：目标明显运动或没锁住目标
- 比"相位标准差"更稳健，因为相位是 wrap 的（标准差对 ±π 跳变敏感）

---

## Layer 2 — Detection Reliability

> "能不能可靠地检出呼吸？"

### 4. Breath Confidence (×) — `compare_conditions.py:117-122`

```python
rf  = np.abs(np.fft.rfft(rs))              # 呼吸信号的振幅谱
rfq = np.fft.rfftfreq(len(rs), d=T)
rmk = (rfq >= 0.1) & (rfq <= 1.0)           # 6-60 bpm 范围
bc  = np.max(rf[rmk]) / (np.median(rf[rmk]) + 1e-9)
```

**数学：**
$$
\text{conf} = \frac{\max_{f \in [0.1, 1.0]} |R(f)|}{\text{median}_{f \in [0.1, 1.0]} |R(f)|}
$$

**思路：** 谱峰高度 vs 谱中位数——如果呼吸有清晰主频，主峰应该远高于中位数。

**怎么读：**
- > 5×：高度可信，单峰极突出
- 2–5×：可识别但谱有噪声
- < 2×：拒绝输出（在 `rr = ...` 行用作 gate）
- 用 median 不用 mean，对异常值稳健
- **频域**互补于 Resp SNR（SNR 看能量占比，confidence 看是否单峰）

### 5. Breathing Rate (bpm) — same block

```python
rr = 0.0
if rmk.any() and bc > 2:
    rr = rfq[rmk][np.argmax(rf[rmk])] * 60
```

**数学：** 谱在 0.1–1 Hz 内的 argmax × 60，仅当 `confidence > 2` 时报告。

**怎么读：**
- 不是性能指标，是**观测结果**
- 12–20 bpm 是健康成人静息值
- 配合 confidence 解读：confidence 高 + bpm 合理 = 真实呼吸；confidence 低 → bpm 不可信，宁可不报

---

## Layer 3 — Signal Strength

> "改善了多少？"

### 6. Displacement Amplitude (mm) — `compare_conditions.py:123`

```python
da = np.percentile(rs, 97.5) - np.percentile(rs, 2.5)
```

**数学：** 呼吸波形的 95% peak-to-peak 范围。

**思路：** 不用 `max - min`（被异常值主导），用 P97.5 − P2.5 robust 估计真实呼吸幅度。

**怎么读：**
- 单位 mm（因为 `disp_mm` 已经是毫米）
- 健康呼吸的胸壁位移大约 5–10 mm（正面）
- 数值受耦合方式、距离、角度影响——**单看绝对值意义不大，看相对 baseline 的倍数**
- 异常大的值（如 50+ mm）建议配合 Resp SNR 判断是否为运动/相位放大伪影

### 7. Tracking Jitter (cm) — `compare_conditions.py:124`

```python
jt = np.std(pd) * 100
```

**数学：** argmax 追踪曲线的标准差（米→厘米单位转换）。

`pd = range_axis[sm.astype(int)]` 是平滑后的距离追踪输出。

**思路：** 静态目标场景下追踪曲线应该是一条直线——抖动越大说明 argmax 越不稳。

**怎么读：**
- **唯一一个"低 = 好"的指标**（与其他 6 个相反，做 radar 图时要 invert）
- < 1 cm：管子稳稳锁住
- 5–10 cm：argmax 在多个 bin 之间漂移
- > 20 cm：基本没锁住目标
- 是 motion channel 的稳定性，间接反映 phase 通道质量（target_bin 漂移 → 相位读错 bin）

---

## 指标横向关系

| 指标对 | 关系 |
|---|---|
| Resp SNR ↔ Breath Confidence | 互补（时域 vs 频域）。两个都高才是真信号 |
| CSR ↔ Resp SNR | 解耦。CSR 看结构，SNR 看呼吸内容 |
| Phase Coherence ↔ Tracking Jitter | 同向但不同尺度。Coherence 看亚毫米相位稳定性，Jitter 看厘米级 bin 稳定性 |
| Displacement Amp ↔ Resp SNR | **必须配对看**。大幅度 + 高 SNR = 真信号；大幅度 + 低 SNR = 噪声/运动伪影 |
| Breath Confidence ↔ Resp Rate | 后者是前者的副产物，confidence < 2 时 rate 直接被 gate 掉 |

---

## 分析流程（compare_conditions.py 怎么用这 7 个数）

### Step A：每个录音 → 7 个数
`process_one()` 跑一遍每条 WAV，返回 dict。

### Step B：每个 metric 找最优条件
```python
best_idx = np.argmax(all_vals) if higher_is_better else np.argmin(all_vals)
```
（`compare_conditions.py:201-203`，注意 jitter 是 `higher_is_better=False`）

### Step C：与 baseline 的相对改善
- 加性指标（dB）：`Δ = condition[i] − condition[0]`
- 乘性指标（×、mm）：`ratio = condition[i] / condition[0]`

### Step D："best overall" 的投票
```python
wins = [0]*nc
for key, hb in ALL_WIN:                     # 7 个核心指标
    vals = [r[key] for r in results]
    wins[np.argmax(vals) if hb else np.argmin(vals)] += 1
ob = np.argmax(wins)                        # 赢得最多 metric 的 condition
```
（`compare_conditions.py:303-308`）

### Step E：可视化
- **Radar chart**：7 个指标归一化到 [0.25, 0.95]，每个 condition 一条多边形
  - 归一化是 per-metric min-max（`compare_conditions.py:163-166`），所以**不是绝对水平**——只能读相对排序
- **Bar chart**：原始数值横向条带 + `*best` 标签
- **Table**：所有数 + Δ vs baseline 行

---

## 优先级建议（如果要在 paper / slide 上挑核心 3 个）

1. **CSR** — 管子方向性的直接证据，与呼吸无关，最干净
2. **Resp SNR** — 呼吸信号清晰度，时域，鲁棒
3. **Breath Confidence** — 频域单峰显著性，可信度的最终判据

其余 4 个放进 supplementary：
- Phase coherence — 高级稳定性指标（学术加分）
- Displacement amp — 单看意义弱，必须配 Resp SNR
- Tracking jitter — motion channel 副产物
- Breathing rate — 观测值不是性能指标
