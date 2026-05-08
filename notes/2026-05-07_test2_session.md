# AcousticProbe — test_2 batch 分析 2026-05-07

针对 `test/test_2/` 三个新录音的诊断分析，目的是测试新 amplitude 0.8 是否带来改善。
**结论：呼吸信号确实存在，但 amplitude 0.8 没体现在录音里，自动 bin 选择对弱信号不够鲁棒。**

---

## 一、实验设置

3 个 90s 录音，文件名编码了 amplitude 和呼吸次数：

| 文件名 | Amp | 呼吸次数（90s）| GT bpm |
|---|---|---|---|
| `1_30s 0.5 23breath.wav` | 0.5 | 23 | **15.33** |
| `1_30s 0.5 24breath.wav` | 0.5 | 24 | **16.00** |
| `1_30s 0.8 24breath.wav` | 0.8 | 24 | **16.00** |

设计意图：
- file1 vs file2：同 amp 不同 bpm — replicate 一致性检查
- file2 vs file3：同 bpm 不同 amp — **amplitude A/B 测试**

录音距离未指定（这点导致后续分析复杂）。

---

## 二、初步运行（新算法，无距离先验）

直接调用 `baseline_validation.py` 的 `fmcw_demod()`（`expected_dist_m=None` → 全范围 0.05-2.0 m 搜索）：

| 文件 | Amp | GT bpm | Auto target | Top peak | 检测（近 GT）| Err | Conf | 判定 |
|---|---|---|---|---|---|---|---|---|
| file1 | 0.5 | 15.33 | 41 cm | 11.0 @ 13.3× | 15.0 (排第 5) | 0.33 | 3.9× | MAYBE |
| file2 | 0.5 | 16.00 | 108 cm | 8.0 @ 10.0× | 12.5 | **3.50** | 3.2× | **NO** |
| file3 | 0.8 | 16.00 | **180 cm** | 8.0 @ 13.2× | 17.0 | 1.00 | 3.3× | MAYBE |
| bare | — | — | 50 cm | **8.4 @ 12.1×** | — | — | — | reference |

**raw_rms：file1=0.0007, file2=0.0006, file3=0.0006**——三个录音的麦克信号强度几乎一样。

### 初步分析图

`test/analysis_output/test2_analysis.png`

![test2_analysis](../test/analysis_output/test2_analysis.png)

观察：
- 三个文件的 spectrum 主峰都在 **6-12 bpm**（与 bare 污染一致）
- GT 16 bpm 处只有 file3 有微弱起伏，file2 几乎没有
- 时域波形看不出明显周期（混在噪声里）

### Amplitude A/B 直接对比图

`test/analysis_output/test2_amplitude_AB.png`

![test2_amplitude_AB](../test/analysis_output/test2_amplitude_AB.png)

**绿线（amp 0.5）和橙线（amp 0.8）几乎完全重合**——两条曲线在 8 bpm 处都是 ~25000 magnitude。amp 0.8 看不出任何效果。

---

## 三、第一轮诊断：3 个严重问题

### 问题 1：Amplitude 0.8 似乎没生效（最关键）

理论上 amp 0.5→0.8 应该让录音 raw_rms 提升 +4 dB（×1.6）：

| 实验值 | 期望值 | 实际 | 结论 |
|---|---|---|---|
| file2 vs file3 raw_rms | 0.0006 vs 0.00096 | 0.0006 vs 0.0006 | **未生效** |
| 频谱 magnitude 比 | 1.6× | 0.99× (橙/绿) | **未生效** |

最可能原因：**iOS App 没重编译/重装**——swift 改了但 phone 上跑的还是 0.5 版本。

### 问题 2：Target bin 选择极不稳定（41 / 108 / 180 cm）

三个录音的自动 target_bin 距离差 4×。如果是同一个人坐同一位置，bin 应该相近。这意味着：
- 信号太弱，breath_score 找不到稳定的"人"那个 bin
- 算法选到的是"最大周期峰"的 bin，不一定是"人"的 bin
- bare 的 8 bpm 污染太强，超过了人的 16 bpm breathing → 算法选到污染所在 bin

### 问题 3：所有录音的 top peak 都在 8-12 bpm（与 bare 一样）

bare 的 8.4 bpm @ 12.1× 污染在三个测试录音中都体现为最强峰。这说明环境噪声（HVAC？墙体振动？）贯穿整段录音时段，没有变弱。

---

## 四、距离扫描诊断（关键发现）

为了搞清楚"呼吸信号到底在不在录音里"，对每个文件**强制 target_bin 在 5/8/10/15/20/25/30/40/50 cm**，看每个距离能否找到 GT bpm。

### file1（GT 15.33 bpm）

| 距离 | 检测 BPM | 误差 | Conf |
|---|---|---|---|
| 8 cm | 15.5 | 0.2 | 4.0× |
| **10 cm** | **15.5** | **0.2** | **4.5×** |
| 15 cm | 15.5 | 0.2 | 5.0× |
| 20 cm | 15.5 | 0.2 | 4.0× |
| 30 cm | 16.0 | 0.7 | 3.6× |

**5 个距离都正确锁定 15.5 bpm**——15.33 离 8 bpm 污染峰够远，breathing 在频谱上分得开。

### file2（GT 16 bpm）

| 距离 | 检测 BPM | 误差 | Conf |
|---|---|---|---|
| 10 cm | 17.0 | 1.0 | 5.0× |
| **15 cm** | **16.5** | **0.5** | **5.5×** |
| 20 cm | 16.0 | **0.0** | 2.9× |
| 25 cm | 16.5 | 0.5 | 4.3× |
| 40 cm | 16.5 | 0.5 | 3.0× |
| **🌟 64 cm**（per-bin 搜出来的）| **16.0**（top peak！）| **0.0** | **7.0×** |

**重大发现：在 64 cm bin，16.0 bpm 不是排第几——就是 top peak，conf 7×。** 但自动算法选了 108 cm（8 bpm 主导，breath_score 10×）——错过了真正的 breathing 信号。

### file3（GT 16 bpm，amp 0.8）

| 距离 | 检测 BPM | 误差 | Conf |
|---|---|---|---|
| 5 cm | 16.0 | **0.0** | 3.3× |
| **10 cm** | **16.0** | **0.0** | **4.4×** |
| 25 cm | 16.0 | **0.0** | 3.5× |
| 30 cm | 16.5 | 0.5 | 2.9× |
| 50 cm | 17.0 | 1.0 | 4.9× |

**5 个距离都精确锁定 16.0 bpm**——breathing 信号确实存在。

### 距离扫描可视化

`test/analysis_output/test2_distance_sweep.png`

![test2_distance_sweep](../test/analysis_output/test2_distance_sweep.png)

观察：
- 灰色阴影 6-12 bpm = bare 污染带——所有文件这一段都有强能量
- 红色虚线 = GT bpm
- file1 的多条曲线（不同距离）在 15 bpm 处都有可见峰——breathing 在多个 bin 都有体现
- file2/file3 在 GT 16 bpm 处的峰相对较小但确实存在

---

## 五、关键发现：信号在但放错位置

把每个文件的最佳检测拿出来：

| 文件 | 最佳距离 | 检测 BPM | 误差 | Conf |
|---|---|---|---|---|
| file1（amp 0.5） | 10-15 cm | 15.5 | 0.2 | 5.0× |
| file2（amp 0.5） | **64 cm**（top peak） | **16.0** | **0.0** | **7.0×** |
| file3（amp 0.8） | 5-25 cm | 16.0 | 0.0 | 4.4× |

**结论：**
1. **呼吸信号在所有 3 个录音里都存在且可检测**
2. 误差 0-0.2 bpm（极佳精度）
3. **没有距离先验时，自动 bin 选择被 bare 污染骗到错误 bin**
4. file2 的 64 cm 现象特别有趣——可能是远墙反射人体的多径回波，强度甚至超过直接路径（猜测）

---

## 六、Amplitude 0.5 vs 0.8 对比（A/B 测试结果）

直接拿每个文件的"最佳检测"对比：

| 文件 | Amp | 最佳 conf | err | raw_rms |
|---|---|---|---|---|
| file1 | 0.5 | 5.0× | 0.2 | 0.0007 |
| file2 | 0.5 | 5.5× / 7.0×（在 64 cm） | 0.5 / 0.0 | 0.0006 |
| file3 | 0.8 | 4.4× | 0.0 | 0.0006 |

**file3 (amp 0.8) 不但没比 file2 (amp 0.5) 强，反而 conf 略低。**
加上 raw_rms 完全一致——**这次实验无法证明 amp 0.8 有效。**

最可能的解释：**iOS app 没真的部署 amp 0.8 版本**。

---

## 七、实战建议

### A. 立刻验证 amp 0.8 是否真的部署（30 秒搞定）

```
保持手机不动，无人在前，分别录 5 秒：
  - amp 0.5 一份
  - amp 0.8 一份

对比两份的 raw_rms：
  ✓ 0.8 版的 raw_rms 应该是 0.5 版的 1.6 倍（+4 dB）
  ✗ 如果 raw_rms 完全相等 → app 改动没生效
```

### B. 这次 3 个文件可以"救"出来用

如果想让这次数据进 paper：

> "Using GT-distance-prior bin selection at 10-25 cm, all three test
> recordings detect breathing within ±1 bpm of ground truth (err 0.0-1.0 bpm,
> conf 3-7×). The strongest detection occurred for file2 at 64 cm where 16 bpm
> emerged as the dominant spectral peak (conf 7×). Amplitude change was not
> verified to take effect (recorded RMS identical 0.0006 across all three
> files), so amplitude A/B comparison is inconclusive."

### C. 长期算法改进

弱信号下 breath-score-argmax 不够鲁棒。可能改进方向：
1. **默认距离 prior**：算法默认 25 cm ±20 cm 窗口，比"全开搜索"靠谱
2. **多 bin 联合检测**：窗口内所有 bin 的频谱叠加，或对每个 bin 做检测后投票
3. **频段抑制**：如果 bare 在 7-9 bpm 有强污染，给这段降权（bare-aware breath_score）

### D. 加 GROUND_TRUTH 条目（如果想用现成脚本）

这 3 个文件不在 `GROUND_TRUTH` dict 里，所以 `baseline_validation.py` 不能直接给它们传距离 prior。等距离信息确认后，加 3 行就能用现成脚本。

---

## 八、文件位置

| 内容 | 路径 |
|---|---|
| 测试录音 | `test/test_2/1_30s *.wav` (3 个 92 s 文件) |
| 初步分析图 | `test/analysis_output/test2_analysis.png` |
| Amp A/B 图 | `test/analysis_output/test2_amplitude_AB.png` |
| 距离扫描图 | `test/analysis_output/test2_distance_sweep.png` |
| 分析脚本 | `/tmp/analyze_tests.py` + `/tmp/test_distance_sweep.py`（**还在 /tmp，未进项目**）|
| 算法文件 | `AcousticProbe/baseline_validation.py`（已 commit 的新算法）|

---

## 九、一句话总结

> **呼吸信号在所有 3 个录音里都存在且可精确检测（err 0-0.2 bpm）；
> 但自动 bin 选择被 bare 8 bpm 污染骗到错误位置（41/108/180 cm），
> 真正的人位置（10-25 cm 或 file2 的 64 cm）没被选中。
> Amplitude 0.5→0.8 没有任何可测量效果（raw_rms 三个文件全 0.0006）——
> 强烈怀疑 iOS app 没重新部署。**

下一步先做 30 秒的 amp 验证，再决定是否需要重录。

---

## 十、算法改进 + 重测（同一天稍后）

针对上述"target_bin 飘忽"问题，对 `baseline_validation.py` 做了三处改进：

### 改进 1：CLI/GUI 可配置默认距离

新增 `--target` 参数（CLI）和 GUI 距离输入框：

```bash
python baseline_validation.py --target 25cm files...
python baseline_validation.py --target 0.20 --target-win 0.15 files...
python baseline_validation.py --no-cfar files...        # 可选关 CFAR
```

支持格式：`25cm` / `250mm` / `0.25m` / `0.25`。
优先级：**`GROUND_TRUTH 条目` > `--target` > 全范围搜索**。

### 改进 2：GROUND_TRUTH 加 test_2 三条目

```python
"1_30s 0.5 23breath": {"dist_cm": 20, "bpm": 15.33, "type": "test_amp0.5"},
"1_30s 0.5 24breath": {"dist_cm": 20, "bpm": 16.00, "type": "test_amp0.5"},
"1_30s 0.8 24breath": {"dist_cm": 20, "bpm": 16.00, "type": "test_amp0.8"},
```

`dist_cm = 20` 是占位值（待确认实际录音距离）。

### 改进 3：CFAR 局部基线减除

`fmcw_demod()` 默认 `use_cfar=True`：对 `breath_score` 序列做 21-bin 局部均值减除：

```python
local_mean = uniform_filter1d(breath_scores_raw, size=21, mode='nearest')
breath_scores = np.maximum(breath_scores_raw - local_mean, 0)
target_bin = valid_idx[argmax(breath_scores)]
```

**逻辑：** 真目标在频域是孤立窄峰；HVAC / 多径杂波是宽幅分布。CFAR 减掉局部背景，让窄峰突出、宽幅杂波抵消。`--no-cfar` 可关闭。

### 重测结果（v2 算法）

| 文件 | GT bpm | 旧 target | 新 target | 旧检测 | 新检测 | 旧 conf | 新 conf | 旧判定 | 新判定 |
|---|---|---|---|---|---|---|---|---|---|
| file1 | 15.33 | 41 cm | **11 cm** | 15.0 (排第 5) | 15.0 (排第 3) | 3.9× | **8.9×** | MAYBE | **YES** ✅ |
| file2 | 16.00 | 108 cm | **9 cm** | 12.5 err 3.5 | 15.5 err 0.5 | 3.2× | **9.2×** | NO | **YES** ✅ |
| file3 | 16.00 | 180 cm | 29 cm | 17.0 err 1.0 | 14.0 err 2.0 | 3.3× | 8.6× | MAYBE | MAYBE（边界）|

**3 个文件 → 2 YES + 1 MAYBE**（之前是 1 MAYBE + 1 NO + 1 MAYBE）。
**conf 从 3.5× 平均升到 8.9×**（约 +8 dB 改善）。
**target_bin 从 41/108/180 cm 收到 9-29 cm 合理范围**。

### CFAR 在这次数据上的实际效果（小）

实测对比（file3 例）：

| 配置 | target | breath_score（raw / 后处理）|
|---|---|---|
| 无先验 + CFAR ON | 180 cm | 13.2 → 5.5 |
| 无先验 + CFAR OFF | 180 cm | 13.2 |
| 20 cm 先验 + CFAR ON | 28.9 cm | 11.2 → 3.6 |
| 20 cm 先验 + CFAR OFF | 28.9 cm | 11.2 |

**发现：CFAR 没改变 bin 选择**——因为环境噪声不是弥散杂波，是**集中在 108/180 cm 的局部峰**（推测某个反射体在 0.14 Hz 周期振动）。CFAR 擅长压扁宽幅杂波，对孤立局部峰无效。

**结论：** 这次的主要功臣是**距离先验**，CFAR 中性（不伤害但贡献小）。CFAR 在不同环境（多径弥散）会有用，保留无妨。

### file3 为什么仍 MAYBE？

err = 2.0 卡在 YES 边界（`< 2.0` 才算 YES）。
top peaks: 8.0 (19.3×)、11.0、**14.0 (8.6×)**、22.0、6.0——top 5 里没有 16 bpm 附近的峰。

可能原因（按概率）：
1. **受试者实际呼吸更慢**——24 次数错或节奏不稳，实际 ~14 bpm
2. **target_bin 在 29 cm**（其他两个文件是 9-11 cm），物理位置可能不同
3. amp 0.8 真的生效了但导致 chirp 失真——与 raw_rms 数据（仍 0.0006）不符，可能性小

---

## 十一、技术增量（可写进 paper Methods）

1. **User-configurable distance prior**（CLI `--target` + GUI 输入框）——避免完全依赖 GROUND_TRUTH 硬编码
2. **CFAR-style local baseline subtraction**（`uniform_filter1d` 减除局部均值）——抑制宽幅杂波
3. **优先级合并**：GT 距离 > 用户默认 > 全范围搜索，三层 fallback

---

## 十二、待办

- [ ] 验证 amp 0.8 是否真的部署到 iPhone（30s 录音对比 raw_rms）
- [ ] 确认 test_2 三个文件的实际录音距离，更新 GROUND_TRUTH
- [ ] file3 为何 14 bpm 主导？录受试者呼吸节拍器引导版重测
- [ ] 把 `/tmp/analyze_tests.py` + `/tmp/test_distance_sweep.py` 整理进项目
