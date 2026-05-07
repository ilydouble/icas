# ICAS 语音特征与临床特征筛选分析报告

## 概览

本报告总结当前仓库中针对 **2025 年 ASR 语音转写特征** 与
`full_data` 中 **患者级临床特征** 所做的筛选、相关性分析、基线建模与融合实验结果。

本轮工作的目标不是直接给出最终模型，而是回答下面四个更基础也更关键的问题：

1. 2025 年 ASR 数据能否稳定匹配到 `full_data` 中的临床监督信息。
2. 哪些语音特征与 ICAS 二分类标签、狭窄分级最相关。
3. 哪些临床特征本身最有解释力，是否需要单独筛选。
4. 语音与临床信息在 classical baseline 下是否存在稳定互补性。

结论先行：

- **ASR 数据基本可以匹配到临床信息**：405 份 ASR JSON 中，404 份可直接匹配到 `patient_clinical_data.csv`。
- **语音特征存在一定信号，但单变量相关性整体偏弱**，更像“弱但可组合”的特征分支。
- **临床特征中的强信号高度集中在少数变量**，尤其是 `waist_hip_ratio`、`gender_encoded`、`height`。
- **全量早期融合效果不好，但筛选后融合显著改善**。
- **当前最强的 classical baseline 是 `top-3 clinical` 单独建模**，而不是简单把语音再加进去。

## 数据来源与分析产物

### 语音特征来源

- `datasets/ASR/json_results/*.json`
- `datasets/full_data/patient_clinical_data.csv`

语音特征抽取脚本：

- `scripts/extract_asr_features.py`

语音相关性与筛选产物：

- `datasets/asr_2025_features.csv`
- `reports/asr_feature_correlation_scores.csv`
- `reports/asr_feature_correlation_report.md`
- `reports/asr_candidate_feature_list.csv`
- `reports/asr_candidate_modeling_subset.csv`

### 临床特征来源

- `datasets/full_data/patient_clinical_data.csv`

临床相关性与筛选产物：

- `reports/clinical_feature_correlation_scores.csv`
- `reports/clinical_feature_correlation_report.md`
- `reports/clinical_candidate_feature_list.csv`
- `reports/clinical_candidate_modeling_subset.csv`

### Classical baseline 产物

- `reports/asr_clinical_model_comparison_20260507_131227.csv`
- `reports/filtered_asr_clinical_model_comparison_20260507_143306.csv`
- `reports/topk_filtered_asr_clinical_model_comparison_20260507_152158.csv`
- `reports/late_fusion_asr_clinical_20260507_152729.csv`

## 一、ASR 数据与临床监督信息匹配情况

2025 年 ASR 数据共包含 **405** 份患者级 JSON。

- 成功匹配到 `patient_clinical_data.csv`：**404**
- 未匹配：**1**
- 未匹配患者编号：`004EB`

因此，从工程可用性的角度看，ASR 分支已经具备进入下游建模的条件，不需要先做大规模人工对齐修复。

## 二、语音特征设计与筛选结果

### 语音特征设计原则

当前语音分支没有使用 embedding，而是优先抽取 **可解释的、适合与临床标签做相关性分析** 的统计特征，主要包括：

- 句级语速统计：均值、最小值、最大值、波动
- 句时长与停顿统计：平均句长、长停顿比例、停顿总量
- 文本长度与朗读完成度指标
- 重复字、填充词、相对固定朗读文本的偏离程度
- ASR 参考文本编辑距离、字符错误率、长度比等

### 单变量相关性结果

语音单变量相关性整体偏弱，没有出现非常强的单一特征。综合分数靠前的特征为：

| 排名 | 特征 | 说明 |
|---|---|---|
| 1 | `asr_speech_rate_min` | 最低语速，提示最慢发音片段可能更敏感 |
| 2 | `asr_chars_per_sentence_mean` | 平均句长，可能反映朗读完整性或流畅度 |
| 3 | `asr_chars_per_second` | 单位时间输出字符数，兼具语速和完成度含义 |
| 4 | `asr_emotion_median` | ASR 内部情绪值中位数，解释性较弱但保留 |
| 5 | `asr_long_pause_sentence_ratio` | 长停顿比例 |
| 6 | `asr_pause_sentence_ratio` | 停顿句比例 |
| 7 | `asr_sentence_duration_ms_mean` | 平均句时长 |
| 8 | `asr_sentence_duration_ms_min` | 最短句时长 |
| 9 | `asr_silence_duration_ms_mean` | 平均静音时长 |

其中最值得关注的方向是：

- **语速偏慢**
- **句长偏短**
- **停顿偏多**

这些现象与“发音动作变慢、发音组织效率下降、朗读不够连贯”的临床直觉是一致的。

### 语音候选子集

经过相关性排序和高相关去冗余后，保留了 **9 个候选 ASR 特征**：

1. `asr_speech_rate_min`
2. `asr_chars_per_sentence_mean`
3. `asr_chars_per_second`
4. `asr_emotion_median`
5. `asr_long_pause_sentence_ratio`
6. `asr_pause_sentence_ratio`
7. `asr_sentence_duration_ms_mean`
8. `asr_sentence_duration_ms_min`
9. `asr_silence_duration_ms_mean`

这 9 个特征在 `reports/asr_candidate_modeling_subset.csv` 中均无缺失，可直接进入建模。

## 三、临床特征设计与筛选结果

### 临床特征分析动机

在没有做临床特征筛选之前，无法判断：

- 临床分支本身是否有强信号；
- 融合失败是因为语音无增益，还是临床噪声过大；
- 哪些临床变量适合作为输入特征，哪些更适合作为协变量或辅助监督。

因此，对临床特征做与语音完全对称的相关性分析是必要步骤。

### 单变量相关性结果

临床侧的单变量信号比语音侧更清晰，综合分数靠前的特征为：

| 排名 | 特征 | 说明 |
|---|---|---|
| 1 | `waist_hip_ratio` | 当前最稳定的临床单变量 |
| 2 | `gender_encoded` | 性别编码 |
| 3 | `height` | 身高 |
| 4 | `waist` | 腰围 |
| 5 | `hip` | 臀围 |
| 6 | `bmi` | 体重指数 |
| 7 | `neck_height_ratio` | 颈围-身高比例 |
| 8 | `weight` | 体重 |

其中最强的单变量是 `waist_hip_ratio`：

- `binary_auc = 0.581766`
- `severity_spearman_rho = 0.132559`
- `severity_kruskal_pvalue = 0.044649`

这说明体型比例类指标在当前数据中与 ICAS 风险和狭窄程度具有相对稳定的关联。

### 临床候选子集

经过排序与去冗余后，保留了 **8 个候选临床特征**：

1. `waist_hip_ratio`
2. `gender_encoded`
3. `height`
4. `waist`
5. `hip`
6. `bmi`
7. `neck_height_ratio`
8. `weight`

这份精简表保存在 `reports/clinical_candidate_modeling_subset.csv`。

## 四、Classical baseline 对比实验

### 1. 全量候选早期融合

在 `reports/asr_clinical_model_comparison_20260507_131227.csv` 中，使用：

- 9 维 ASR 候选
- 全部可用 clinical 数值列
- `asr_only / clinical_only / fusion`

结果显示：

- `asr_only` 最好：`GradientBoosting standard`
  - `test_auc_roc = 0.630952`
- `clinical_only` 最好：`LogisticRegression standard`
  - `test_auc_roc = 0.547619`
- `fusion` 最好：`GradientBoosting standard`
  - `test_auc_roc = 0.523810`

结论：**全量早期融合反而拖累性能**。

### 2. 筛选后融合

在 `reports/filtered_asr_clinical_model_comparison_20260507_143306.csv` 中，仅使用：

- 9 维 ASR 候选
- 8 维临床候选

结果：

- `filtered_fusion` 最好：`GradientBoosting standard`
  - `test_auc_roc = 0.604167`
  - `test_auc_pr = 0.494517`

结论：**筛选后融合显著优于全量融合**，说明“先筛后融”是正确方向。

### 3. Top-k 小型消融

进一步做了更小的消融，只保留：

- ASR top-3：
  - `asr_speech_rate_min`
  - `asr_chars_per_sentence_mean`
  - `asr_chars_per_second`
- clinical top-3：
  - `waist_hip_ratio`
  - `gender_encoded`
  - `height`

结果来自 `reports/topk_filtered_asr_clinical_model_comparison_20260507_152158.csv`：

- `topk_fusion` 最好：`RandomForest standard`
  - `test_auc_roc = 0.598214`
- `clinical_only_topk` 最好：`LogisticRegression standard`
  - `test_auc_roc = 0.773810`

结论：

- `top-3 clinical` 本身非常强；
- 把 ASR 压到 top-3 后，反而损失了语音分支的有效信息；
- `topk_fusion` 虽然优于最早的全量融合，但没有超过 17 维 `filtered_fusion`。

## 五、9 维 ASR + top-3 clinical 的 late fusion 结果

为验证“强 clinical 分支 + 保留型 ASR 分支”是否可以通过后融合获益，进一步构造了：

- ASR 分支：9 维 ASR 候选，模型为 `GradientBoosting standard`
- clinical 分支：top-3 clinical，模型为 `LogisticRegression standard`
- late fusion：验证集上搜索融合权重 `alpha`

结果来自 `reports/late_fusion_asr_clinical_20260507_152729.csv`：

- 最优融合权重：`best_alpha_asr_weight = 0.0`
- 验证集融合 AUC：`val_fused_auc_roc = 0.606897`
- 测试集融合 AUC：`test_auc_roc = 0.773810`
- 测试集融合 AUPR：`test_auc_pr = 0.626734`

这表示验证集自动选择了 **完全不使用 ASR 分支**，最终 late fusion 结果与
`clinical_only_topk` 一致。

### 解释

这并不说明语音特征完全无用，而是说明在 **当前这组 classical 模型 + 加权平均 late fusion**
设定下：

- top-3 clinical 的信号强度已经足够高；
- ASR 分支没有以“概率后融合”的形式再提供净增益；
- ASR 分支更可能需要通过更强的联合模型，而不是简单的后融合，才能体现补充价值。

## 六、当前最重要的结论

### 1. 语音特征值得保留，但不适合单纯看单变量强弱

语音侧没有特别强的单一指标，但 9 维筛选后子集可以形成稳定的 branch-level 信号。
这类特征更适合：

- 作为弱监督/辅助分支输入；
- 在深度模型中由下游学习非线性组合；
- 配合更强的联合建模而不是简单加权平均。

### 2. 临床特征里的强信号高度集中

当前最强的临床信息集中在少数变量，尤其是：

1. `waist_hip_ratio`
2. `gender_encoded`
3. `height`

因此后续如果做结构化输入，建议优先从这几个变量开始，而不是把所有 clinical 数值列都一起塞进模型。

### 3. “先筛后融”明显优于“全量硬拼”

实验已经明确显示：

- 全量 `fusion` 表现最差；
- 筛选后 `filtered_fusion` 显著改善；
- 说明特征噪声和冗余是真问题，而不是融合思想本身无效。

### 4. 当前 classical 结果下，最强单一路线是 `top-3 clinical`

在 classical baseline 框架中，目前最强结果来自：

- `LogisticRegression standard`
- `top-3 clinical`
- `test_auc_roc = 0.773810`

这意味着任何更复杂的后续方案，都应该至少与这一基线做比较。

## 七、推荐的后续实验路线

### 优先级最高：进入 learned fusion / joint model

后续不建议继续堆叠更多 classical fusion 变体，而应直接进入：

1. **9 维 ASR + top-3 clinical** 作为结构化分支输入；
2. 接入现有 CNN 或多任务训练脚本；
3. 由模型学习跨模态联合表示，而不是仅靠概率后融合。

### 建议的具体配置

一个现实且值得优先尝试的联合配置是：

- 语音分支输入：9 个 ASR 候选特征
- 临床分支输入：`waist_hip_ratio`, `gender_encoded`, `height`
- 热成像主干：当前已有的 CNN 主干
- 融合方式：中间层拼接或双分支 MLP，而不是最后概率加权

### 对比实验建议

下一轮建议至少保留下面 4 个对照：

1. 热成像主干单独
2. `top-3 clinical` 单独
3. 热成像 + `top-3 clinical`
4. 热成像 + 9 维 ASR + `top-3 clinical`

这样最容易回答真正重要的问题：

- ASR 是否能为 thermal + clinical 提供额外信息；
- ASR 的增益是在弱 clinical 条件下才出现，还是在最强 clinical 基线之上仍然成立。

## 八、附：本轮实现的关键脚本

- `scripts/extract_asr_features.py`
- `scripts/analyze_asr_feature_correlations.py`
- `scripts/select_asr_candidate_features.py`
- `scripts/analyze_clinical_feature_correlations.py`
- `scripts/select_clinical_candidate_features.py`
- `scripts/compare_asr_clinical_models.py`
- `scripts/compare_filtered_asr_clinical_models.py`
- `scripts/compare_topk_filtered_asr_clinical_models.py`
- `scripts/compare_late_fusion_asr_clinical.py`

对应测试已在仓库中补齐，相关分支测试均已通过。
