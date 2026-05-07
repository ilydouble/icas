# ICAS 热力图-临床-ASR 多模态融合实验报告

## 概览

本报告汇总当前仓库中与多模态融合直接相关的实验链路，包括：

- 语音与临床特征筛选结果
- classical baseline 融合结果
- 热力图单模态 CNN 搜索结果
- 热力图 + 临床 + ASR 的 sample-level 多模态 CNN 实验
- 失败路径与下一步建议

本轮最重要的目标不是寻找单一最高分，而是回答三个更关键的问题：

1. 语音与临床哪一支在当前融合框架下真正提供增益。
2. 为什么 `clinical top-3` 和热力图单独都不错，但直接融合反而不稳。
3. 在当前数据规模下，下一步应该保留哪条主线，砍掉哪些试验方向。

结论先行：

- `clinical top-3` 是目前最稳定的结构化分支。
- ASR 分支在 current concat fusion 下更像噪声源，而不是稳定增益。
- `thermal + ASR + clinical` 比 `thermal + ASR` 更好，说明 clinical 能纠偏。
- 但当前多模态 CNN 仍未稳定超过强热力图单模态基线。
- 新增的 `residual_clinical` 融合也已完成首轮实验，但目前仍未优于普通 `thermal + clinical` concat。

## 一、结构化分支的前置筛选结果

完整筛选过程已在 [asr_clinical_feature_screening_report.md](/Users/liruirui/Documents/code/icas/docs/asr_clinical_feature_screening_report.md) 记录，这里仅保留与后续融合最相关的结论。

### 1. ASR 特征

基于 `datasets/ASR/json_results/*.json` 提取了可解释语音统计特征，并按 `canonical_patient_id` 合并 `datasets/full_data/patient_clinical_data.csv`。

当前保留的 9 维候选 ASR 特征为：

1. `asr_speech_rate_min`
2. `asr_chars_per_sentence_mean`
3. `asr_chars_per_second`
4. `asr_emotion_median`
5. `asr_long_pause_sentence_ratio`
6. `asr_pause_sentence_ratio`
7. `asr_sentence_duration_ms_mean`
8. `asr_sentence_duration_ms_min`
9. `asr_silence_duration_ms_mean`

这些特征在相关性分析中呈现“弱但可组合”的模式，单变量解释力整体不强。

### 2. 临床特征

当前保留的临床候选特征里，最强信号高度集中在 top-3：

1. `waist_hip_ratio`
2. `gender_encoded`
3. `height`

其中 `waist_hip_ratio` 是最稳定的单变量。

### 3. Classical baseline 的结论

相关产物：

- `reports/asr_clinical_model_comparison_20260507_131227.csv`
- `reports/filtered_asr_clinical_model_comparison_20260507_143306.csv`
- `reports/topk_filtered_asr_clinical_model_comparison_20260507_152158.csv`
- `reports/late_fusion_asr_clinical_20260507_152729.csv`

核心结论：

- `clinical_only_topk` 很强，最好 `test_auc_roc = 0.7738`
- `ASR-only` 有一定信号，但不如 top-3 clinical 稳
- 简单 late fusion 没有让 `9 ASR + top-3 clinical` 超过 `clinical_only_topk`

这已经提前预警：结构化信息里，clinical 更可能是强而稳定的支路，ASR 更可能需要更保守的接入方式。

## 二、热力图单模态 CNN 主线

网格与精细化搜索结果见 [model_search_report.md](/Users/liruirui/Documents/code/icas/docs/model_search_report.md)。

### 1. Coarse / refined search 结论

当前最关键的热力图单模态候选有两条：

- 冲单点最优：`deeper_profile_a__dropout_02`
  - `test_auc_roc = 0.6922`
  - `test_auc_pr = 0.5124`
- 走稳定主线：`mobilenet_multi_task_profile_a`
  - 三 seed `mean_auc = 0.6277`
  - `std_auc = 0.0087`

### 2. 当前多模态主线选用的热力图初始化

本轮多模态实验主要使用：

- `model = mobilenet`
- `multi_task = true`
- `dropout = 0.3`
- `lr = 0.001`
- `lambda_sev = 0.3`
- `init_checkpoint = reports/best_cnn_v3.pt`

对应的单模态结果文件为：

- `reports/cnn_v3_results_20260507_171253.json`

该单模态结果为：

- `test_auc_roc = 0.6397`
- `test_auc_pr = 0.4558`
- `test_f1 = 0.5140`
- tuned `test_f1 = 0.5401`

这是后续所有多模态对照的主要参照物。

## 三、多模态 CNN 实验记录

### 1. Full multimodal, 错误 checkpoint 选择

结果文件：

- `reports/cnn_multimodal_results_20260507_171552.json`

配置特点：

- `selection_metric = f1`
- `freeze_thermal_epochs = 1`
- `fusion_mode = concat`
- `ASR + clinical` 全开

测试结果：

- `test_auc_roc = 0.5000`
- `test_auc_pr = 0.3661`
- `test_f1 = 0.5359`

这是一个**失败实验**。虽然 `F1` 不算极低，但模型基本退化成近似“全阳性”：

- `recall = 1.0`
- `precision = 0.3661`
- `bal_acc = 0.5`

根因分析：

- 这版按 `val_f1` 选 checkpoint，选中了阈值行为投机的 epoch，而不是排序能力更好的 epoch。
- 从 `cnn_multimodal_history_20260507_171552.json` 可以看到，`val_auc_roc` 最好的 epoch 并没有被保存下来。

结论：

- `selection_metric = f1` 不适合当前多模态训练默认值。
- 这个失败直接促成后续默认值改成 `selection_metric = auc_roc`。

### 2. Full multimodal, 修正 checkpoint 选择

结果文件：

- `reports/cnn_multimodal_results_20260507_172418.json`

配置特点：

- `selection_metric = auc_roc`
- `freeze_thermal_epochs = 0`
- `fusion_mode = concat`
- `ASR + clinical` 全开

测试结果：

- `test_auc_roc = 0.5201`
- `test_auc_pr = 0.4037`
- `test_f1 = 0.4412`
- tuned `test_f1 = 0.5253`

这版的意义不是分数特别高，而是：

- 模型从“全阳性塌缩”中恢复成正常分类器
- `precision / recall` 达到更平衡的状态
- 在当前 complete-case 子集上，成为后续消融的合理基准

但它依然没有超过强单模态热力图基线。

### 3. `thermal + clinical` concat

结果文件：

- `reports/cnn_multimodal_results_20260507_173225.json`

配置特点：

- `--disable-asr`
- `fusion_mode = concat`

测试结果：

- `test_auc_roc = 0.5330`
- `test_auc_pr = 0.3399`
- `test_f1 = 0.5083`
- tuned `test_f1 = 0.5103`

注意：这版运行在更大的样本池上：

- `train/val/test samples = 1096 / 140 / 135`
- `patients = 416 / 53 / 53`

也就是说，它不受 ASR complete-case 限制，因此**不能和 full multimodal 直接硬比**，但可以作为趋势参考。

趋势上看：

- clinical 分支单独接入 thermal 后，没有出现像 ASR 那样明显的训练不稳定
- 它更像“可用但增益有限”的结构化支路

### 4. `thermal + ASR` concat

结果文件：

- `reports/cnn_multimodal_results_20260507_173626.json`

配置特点：

- `--disable-clinical`
- `fusion_mode = concat`

测试结果：

- `test_auc_roc = 0.4885`
- `test_auc_pr = 0.3582`
- `test_f1 = 0.0000`
- tuned `test_f1 = 0.4923`

这是另一个明显的**失败实验**。

其失败模式与前面的 clinical-only 不同：

- 默认测试结果直接退化到全阴性
- `best_epoch = 1`
- 验证集最优 AUC 出现在非常早的阶段

结论：

- 在当前 concat fusion 框架里，ASR 单独接入 thermal 时最不稳定
- ASR 当前更像噪声源，而不是稳定增益

### 5. `thermal + clinical` residual correction

结果文件：

- `reports/cnn_multimodal_results_20260507_175600.json`

配置特点：

- `--disable-asr`
- `--fusion-mode residual_clinical`

设计动机：

- 不再让 clinical 与 thermal 平权拼接
- 改为让 thermal 先给出主预测
- clinical 学一个 residual correction，去修正 thermal logits

测试结果：

- `test_auc_roc = 0.4521`
- `test_auc_pr = 0.3381`
- `test_f1 = 0.5083`
- tuned `test_f1 = 0.4756`

这是一次**没有达到预期的失败实验**。

虽然 residual clinical 在理论上更保守，但当前这次首轮结果说明：

- clinical correction 的方式还没有稳定转化成排序增益
- 现阶段它甚至不如简单的 `thermal + clinical` concat

所以目前不能直接把 residual clinical 当成更优默认方案。

## 四、当前消融结论

### 1. 在当前 concat 结构里，clinical 比 ASR 更有价值

最稳的对照信号是：

- `thermal + ASR` 最弱
- `thermal + ASR + clinical` 明显好于 `thermal + ASR`

这说明：

- clinical 分支在 current setting 下确实能纠偏
- ASR 分支单独接入 thermal 的收益最差

### 2. ASR 当前更像“高方差弱补充”，不是主线支路

从 classical baseline 到 CNN concat fusion，ASR 都呈现类似现象：

- 单独存在一些信号
- 但不稳定
- 与强 clinical 分支融合时，常常没有带来净增益

因此，当前项目阶段不建议把 ASR 作为默认必选支路。

### 3. `thermal + clinical` 是当前最值得保留的结构化融合主线

尽管它还没有稳定超过强热力图单模态基线，但它至少满足：

- 结构简单
- 解释性强
- 失败模式比 ASR 更轻
- 更接近 clinical-only 的已有结论

### 4. 当前没有任何多模态版本明确击败强热力图单模态

这是整个报告里最重要的保守结论。

现有结果足以说明：

- “加结构化信息”不是自动增益
- 当前多模态实验更多是在做机制探索
- 在没有更稳的融合策略前，热力图单模态仍然是最可靠主线

## 五、失败情况汇总

本轮应明确保留的失败经验包括：

1. `selection_metric = f1` 的 full multimodal
   - 容易把 checkpoint 选到近似全阳性策略上
   - 对多模态训练尤其危险

2. `thermal + ASR` concat
   - 排序能力弱
   - 训练早期高波动
   - 默认测试结果直接退化到全阴性

3. `thermal + clinical residual correction` 首轮版本
   - 理论上更保守
   - 但当前实现尚未带来实际提升

这些失败结果不应被删除，因为它们明确缩小了下一步搜索空间。

## 六、推荐的阶段性结论

如果现在需要给项目一个清晰的阶段性判断，我会写成：

1. 热力图单模态依然是当前最可靠主线。
2. 临床 top-3 特征是最有价值的结构化补充分支。
3. ASR 特征当前不建议继续作为默认融合输入。
4. 现有多模态结果显示“直接 concat 融合”不足以稳定释放临床与语音的互补性。

## 七、下一步建议

今天之后，建议主线收缩为：

1. 保留热力图单模态 baseline
   - `mobilenet_multi_task_profile_a`
   - `deeper_profile_a__dropout_02`

2. 继续推进 `thermal + clinical`
   - 但优先做更小、更稳的策略改动
   - 不再默认带 ASR

3. 暂停 ASR 深度融合扩展
   - 至少在 current pipeline 下先暂停
   - 如要重启，建议改为更轻的 gating 或后验修正，而不是 feature concat

4. 报告比较时始终注明评估池差异
   - `thermal + clinical` 目前运行在更大的 53-patient 测试池
   - `full multimodal` 和 `thermal + ASR` 受 complete-case 约束，只在 38-patient 测试池上可直接互比

这意味着明天之后最合理的工作方向是：

- 继续把 `thermal + clinical` 主线做稳
- 用更严格的同池比较判断它是否真超过热力图单模态
- 暂不继续在 ASR 上投入大量调参时间
