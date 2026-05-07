# Future Work for ICAS Thermal Imaging, Clinical Modeling, and Multimodal Fusion

## Overview

The current study established three important findings. First, a compact clinical model built on `waist_hip_ratio`, `gender_encoded`, and `height` provides the most stable structured-signal baseline. Second, thermal facial imaging contains useful predictive information, but its gains are not consistently realized through end-to-end multimodal fusion. Third, the strongest fusion result so far comes from a lightweight strategy: a fixed thermal CNN combined with the top-3 clinical features through shallow logistic stacking.  

These results suggest that future work should not primarily focus on adding more complex fusion architectures. Instead, the next phase should prioritize improving the quality, robustness, and interpretability of the thermal signal itself, while preserving the clinical branch as a strong and simple comparator.

## 1. Quality Stratification of Thermal Images

The most actionable next step is to stratify the thermal dataset by acquisition quality. Facial thermal prediction is likely sensitive to factors such as pose deviation, facial occlusion, hair interference, glasses, imperfect face masking, missing regions of interest, and unstable temperature boundaries. These sources of variation may dilute disease-related thermal patterns and limit the observed benefit of the imaging branch.

Future work should assign each thermal sample to high-, medium-, or low-quality strata using a combination of manual review and rule-based indicators derived from mask coverage, ROI completeness, and image consistency. After stratification, three analyses should be performed:

1. Compare thermal-only model performance across quality strata.
2. Reweight or exclude low-quality samples during training.
3. Reassess thermal-clinical fusion after restricting to high-quality images.

If performance improves substantially in the high-quality subset, this would support an important clinical conclusion: strict acquisition and quality control may be a prerequisite for reliable thermal screening in ICAS.

## 2. External Thermal Data for Representation Learning

Another promising direction is the use of external thermal face datasets. Public infrared face datasets are unlikely to provide ICAS labels, but they may still be valuable for representation learning. Their main utility would be in pretraining a thermal backbone to better capture generic facial thermal structure, cross-subject variability, and robust region-level features before fine-tuning on the ICAS cohort.

This strategy should be approached cautiously. External datasets may differ in sensor type, temperature calibration, acquisition distance, and population distribution. Therefore, future work should treat them primarily as a pretraining resource rather than as directly merged supervised training data.

The recommended sequence is:

1. Pretrain a thermal backbone on external infrared face data.
2. Fine-tune the model on the ICAS thermal dataset only.
3. Compare the pretrained backbone against the current best thermal-only baselines.

This design would allow the study to test whether external thermal data improves representation robustness without conflating domain shift with disease classification.

## 3. ROI and Thermal Signal Refinement

Current experiments indicate that thermal performance may still be constrained by preprocessing quality. Beyond broad quality stratification, future work should focus on more precise thermal ROI definition and signal refinement. Potential directions include:

- more stable face alignment and region localization,
- better mask quality control,
- region-specific normalization,
- explicit foreground-background thermal contrast features,
- and refined temperature summary statistics from forehead, cheek, and periorbital regions.

These refinements are attractive because they are clinically interpretable. If regional thermal patterns become more stable after preprocessing improvements, the study could move from a generic image-classification framing toward a more clinically meaningful biomarker-oriented analysis.

## 4. Patient-Level Robustness and Evaluation

Although the current thermal pipeline is operational, future work should continue to distinguish between sample-level and patient-level conclusions. Even when a sample-level setup is retained for fairness, downstream reporting should emphasize patient-level robustness, patient-balanced weighting, and subgroup consistency. This is especially important if acquisition count varies across patients or if image quality differs within the same subject.

Future analyses should therefore include:

- repeated patient-level resampling or multi-seed evaluation,
- subgroup performance by quality tier,
- subgroup performance by stenosis severity,
- and calibration analysis for clinically meaningful decision thresholds.

This would strengthen the translational value of the results and reduce the risk that isolated sample-level gains overstate clinical utility.

## 5. Conservative Multimodal Development

Current evidence suggests that future multimodal work should remain conservative. The clinical branch is already strong and interpretable, while ASR features have shown limited and unstable incremental value. Therefore, the most sensible multimodal roadmap is:

1. keep the clinical top-3 model as the structured baseline,
2. continue to improve the thermal branch independently,
3. retain shallow thermal-clinical fusion as the primary multimodal strategy,
4. and treat ASR as an exploratory branch rather than a default component.

This roadmap is methodologically important. It reframes multimodal modeling not as a search for maximal architectural complexity, but as a search for the most reliable and clinically defensible combination of signals.

## 6. Likely Clinical Contribution

If the study evolves into a clinically oriented manuscript, the future work agenda should support a clear narrative: noninvasive ICAS assessment may benefit from multimodal information, but the most meaningful advances will likely come from better thermal signal quality and disciplined model integration rather than from increasingly complex deep fusion schemes.

Under this framing, future work can contribute in two ways at once:

- methodologically, by identifying which fusion strategies are robust in small multimodal cohorts;
- clinically, by clarifying when facial thermal imaging adds value beyond simple anthropometric risk indicators.

## Recommended Priority Order

The most practical order for the next phase is:

1. thermal image quality stratification and filtering,
2. thermal-only re-evaluation on high-quality subsets,
3. shallow thermal-clinical fusion re-evaluation after quality control,
4. external thermal pretraining for backbone improvement,
5. refined ROI-level thermal feature extraction,
6. and only then renewed exploration of ASR or deeper multimodal fusion.

This sequence is designed to maximize interpretability, minimize wasted experimentation, and keep the project aligned with a strong clinical-paper narrative.
