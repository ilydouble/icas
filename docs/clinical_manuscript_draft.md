# Noninvasive Identification of Intracranial Atherosclerotic Stenosis Using Facial Thermal Imaging and Simple Anthropometric Clinical Features: A Retrospective Multimodal Study

## Short Title

Facial thermal imaging and clinical indicators for ICAS screening

## Abstract

### Background

Intracranial atherosclerotic stenosis (ICAS) is a major cause of ischemic stroke, yet practical noninvasive screening tools remain limited. We evaluated whether facial thermal imaging, simple anthropometric clinical variables, and their combination could support ICAS risk identification in a retrospective cohort.

### Methods

We constructed a thermal imaging dataset from 2024-2025 clinical acquisitions, yielding 1,387 labeled frontal thermal samples from 522 labeled patients. Patient-level clinical variables were merged from curated clinical and multimodal records. Thermal models were trained with patient-level data splitting to avoid leakage, while inference remained sample-based. Clinical features were ranked by association with binary ICAS labels and stenosis severity, followed by correlation pruning. Multimodal strategies included end-to-end feature concatenation, residual correction, probability-level late fusion, and shallow stacking. Automatic speech recognition (ASR)-derived features were evaluated as an exploratory branch.

### Results

Among structured variables, the strongest signal was concentrated in three anthropometric features: `waist_hip_ratio`, `gender_encoded`, and `height`. A clinical-only model based on these top-3 variables achieved a test AUC-ROC of 0.7738 in the original classical screening pipeline and 0.7680 in refined re-evaluation. The best thermal-only convolutional model reached a test AUC-ROC of 0.6922, while the most stable thermal backbone achieved 0.6397. End-to-end multimodal fusion did not consistently outperform thermal-only baselines. In contrast, a lightweight strategy using a fixed thermal CNN combined with the top-3 clinical features through logistic stacking achieved the best overall multimodal result, with test AUC-ROC 0.7836, test AUC-PR 0.6022, and test F1 0.6446. ASR features provided limited and unstable incremental value and did not improve the best thermal-clinical stacking result.

### Conclusions

In this retrospective study, simple anthropometric clinical variables were stronger and more stable than end-to-end multimodal fusion components, while facial thermal imaging provided complementary information when integrated conservatively. The most promising current strategy for noninvasive ICAS risk identification is a lightweight thermal-clinical framework rather than a complex deep fusion architecture.

## Keywords

intracranial atherosclerotic stenosis; thermal imaging; infrared thermography; multimodal learning; clinical prediction; noninvasive screening

## Introduction

Intracranial atherosclerotic stenosis is an important cause of ischemic stroke and recurrent cerebrovascular events, particularly in Asian populations. Early identification of individuals at elevated ICAS risk is therefore clinically meaningful. However, routine screening typically depends on vascular imaging modalities that are not always convenient for large-scale or repeated assessment. This creates interest in low-burden, noninvasive approaches that could serve as screening or triage tools before confirmatory imaging.

Facial thermal imaging is attractive in this context because it is contact-free, rapid, and potentially sensitive to vascular, autonomic, and perfusion-related changes. At the same time, ICAS risk is also influenced by broader constitutional and metabolic factors that may be partially captured by simple anthropometric measurements. From a translational perspective, the most useful model may not be the most complex one, but rather the one that combines noninvasive imaging with a small number of robust, clinically interpretable variables.

The present study was designed to answer three practical questions. First, how much predictive signal can be extracted from frontal facial thermal images alone? Second, which structured clinical variables provide the strongest and most stable complementary information? Third, does multimodal fusion meaningfully improve performance, and if so, which fusion strategy is most appropriate under realistic sample-size constraints? We also explored ASR-derived speech features as an auxiliary modality, but treated them as exploratory because their mechanistic relationship to ICAS was less direct than that of thermal and anthropometric variables.

## Methods

### Study Design and Data Sources

This was a retrospective modeling study based on a curated multimodal dataset assembled from clinical acquisitions performed in 2024 and 2025. The full data build integrated raw facial thermal images and temperature files, a clinical spreadsheet used for canonical patient identities and clinical descriptors, and a multimodal patient-level table used to supplement labels and anthropometric measurements.

The resulting `full_data` bundle contained 1,424 frontal thermal image samples from 538 included patients. Of these, 1,387 samples from 522 patients had usable binary labels for supervised ICAS modeling and were used for thermal model development. Sample extraction and patient linkage logic are documented in the repository data-construction workflow, including direct patient ID matching and fallback exclusion rules for unmatched records.

### Cohort Characteristics

Among the 522 labeled patients, 355 were classified as non-ICAS and 167 as ICAS, corresponding to a patient-level positive rate of 32.0%. The labeled subset comprised 503 samples from 2024 and 884 samples from 2025. The mean age of labeled patients was 67.28 years (standard deviation 9.07), with an age range of 45 to 89 years. Sex distribution in the labeled cohort was 326 female, 195 male, and 1 missing entry.

### Thermal Image Dataset and Evaluation Split

The thermal pipeline used a sample-level prediction setup in which each frontal thermal image represented one model input. However, the train/validation/test split was performed at the patient level to prevent leakage across images from the same subject. In the currently used split, the thermal-clinical experiments operated on 1,096 training samples, 140 validation samples, and 135 test samples, corresponding to 416, 53, and 53 patients, respectively.

### Clinical Variables

Patient-level structured variables were derived from the merged clinical table. Candidate numeric clinical features were screened by association with binary ICAS labels and stenosis severity, followed by redundancy pruning. The strongest and most stable signal was concentrated in three anthropometric variables:

1. `waist_hip_ratio`
2. `gender_encoded`
3. `height`

Expanded clinical subsets including five, eight, or more features were also evaluated in later refinement analyses.

### Exploratory ASR Features

ASR features were extracted from 2025 speech JSON records and matched to the clinical table by canonical patient ID. Of 405 ASR records, 404 were successfully matched. Candidate ASR features represented speaking rate, pause burden, sentence duration, and related interpretable statistics. Because ASR showed weak and unstable incremental value relative to thermal and clinical information, it was treated as exploratory rather than central to the main clinical manuscript.

### Thermal Modeling

We evaluated multiple convolutional neural network families and training profiles, including MobileNet-based, region-attention, multi-task, simple baseline, and deeper baseline configurations. Grid and refined search showed two important thermal candidates:

- the strongest single thermal result: `deeper_profile_a__dropout_02`, test AUC-ROC 0.6922;
- the most stable thermal branch: `mobilenet_multi_task_profile_a`, test AUC-ROC 0.6397.

Because later multimodal stacking experiments used a fixed thermal CNN, the MobileNet multi-task branch served as the primary thermal probability source in the best current fusion workflow.

### Multimodal Strategies

We evaluated several multimodal strategies:

1. End-to-end sample-level feature concatenation of thermal, clinical, and optional ASR branches.
2. Residual clinical correction, where the thermal branch produced a primary logit and the clinical branch learned a correction term.
3. Probability-level weighted late fusion.
4. Shallow stacking using thermal and clinical probabilities.

The end-to-end models were useful for stress-testing multimodal integration, whereas the shallow fusion experiments were designed to preserve already learned unimodal decision boundaries and test whether complementary information existed at the decision level.

### Statistical Analysis and Outcome Metrics

The primary outcome was binary ICAS classification. Stenosis severity was used in exploratory correlation analysis and in selected severity-weighted model variants. Model performance was summarized using test AUC-ROC, test AUC-PR, and test F1. Threshold-tuned F1 on the validation set was additionally examined for some classical structured-data models. Comparisons were primarily descriptive and focused on ranking practical modeling strategies rather than formal inferential hypothesis testing.

## Results

### Clinical Signal Was Concentrated in a Compact Top-3 Feature Set

Correlation-driven screening and downstream refinement consistently showed that clinical prediction signal was concentrated in a small set of anthropometric features rather than broadly distributed across all available structured variables. In the initial clinical screening pipeline, the top-3 clinical subset reached a test AUC-ROC of 0.7738. In a later refined re-evaluation that explicitly searched across feature subset sizes and model families, the best AUC remained associated with the same top-3 set (`waist_hip_ratio`, `gender_encoded`, and `height`), using a logistic regression model, with test AUC-ROC 0.7680 and test AUC-PR 0.6199. Adding more clinical variables beyond this compact subset did not improve performance and generally reduced AUC.

### Thermal-Only Modeling Produced Moderate but Meaningful Signal

Thermal-only CNN development demonstrated that frontal facial thermography contained useful ICAS-related information, although performance was more modest than the best clinical-only result. In the latest thermal model search, the strongest single configuration was `deeper_profile_a__dropout_02`, which achieved test AUC-ROC 0.6922 and test AUC-PR 0.5124. The most stable thermal family was `mobilenet_multi_task_profile_a`, which achieved test AUC-ROC 0.6397 and test AUC-PR 0.4558. These results indicate that thermal imaging contributes measurable predictive value, but the thermal branch is more sensitive to model configuration and likely to image quality than the compact clinical model.

### End-to-End Multimodal Fusion Did Not Consistently Improve Performance

The first end-to-end multimodal experiments did not deliver reliable improvement. A full thermal-clinical-ASR model selected by validation F1 collapsed into a near-all-positive classifier, yielding test AUC-ROC 0.5000 despite superficially nontrivial F1. After changing checkpoint selection to validation AUC-ROC, the same full multimodal setup improved to test AUC-ROC 0.5201, but still failed to outperform the thermal-only baseline. A thermal-plus-clinical concat model without ASR reached test AUC-ROC 0.5330, and a thermal-plus-ASR concat model reached only 0.4885. A residual clinical correction strategy also underperformed, with test AUC-ROC 0.4521. Together, these results indicate that direct end-to-end fusion was not the most effective way to exploit complementary information in this cohort.

### Conservative Thermal-Clinical Stacking Produced the Best Overall Multimodal Result

Because end-to-end multimodal learning was unstable, we next fixed the thermal CNN and evaluated lightweight probability-level fusion with the top-3 clinical model. In this setting:

- `thermal_only` achieved test AUC-ROC 0.6402 and test AUC-PR 0.4530;
- `clinical_only` achieved test AUC-ROC 0.7572 and test AUC-PR 0.5706;
- weighted late fusion achieved test AUC-ROC 0.7457;
- shallow logistic stacking achieved the best overall result, with test AUC-ROC 0.7836, test AUC-PR 0.6022, and test F1 0.6446.

This result is clinically important because it shows that thermal imaging does add useful information, but only when integrated conservatively rather than forced into an unstable end-to-end fusion architecture.

### ASR Features Did Not Improve the Best Thermal-Clinical Result

ASR was further evaluated as a third modality in shallow fusion. However, adding ASR probability to linear stacking reduced performance substantially: the three-way logistic stacking result dropped to test AUC-ROC 0.5991. A shallow depth-2 decision tree produced a higher F1 in the three-way setting, but its AUC remained below that of the two-branch thermal-clinical logistic stacker. Accordingly, ASR was not retained as part of the preferred multimodal workflow for the current clinical manuscript.

## Discussion

### Principal Findings

This study provides three main findings. First, the strongest structured clinical signal for ICAS risk identification was concentrated in a remarkably small set of anthropometric variables: `waist_hip_ratio`, sex, and height. Second, facial thermal imaging carried meaningful but less stable predictive information. Third, the best overall performance was achieved not by a complex end-to-end multimodal network, but by a lightweight combination of a fixed thermal CNN and a simple clinical model through logistic stacking.

These observations are relevant because they challenge a common assumption in multimodal machine learning: that more deeply integrated models should automatically outperform simpler methods. In our cohort, this was not the case. The most complex multimodal models were the most unstable, while the most clinically defensible and practically useful result came from preserving each branch's strength and combining them only at the decision level.

### Interpretation of the Clinical Findings

The dominance of `waist_hip_ratio`, sex, and height suggests that body habitus and anthropometric constitution are strongly associated with ICAS status in this dataset. These variables are easy to acquire, inexpensive, and clinically interpretable. Their concentration into a top-3 signal is also important methodologically: it implies that adding more structured variables may introduce noise rather than information. For a clinically oriented manuscript, this supports a simple and translatable message: a small number of anthropometric indicators may already provide substantial ICAS-related risk information.

### Interpretation of the Thermal Findings

Thermal performance was consistently lower than the best clinical-only performance, but still clearly above null in the best CNN configurations. This indicates that facial thermography is not merely a passive correlate of the structured variables. Instead, it likely captures additional physiological information related to perfusion, vascular asymmetry, autonomic regulation, or microvascular response. The fact that thermal information improved overall performance only when combined conservatively suggests that the signal is real but fragile. Its clinical value may depend heavily on image quality, acquisition consistency, and careful integration strategy.

### Why Simple Stacking Outperformed Deep Fusion

The relative success of shallow logistic stacking probably reflects the structure of the current dataset. The clinical branch is low-dimensional, strong, and stable, whereas the thermal branch is high-dimensional and more variable. End-to-end deep fusion forces these signals to interact during optimization and may allow one branch to destabilize the other. By contrast, shallow stacking allows each branch to learn independently and only combines final predictive probabilities. This approach is particularly attractive in modest-sized multimodal cohorts, where optimization instability can easily outweigh theoretical representational benefits.

### Clinical Implications

From a translational standpoint, the current findings support a pragmatic screening framework. A simple anthropometric clinical model can serve as a strong baseline, while facial thermal imaging can provide complementary information when integrated carefully. This suggests a realistic pathway for noninvasive ICAS risk stratification in settings where rapid, low-burden assessment is desirable. Importantly, the data do not support the routine inclusion of ASR-derived features at this stage, which simplifies the pathway toward clinical deployment.

## Limitations

Several limitations should be acknowledged. First, this was a retrospective single-center study without external validation, so generalizability remains uncertain. Second, the thermal pipeline used a sample-level inference design, even though train/validation/test partitioning was performed at the patient level. Third, thermal performance likely depends on acquisition quality, face masking, and ROI stability, but these factors were not yet explicitly modeled through formal quality stratification. Fourth, multimodal comparisons involving ASR were limited by the smaller matched speech subset. Finally, the best multimodal result currently uses a stable MobileNet thermal branch rather than the strongest single-run thermal backbone, so the present stacking result may still underestimate the maximum achievable thermal-clinical performance.

## Conclusions

In this retrospective multimodal ICAS study, simple anthropometric clinical variables provided the strongest and most stable structured signal, while facial thermal imaging contributed complementary value when integrated through a lightweight decision-level strategy. Complex end-to-end multimodal fusion was not superior to conservative fusion. The current best-performing workflow is a fixed thermal CNN combined with three simple clinical variables through logistic stacking. These findings support a clinically oriented development path centered on interpretable risk indicators, disciplined thermal preprocessing, and simple but robust multimodal integration.

## Suggested Figure and Table List

### Figures

1. Study flow diagram showing source cohorts, exclusions, and final labeled dataset.
2. Thermal modeling and multimodal analysis workflow.
3. Comparison of thermal-only, clinical-only, and thermal-clinical fusion performance.
4. Optional feature-importance or coefficient plot for the top-3 clinical model.

### Tables

1. Baseline characteristics of the labeled cohort.
2. Distribution of thermal samples and patient-level labels.
3. Top-ranked clinical and ASR features.
4. Main model performance comparison across thermal-only, clinical-only, and fusion strategies.
5. Exploratory multimodal ablation results, including negative ASR findings.

## Suggested Journal Positioning

This draft is best aligned with a clinically oriented imaging, digital medicine, cerebrovascular, or translational diagnostics journal. The strongest paper narrative is not “a novel deep network,” but rather “a systematic evaluation of what truly adds value in noninvasive multimodal ICAS screening.” That framing is both more defensible and more consistent with the current experimental evidence.
