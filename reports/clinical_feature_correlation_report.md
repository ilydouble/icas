# Clinical Feature Correlation Analysis

## Dataset Summary

- Total patient rows: 538
- Binary ICAS positives: 172
- Candidate clinical numeric features: 13

## Top Features For Binary ICAS

| feature_name | binary_corr | binary_auc | binary_effect_size_d | combined_score |
| --- | --- | --- | --- | --- |
| waist_hip_ratio | 0.119904 | 0.581766 | 0.258431 | 0.148165 |
| gender_encoded | -0.081631 | 0.457671 | -0.175255 | 0.096068 |
| height | 0.080625 | 0.547663 | 0.173123 | 0.092128 |
| waist | 0.054415 | 0.536768 | 0.116520 | 0.065937 |
| bmi | -0.048424 | 0.484600 | -0.103737 | 0.042275 |
| neck_height_ratio | -0.045819 | 0.490824 | -0.098144 | 0.040963 |
| bmi_category | -0.043580 | 0.481108 | -0.093338 | 0.040931 |
| weight | 0.034577 | 0.518151 | 0.074011 | 0.039367 |
| hip | -0.034518 | 0.476760 | -0.073848 | 0.042585 |
| waist_height_ratio | 0.024839 | 0.524391 | 0.053166 | 0.035590 |

## Top Features For Stenosis Severity

| feature_name | severity_spearman_rho | severity_kruskal_pvalue | combined_score |
| --- | --- | --- | --- |
| waist_hip_ratio | 0.132559 | 0.044649 | 0.148165 |
| gender_encoded | -0.085059 | 0.199128 | 0.096068 |
| height | 0.067100 | 0.172692 | 0.092128 |
| waist | 0.056042 | 0.658476 | 0.065937 |
| age | 0.042922 | 0.009789 | 0.015346 |
| waist_height_ratio | 0.039966 | 0.848044 | 0.035590 |
| hip | -0.038733 | 0.868427 | 0.042585 |
| age_group | 0.031394 | 0.110489 | 0.019460 |
| weight | 0.030729 | 0.315441 | 0.039367 |
| neck | 0.017186 | 0.653301 | 0.017317 |

## Notes

- `binary_corr` is Pearson correlation against `has_icas`.
- `binary_auc` measures single-feature discrimination ability for ICAS.
- `severity_spearman_rho` measures monotonic association with `stenosis_multiclass`.
