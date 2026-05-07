# ASR Feature Correlation Analysis

## Dataset Summary

- Total ASR rows: 405
- Matched clinical rows: 404
- Binary ICAS positives: 128
- Candidate ASR numeric features: 44

## Top Features For Binary ICAS

| feature_name | binary_corr | binary_auc | binary_effect_size_d | combined_score |
| --- | --- | --- | --- | --- |
| asr_speech_rate_min | -0.084918 | 0.463457 | -0.182733 | 0.090687 |
| asr_chars_per_sentence_mean | -0.047483 | 0.476279 | -0.101923 | 0.053860 |
| asr_chars_per_sentence | -0.047483 | 0.476279 | -0.101923 | 0.053860 |
| asr_long_pause_sentence_ratio | -0.039009 | 0.485295 | -0.083704 | 0.043654 |
| asr_sentence_duration_ms_mean | -0.038016 | 0.485762 | -0.081569 | 0.038989 |
| asr_pause_sentence_ratio | -0.037180 | 0.472175 | -0.079774 | 0.042363 |
| asr_chars_per_second | -0.035896 | 0.480384 | -0.077015 | 0.045066 |
| asr_sentence_duration_ms_min | -0.035848 | 0.485536 | -0.076912 | 0.037219 |
| asr_emotion_median | -0.034868 | 0.473590 | -0.074805 | 0.043801 |
| asr_speech_rate_std | 0.032501 | 0.507756 | 0.069722 | 0.033779 |

## Top Features For Stenosis Severity

| feature_name | severity_spearman_rho | severity_kruskal_pvalue | combined_score |
| --- | --- | --- | --- |
| asr_silence_duration_ms_max | -0.062297 | 0.669115 | 0.027968 |
| asr_speech_rate_min | -0.058555 | 0.797405 | 0.090687 |
| asr_silence_duration_ms_mean | -0.052191 | 0.479311 | 0.034605 |
| asr_repeated_adjacent_char_count | -0.050753 | 0.142383 | 0.024670 |
| asr_silence_duration_ms_total | -0.049161 | 0.280462 | 0.033201 |
| asr_chars_per_second | -0.047735 | 0.004526 | 0.045066 |
| asr_silence_to_duration_ratio | -0.043670 | 0.340282 | 0.033844 |
| asr_reference_insertion_proxy | -0.042909 | 0.074239 | 0.020578 |
| asr_reference_length_ratio | -0.042813 | 0.057159 | 0.021720 |
| asr_observed_length | -0.042813 | 0.057159 | 0.021720 |

## Notes

- `binary_corr` is Pearson correlation against `has_icas` and is equivalent to a point-biserial view here.
- `binary_auc` measures single-feature discrimination ability for ICAS.
- `severity_spearman_rho` measures monotonic association with `stenosis_multiclass`.
- Prioritize features that rank well in both binary and severity views before entering multi-task modeling.
