# Bootstrap Comparisons

Bootstrap uses `2000` paired resamples with seed `7`.

| Comparison | Macro F1 Delta (95% CI) | Exact Delta (95% CI) | Abnormal Partial Delta (95% CI) |
| --- | --- | --- | --- |
| MIMIC audited: MedGemma-27B structured multi-agent minus single-VLM | 0.174 [0.059, 0.290] | -0.041 [-0.100, 0.000] | 0.362 [0.179, 0.556] |
| Private: CheXagent labels_only single-VLM minus multi-agent | 0.295 [0.192, 0.387] | -0.040 [-0.140, 0.060] | 0.678 [0.450, 0.880] |
| MIMIC audited: MedGemma-27B structured multi-agent minus CheXagent structured multi-agent | 0.160 [0.018, 0.300] | 0.219 [0.040, 0.400] | 0.325 [0.069, 0.572] |