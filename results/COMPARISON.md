# 2D vs 2.5D vs 3D — cross-representation comparison

All arms share one pipeline: identical HU window [-1000, 400], identical patient-level splits, identical label columns, identical loss (class-weighted cross-entropy, label smoothing 0.10), optimiser (AdamW, lr 1e-3, weight decay 0.05, cosine), early stopping on validation macro-F1, and identical metric code. Only the input representation and encoder change.

| representation | input | encoder |
|---|---|---|
| 2D | `(1, 72, 80)` — central slice 52 | ResNet-18 |
| 2.5D | `(5, 72, 80)` — slices 50–54 | ResNet-18 |
| 3D | `(1, 104, 72, 80)` — full crop | 3D ResNet-10 |

All numbers are test-set, mean ± std over 3 seeds.

**Majority-class baselines:** 3-class macro-F1 0.2138 (accuracy 0.4722); binary macro-F1 0.3589 (accuracy 0.5598).

## Model A — supervised, no augmentation

| metric | 2D (1 slice) | 2.5D (5 slices) | 3D (104 slices) |
|---|---|---|---|
| **macro F1** | 0.6059 ± 0.0239 | 0.5785 ± 0.0215 | 0.5653 ± 0.0407 |
| accuracy | 0.6002 ± 0.0239 | 0.5690 ± 0.0203 | 0.5564 ± 0.0403 |
| balanced accuracy | 0.6061 ± 0.0227 | 0.5809 ± 0.0209 | 0.5542 ± 0.0394 |
| macro precision | 0.6097 ± 0.0231 | 0.5911 ± 0.0067 | 0.5847 ± 0.0435 |
| macro recall | 0.6061 ± 0.0227 | 0.5809 ± 0.0209 | 0.5542 ± 0.0394 |
| ROC-AUC | 0.7565 ± 0.0146 | 0.7405 ± 0.0110 | 0.7223 ± 0.0217 |
| F1 benign | 0.5362 ± 0.0427 | 0.5328 ± 0.0145 | 0.4583 ± 0.0604 |
| F1 indeterminate | 0.6001 ± 0.0270 | 0.5550 ± 0.0332 | 0.5733 ± 0.0357 |
| F1 malignant | 0.6814 ± 0.0132 | 0.6477 ± 0.0488 | 0.6643 ± 0.0316 |
| best val macro F1 | 0.5982 | 0.6064 | 0.5698 |
| best epoch | [30, 14, 8] | [37, 14, 12] | [17, 25, 34] |
| parameters | 11,171,779 | 11,184,323 | 14,357,059 |
| train min/seed | 2.1 | 3.2 | 24.5 |

## Model B — MoCo v2 → fine-tune

| metric | 2D (1 slice) | 2.5D (5 slices) | 3D (104 slices) |
|---|---|---|---|
| **macro F1** | 0.5957 ± 0.0086 | 0.5875 ± 0.0111 | 0.6083 ± 0.0078 |
| accuracy | 0.5926 ± 0.0086 | 0.5758 ± 0.0115 | 0.6052 ± 0.0093 |
| balanced accuracy | 0.5908 ± 0.0129 | 0.5798 ± 0.0098 | 0.5965 ± 0.0106 |
| macro precision | 0.6032 ± 0.0019 | 0.6000 ± 0.0139 | 0.6279 ± 0.0065 |
| macro recall | 0.5908 ± 0.0129 | 0.5798 ± 0.0098 | 0.5965 ± 0.0106 |
| ROC-AUC | 0.7579 ± 0.0087 | 0.7368 ± 0.0021 | 0.7373 ± 0.0073 |
| F1 benign | 0.5169 ± 0.0108 | 0.4906 ± 0.0190 | 0.4860 ± 0.0156 |
| F1 indeterminate | 0.6080 ± 0.0054 | 0.5801 ± 0.0146 | 0.6315 ± 0.0073 |
| F1 malignant | 0.6623 ± 0.0117 | 0.6917 ± 0.0192 | 0.7074 ± 0.0085 |
| best val macro F1 | 0.6135 | 0.5997 | 0.5662 |
| best epoch | [15, 39, 9] | [15, 40, 19] | [49, 21, 46] |
| parameters | 11,171,779 | 11,184,323 | 14,357,059 |
| train min/seed | 2.2 | 3.0 | 31.7 |

## Model C — supervised, mild augmentation

| metric | 2D (1 slice) | 2.5D (5 slices) | 3D (104 slices) |
|---|---|---|---|
| **macro F1** | 0.6134 ± 0.0075 | 0.5934 ± 0.0158 | 0.5738 ± 0.0134 |
| accuracy | 0.6128 ± 0.0146 | 0.5850 ± 0.0211 | 0.5657 ± 0.0090 |
| balanced accuracy | 0.6110 ± 0.0032 | 0.5913 ± 0.0161 | 0.5676 ± 0.0150 |
| macro precision | 0.6223 ± 0.0092 | 0.5994 ± 0.0176 | 0.5842 ± 0.0136 |
| macro recall | 0.6110 ± 0.0032 | 0.5913 ± 0.0161 | 0.5676 ± 0.0150 |
| ROC-AUC | 0.7595 ± 0.0046 | 0.7340 ± 0.0208 | 0.7197 ± 0.0138 |
| F1 benign | 0.5421 ± 0.0099 | 0.5228 ± 0.0524 | 0.4436 ± 0.0484 |
| F1 indeterminate | 0.6274 ± 0.0359 | 0.5853 ± 0.0286 | 0.5809 ± 0.0041 |
| F1 malignant | 0.6708 ± 0.0274 | 0.6720 ± 0.0378 | 0.6970 ± 0.0222 |
| best val macro F1 | 0.5962 | 0.5879 | 0.5375 |
| best epoch | [12, 50, 11] | [38, 17, 29] | [43, 40, 42] |
| parameters | 11,171,779 | 11,184,323 | 14,357,059 |
| train min/seed | 2.4 | 3.4 | 42.7 |

## Binary — supervised, mild augmentation

| metric | 2D (1 slice) | 2.5D (5 slices) | 3D (104 slices) |
|---|---|---|---|
| **macro F1** | 0.8227 ± 0.0024 | 0.8047 ± 0.0063 | 0.7866 ± 0.0307 |
| accuracy | 0.8293 ± 0.0045 | 0.8118 ± 0.0045 | 0.7943 ± 0.0310 |
| balanced accuracy | 0.8182 ± 0.0007 | 0.8009 ± 0.0073 | 0.7826 ± 0.0295 |
| macro precision | 0.8391 ± 0.0122 | 0.8189 ± 0.0078 | 0.8003 ± 0.0361 |
| macro recall | 0.8182 ± 0.0007 | 0.8009 ± 0.0073 | 0.7826 ± 0.0295 |
| ROC-AUC | 0.8644 ± 0.0041 | 0.8612 ± 0.0159 | 0.8443 ± 0.0444 |
| F1 benign | 0.8565 ± 0.0080 | 0.8413 ± 0.0047 | 0.8269 ± 0.0287 |
| F1 malignant | 0.7888 ± 0.0033 | 0.7681 ± 0.0133 | 0.7462 ± 0.0328 |
| best val macro F1 | 0.8328 | 0.8194 | 0.7819 |
| best epoch | [20, 19, 16] | [27, 14, 16] | [35, 22, 35] |
| parameters | 11,171,266 | 11,183,810 | 14,356,546 |
| train min/seed | 1.2 | 1.7 | 19.0 |
