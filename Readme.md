# VinoHealth MLP Lab (PyTorch)

Multilayer perceptrons (MLPs) built in PyTorch for two research streams at the fictional firm VinoHealth Analytics: wine variety classification and diabetes progression prediction.

## Overview

| Stream | Task | Dataset | Loss | Optimizer |
|---|---|---|---|---|
| Wine classification | Predict grape variety (3 classes) from 13 chemical measurements | scikit-learn `load_wine` | `CrossEntropyLoss` | Adam (lr = 0.01) |
| Diabetes progression | Predict a disease progression score from 10 patient biomarkers | scikit-learn `load_diabetes` | `MSELoss` | Adam (lr = 0.01) |

## Contents

- `CG_C09_M07.ipynb` - the completed lab notebook (all steps run top to bottom)
- `C09_M07_TL.ipynb` - lecture notebook: MLPs for regression (California housing) and classification (Iris) in both PyTorch and TensorFlow
- `C09_M07_Lecture.pptx` - lecture slides
- `C09_M07_TL01.pptx` - lecture slides
## Workflow

1. **Data preparation:** load both datasets, split 80/20 with `random_state=42`, standardize features with `StandardScaler` (fit on the training set only), and convert to PyTorch tensors.
2. **Model building:**
   - `WineMLP`: 13 inputs, 16 hidden units (ReLU), 3 raw-logit outputs.
   - `DiabetesMLP`: 10 inputs, 32 hidden units (ReLU), 1 linear output.
3. **Training:** 50 epochs with mini-batches of 16 and shuffling. Epoch output is not printed.
4. **Evaluation:** switch to `model.eval()` and wrap inference in `torch.no_grad()`. Report accuracy for classification and MSE/MAE for regression, rounded to 2 decimals.
5. **Sample predictions:** spot-check one test sample per model (index 17 for wine, index 0 for diabetes).

## Results

| Metric | Value |
|---|---|
| Wine test accuracy | 1.00 |
| Wine sample 17 | predicted class 1, true class 1 (correct) |
| Diabetes test MSE | about 2816.73 |
| Diabetes test MAE | about 42.14 |
| Diabetes sample 0 | predicted about 164.39, true 219 |

Both seeds are fixed (`torch.manual_seed(42)`) so the results are reproducible.

## Interpretation

- **Wine alerting pipeline:** the classifier sorts grape varieties accurately on this small test set (36 samples). Samples with low or ambiguous predicted probability should still be flagged for a quality-control check.
- **Diabetes forecasting service:** the regressor captures the general trend but is imprecise for individual patients (sample 0 is under-predicted by about 55 points). Use predictions for early warnings and prioritisation, and leave a safety margin below any critical threshold.
