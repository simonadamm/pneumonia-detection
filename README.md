# Pneumonia Detection from Chest X-Ray Images

A deep learning portfolio project for binary classification of chest X-ray images into two classes:

- `NORMAL`
- `PNEUMONIA`

The goal of this project is not to build a production-ready medical diagnostic system, but to demonstrate a complete image classification workflow: image preprocessing, convolutional neural network training, class imbalance awareness, threshold tuning, and evaluation with metrics that are more meaningful than accuracy alone.

> **Disclaimer:** This project is for educational and portfolio purposes only. It is not a medical device and must not be used for clinical diagnosis.

---

## Project overview

This project uses a custom Convolutional Neural Network trained on grayscale chest X-ray images resized to `224 x 224` pixels. The model predicts whether an image belongs to the `NORMAL` or `PNEUMONIA` class.

Because this is a medical screening-style task, evaluation focuses especially on:

- recall for the `PNEUMONIA` class,
- false negatives,
- false positives,
- ROC AUC,
- confusion matrix analysis,
- threshold selection.

Accuracy alone is not sufficient for this type of problem.

---

## Repository structure

```text
pneumonia-detection/
├── data/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── test/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── val/
├── notebooks/
│   └── pneumonia-detection.ipynb
├── pyproject.toml
├── uv.lock
└── README.md
```

The original `data/val` folder is not used as the main validation set because it contains only a very small number of images. Instead, the validation set is created from the training data using a stratified train/validation split.

---

## Dataset split

The available dataset contains:

| Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Train folder | 1,341 | 3,875 | 5,216 |
| Test folder | 234 | 390 | 624 |

For model development, the training folder is split into:

- training set: 80%,
- validation set: 20%.

The split is stratified to preserve class proportions.

---

## Reproducibility

Neural network training is stochastic. Results can vary between runs because of:

- random weight initialization,
- dropout,
- data augmentation,
- non-deterministic TensorFlow operations,
- early stopping / learning rate scheduling behavior.

To reduce this variability, the final experiment fixes random seeds for Python, NumPy, TensorFlow, the train/validation split, and data augmentation layers.

Even with fixed seeds, small differences may still occur depending on hardware and TensorFlow backend behavior.

---

## Preprocessing

Each image is:

1. loaded from disk,
2. converted to grayscale,
3. resized to `224 x 224`,
4. normalized to the `[0, 1]` range,
5. reshaped to include a channel dimension: `(224, 224, 1)`.

The model uses light data augmentation:

- random rotation,
- random zoom,
- random translation,
- random contrast adjustment.

---

## Model architecture

The final model is a custom CNN with three convolutional blocks followed by a dense classification head.

Main components:

- `Conv2D` layers with ReLU activation,
- `BatchNormalization`,
- `MaxPooling2D`,
- `Dropout`,
- `Flatten`,
- dense layer,
- sigmoid output for binary classification.

A `GlobalAveragePooling2D` version was considered, but the `Flatten` version was kept for the final experiment because it performed better in this setup.

---

## Training setup

The final model is trained with:

- loss: `binary_crossentropy`,
- optimizer: `Adam`,
- metrics: accuracy, precision, recall, AUC,
- callbacks:
  - `EarlyStopping`,
  - `ReduceLROnPlateau`.

Class weights were computed and tested during experimentation, but the final reported run does **not** use `class_weight`. In this project, class weighting did not consistently improve the final test behavior and tended to increase the false-positive problem.

---

## Results

### Final test results

The model was evaluated in two ways:

1. using the default threshold `0.5`,
2. using a threshold selected on the validation set based on F1 score.

| Threshold | Accuracy | Precision<br>`PNEUMONIA` | Recall<br>`PNEUMONIA` | AUC | Interpretation |
|---:|---:|---:|---:|---:|---|
| 0.50 | 0.7163 | 0.6885 | 0.9974 | 0.9307 | Very sensitive, but many false positives |
| 0.70 | 0.78 | 0.74 | 0.99 | 0.9307 | Better operating point after threshold tuning |

At the default threshold, the model strongly favors the `PNEUMONIA` class. This produces very high pneumonia recall, but poor recognition of normal X-rays.

After selecting a threshold on the validation set, the test performance becomes more balanced, although the model still produces many false positives for the `NORMAL` class.

---

## Classification report at selected threshold

Using the validation-selected threshold `0.70`, the model achieves the following test-set performance:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| NORMAL | 0.98 | 0.42 | 0.59 | 234 |
| PNEUMONIA | 0.74 | 0.99 | 0.85 | 390 |
| Accuracy |  |  | 0.78 | 624 |
| Macro avg | 0.86 | 0.71 | 0.72 | 624 |
| Weighted avg | 0.83 | 0.78 | 0.75 | 624 |

The model detects almost all pneumonia cases, but it incorrectly classifies a significant number of normal X-rays as pneumonia.

---

## Threshold analysis

The default classification threshold is:

```python
y_pred = (y_pred_proba >= 0.5).astype(int)
```

In this experiment, the default threshold was not the best operating point. The validation-set threshold analysis selected:

```text
threshold = 0.70
```

This improved the test result compared with the default threshold.

The threshold analysis is an important part of the project because it shows that the model has useful class-separation ability, but its raw sigmoid outputs are not perfectly calibrated. A fixed threshold of `0.5` can therefore be misleading.

---

## Interpretation

The final model behaves like a high-sensitivity screening classifier:

- it detects nearly all pneumonia cases,
- it has a very low false-negative rate for `PNEUMONIA`,
- it has a high false-positive rate for `NORMAL`,
- it achieves a strong AUC, suggesting useful ranking ability,
- it requires threshold tuning to produce a more reasonable classification balance.

This means the model may be useful as a learning example for medical image classification and screening-oriented evaluation, but it is not suitable as a standalone diagnostic system.

---

## Why the results can look unusual

During training, accuracy can stay close to the pneumonia class proportion for many epochs while AUC continues improving. This happens because the model may initially predict most samples as `PNEUMONIA` at threshold `0.5`, while still learning to rank pneumonia images above normal images internally.

In other words:

- accuracy at one fixed threshold can look poor,
- AUC can still be strong,
- threshold tuning can reveal a better operating point.

This is why the project reports both the default-threshold result and the validation-selected threshold result.

---

## Key takeaways

- Accuracy alone is not enough for evaluating medical image classifiers.
- High recall for pneumonia can come at the cost of many false positives.
- AUC and threshold-based classification can tell different stories.
- The default threshold `0.5` is not always appropriate.
- Class weighting does not automatically improve performance.
- Neural network results can vary between runs, so reproducibility controls and careful interpretation are important.
- Confusion matrix and classification report analysis are essential.

---

## Limitations

This project has several important limitations:

- The model was evaluated on a limited dataset.
- The validation set is created from the training folder, while the test set comes from a separate folder and may have a different distribution.
- The model produces many false positives for the `NORMAL` class.
- The model has not been validated on an external dataset.
- No clinical metadata is used.
- The model's probability outputs are not calibrated.
- The project is not suitable for real-world medical use.

---

## Possible future improvements

Potential next steps include:

- testing transfer learning with EfficientNet, ResNet, or MobileNet,
- evaluating the model on an external dataset,
- adding Grad-CAM visualizations for interpretability,
- calibrating probabilities with Platt scaling or isotonic regression,
- running multiple seeds and reporting mean/std metrics,
- adding model checkpointing for reproducible best-model selection,
- improving specificity for the `NORMAL` class,
- performing detailed false-positive and false-negative image analysis.

---

## How to run

Install dependencies using `uv`:

```bash
uv sync
```

Then open and run the notebook:

```text
notebooks/pneumonia-detection.ipynb
```

Expected data layout:

```text
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## Technologies used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Pillow

---

## Summary

This project demonstrates a full CNN-based workflow for pneumonia detection from chest X-rays. The final model achieves very high pneumonia recall and strong AUC, but also shows a clear false-positive problem. The main value of the project is the evaluation process: understanding class imbalance, comparing thresholds, interpreting confusion matrices, and explaining why medical image classifiers should not be judged by accuracy alone.
