# Telco Customer Churn — Model Comparison, Evaluated Honestly

Predicting customer churn on the Telco Customer Churn dataset (7,043 customers) — and a case study in why **training-set accuracy is a trap**.

The original notebook (`proj.ipynb`) compares seven classifiers but prints mostly *training* accuracy, where tree models look nearly perfect. `evaluate.py` re-runs the same preprocessing and scores every sklearn model on a proper held-out test set. The gap is the lesson:

## Honest results (20% held-out test set, n=1,407, 26.6% churn)

| Model | Train acc | **Test acc** | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.804 | **0.803** | 0.647 | 0.570 | 0.606 | 0.835 |
| Gradient Boosting | 0.826 | 0.793 | 0.628 | 0.545 | 0.584 | **0.840** |
| SVM (RBF) | 0.823 | 0.788 | 0.622 | 0.519 | 0.566 | 0.799 |
| Random Forest | *0.999* | 0.787 | 0.630 | 0.487 | 0.549 | 0.818 |
| KNN | 0.835 | 0.757 | 0.540 | 0.580 | 0.559 | 0.777 |
| Decision Tree | *0.999* | 0.721 | 0.475 | 0.484 | 0.479 | 0.645 |

(The notebook's Keras neural nets are evaluated on the test split inside the notebook itself: ~78% test accuracy.)

## What this shows

- **The 99%+ figures were memorization.** Decision Tree and Random Forest hit 0.999 on data they trained on, then drop to 0.72–0.79 on unseen customers — a 21–28 point fall.
- **The simplest model nearly wins.** Plain logistic regression posts the best test accuracy and F1, essentially tying gradient boosting on ROC-AUC. On a tabular problem with ~7K rows, model complexity bought almost nothing.
- **Accuracy alone flatters everyone.** With 26.6% churn, the churn-class recall (0.48–0.58 across models) is the number a retention team would actually feel — roughly every other churner goes unflagged.

## What's here

```
proj.ipynb     original walkthrough: cleaning → encoding → scaling → EDA → 7 models
evaluate.py    reproducible honest evaluation (source of the table above)
dataset.csv    Telco Customer Churn data
requirements.txt
```

## Run

```
pip install -r requirements.txt
python evaluate.py          # the honest comparison
jupyter notebook proj.ipynb # the full walkthrough (needs tensorflow for the NN cells)
```

## Tech stack

Python · pandas · scikit-learn · TensorFlow/Keras (notebook only) · matplotlib · seaborn
