# Baseline Predictive Pipeline -- ETAI

This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Name 
Duarte Oliveira 20231587
week1
Logistic Regression we got a Train accuracy of 0.679 and a Test accuracy of 0.680 and with an accuracy of 0.68 . 
Decision tree has a higher train accurary of 0.829 and a test accuracy of 0.629 so we have a overfitting case, since the train is much higher than the test, the model is memorizing the train data, also it has a lower accuracy (0.63) than the Logistic Regression model.
So we can conclude that logistic regression is the better model for now.

--
## Week 2 — Data Diagnosis and Preprocessing

This week, we extended the pipeline with data quality checks.

### Additional consistency checks

I checked whether related columns agreed with each other:

- Found 6 rows where `age_cat` disagreed with a valid numeric age.
- Found 10 rows where `score_text` disagreed with `decile_score`.
- Corrected these categories using the numeric source values as the
  authoritative values.

The age check initially flagged 116 rows. Of these, 110 were caused by
the check's age-45 boundary assumption. We adjusted the check to match
the dataset's convention, leaving 6 inconsistencies.

A generic `apply_consistency_rules()` function now reads intervals,
labels, source columns, destination columns, and actions from the
configuration. It runs after validity checks and category cleanup,
before imputation and encoding.

### Model configuration


We compared last week's baseline with this week's pipeline after
adding data diagnosis and cleaning.

| Metric | Logistic regression: before | After | Decision tree: before | After |
|---|---:|---:|---:|---:|
| Training accuracy | 0.679 | 0.673 | 0.829 | 0.795 |
| Test accuracy | 0.680 | 0.673 | 0.629 | 0.648 |
| Reported train–test gap | -0.001 | 0.000 | 0.199 | 0.148 |
| Class 0 precision | 0.69 | 0.67 | 0.64 | 0.65 |
| Class 0 recall | 0.75 | 0.77 | 0.75 | 0.75 |
| Class 0 F1-score | 0.72 | 0.72 | 0.69 | 0.70 |
| Class 1 precision | 0.66 | 0.67 | 0.62 | 0.64 |
| Class 1 recall | 0.60 | 0.55 | 0.49 | 0.52 |
| Class 1 F1-score | 0.63 | 0.61 | 0.55 | 0.58 |
| Test samples | 1,252 | 1,237 | 1,252 | 1,237 |

Logistic regression's test accuracy decreased slightly, from 68.0%
to 67.3% . Class 1 recall also decreased,
from 0.60 to 0.55, while class 0 F1-score remained unchanged.

The decision tree's test accuracy increased from 62.9% to 64.8%
. Its training accuracy decreased and its
reported train–test gap narrowed from 0.199 to 0.148, suggesting
less overfitting. Class 1 F1-score improved from 0.55 to 0.58.

Logistic regression still achieved the higher test accuracy after
cleaning: 67.3%, compared with 64.8% for the decision tree.

Cleaning improved data consistency, but did not improve every
performance metric. A controlled comparison would use the same
held-out records and model settings for both versions.