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
## Week 2 Data Diagnosis and Preprocessing

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


## Week 3 Preprocessing Pipeline and Cross-Validation

This week, we replaced the single train test evaluation with
cross-validation on a development set, keeping a separate test set
for the final assessment.

### Evaluation method

I used five-fold stratified cross-validation.
Stratification approximately preserves the class proportions in each
fold.

After cross-validation, each model's pipeline was refitted on the
entire development set.

### Model comparison

Both models used random_state=42. The development set contained

| Metric | Logistic regression | Decision tree |
|---|---:|---:|
| Mean training accuracy | 0.677 | 0.700 |
| Mean validation accuracy | 0.675 | 0.599 |
| Validation accuracy standard deviation | 0.013 | 0.023 |
| Mean train–validation gap | 0.002 | 0.101 |
| Class 0 precision  | 0.67 | 0.62 |
| Class 0 recall  | 0.81 | 0.69 |
| Class 0 F1-score  | 0.73 | 0.66 |
| Class 1 precision | 0.69 | 0.56 |
| Class 1 recall  | 0.51 | 0.49 |
| Class 1 F1-score | 0.59 | 0.52 |

Logistic regression achieved a mean validation accuracy of 67.5%,
compared with 59.9% for the decision tree difference of
7.6 percentage points.

Its validation accuracy also varied less across folds, and its
mean train–validation gap was only 0.002. The decision tree's
larger gap of 0.101 suggests greater overfitting under this configuration.

Both models identified class 0 more successfully than class 1.
Logistic regression achieved class 1 recall of 0.51, while the
decision tree achieved 0.49, meaning both missed approximately
half of the actual positive cases.

### Comparison with last week

Last week's post-cleaning test accuracies were 67.3% for logistic
regression and 64.8% for the decision tree. This week's mean
cross-validation accuracies were 67.5% and 59.9%, respectively.

Logistic regression performed better in this development-set comparison.



## Hyperparameter Tuning  Decision Tree and Random Forest

Both models used the same preprocessing and development data, with **five outer folds, 30 Optuna trials and five inner folds**.

| Model | Baseline CV accuracy | Tuned nested CV accuracy | Improvement |
|---|---:|---:|---:|
| Decision Tree | 60.9% | 67.5% | +6.6 percentage points |
| Random Forest | 64.5% | 68.0% | +3.5 percentage points |

Tuning improved both models and reduced their mean training–validation gaps to below **one percentage point**.

The final selected parameters were:

- **Decision Tree:** depth 5, minimum leaf size 3, entropy criterion.
- **Random Forest:** 200 trees, depth 7, minimum leaf size 3, `max_features="sqrt"`.

Random Forest achieved slightly higher accuracy, while Decision Tree offered a simpler model. Both retained an approximately **14-percentage-point FPR gap between African-American and Caucasian groups**.

Both final models were refitted on **5,771 development rows**. The **1,443 test rows remained locked and unevaluated**.


## Ensemble Comparison  AdaBoost and Oblique Random Forest

We compared both models using the same preprocessing and development data, with **five outer folds, 30 Optuna trials per study and five inner folds**.

| Model | Baseline CV accuracy | Tuned nested CV accuracy | Improvement |
|---|---:|---:|---:|
| AdaBoost | 67.6% | 67.9% | +0.3 percentage points |
| Oblique Random Forest | 63.9% | 68.2% | +4.3 percentage points |

Tuning improved both models, particularly Oblique Random Forest. After tuning, both had a mean training–validation gap of approximately **0.8 percentage points**.

The corrected paired t-test returned **t = −0.531** and **p = 0.624**, with AdaBoost winning **2 of 5 folds**. Oblique Random Forest averaged **0.26 percentage points higher accuracy**, but the difference was **not statistically significant at the 5% level**. These results do not establish a clear performance winner.

The fairness audit still showed an approximately **13-percentage-point FPR gap between African-American and Caucasian groups** for both models.

Both final models were refitted on **5,771 development rows**. The **1,443 test rows remained locked and unevaluated**.


### week 8

## Model Comparison and Probability Evaluation

We compared six models using the same development data, **target encoding and robust scaling**, with **five outer folds, 30 Optuna trials per study and five inner folds**. All comparisons used stratified folds with **random state 42**.

**ROC-AUC was the selection metric.** Tuned results below are outer-fold means ± standard deviations.

| Model | Baseline CV ROC-AUC | Tuned nested CV ROC-AUC | Improvement |
|---|---:|---:|---:|
| Histogram Gradient Boosting | 0.717 | **0.729 ± 0.015** | +0.012 |
| AdaBoost | 0.723 | 0.727 ± 0.015 | +0.004 |
| Oblique Random Forest | 0.685 | 0.726 ± 0.015 | +0.041 |
| Random Forest | 0.682 | 0.724 ± 0.015 | +0.042 |
| Decision Tree | 0.605 | 0.717 ± 0.014 | +0.112 |
| Gaussian Naive Bayes | 0.685 | 0.706 ± 0.012 | +0.021 |

Tuning improved ROC-AUC for every model, particularly the Decision Tree. All models exceeded the **dummy-model ROC-AUC floor of 0.500**.

Histogram Gradient Boosting achieved the highest mean ROC-AUC and **average precision of 0.694**. Its accuracy was **67.9%**, positive-class F1 was **0.641**, and Recall was **63.5%**. AdaBoost achieved the highest accuracy at **68.4%**. The small ROC-AUC differences between the leading models do not establish a statistically significant winner without a paired comparison of these updated runs.

Evaluation used **out-of-fold probabilities** for ROC-AUC and average precision. A **0.5 probability threshold** produced the labels used for classification and fairness reporting. Histogram Gradient Boosting’s mean training–validation ROC-AUC gap decreased from **0.058 to 0.010**, indicating reduced overfitting after tuning.

The fairness audit still showed unequal false-positive rates among people who did not reoffend. For Histogram Gradient Boosting, the FPR was **36% for African-American individuals and 22% for Caucasian individuals**, a **14-percentage-point gap**. AdaBoost had rates of **29% and 16%**, respectively, a **13-percentage-point gap**. Therefore, improved predictive performance did not eliminate the observed racial disparities.

All final models were refitted on **5,771 development rows**. The **1,443 test rows remained locked and unevaluated**.


# Week 9

## XGBoost and Pipeline Feature Selection

We evaluated **eight models**, adding XGBoost and standard Gradient Boosting to the previous comparison. All runs used **target encoding, robust scaling and stratified folds with random state 42**. Tuning used **five outer folds, 30 Optuna trials per study and five inner folds**, with **ROC-AUC as the selection metric**. Tuned results below are outer-fold means ± standard deviations.

A learned **VoteSelector** step was placed between preprocessing and the classifier and fitted within each fold’s training data. Its `min_votes` threshold, ranging from **1 to 3**, was tuned jointly with the classifier parameters. **This week’s baseline already used feature selection with `min_votes=2`**, so the improvements below measure the combined tuning procedure.

Pipeline caching reused preprocessing and selection fits when the training data and settings were identical. Temporary caches were removed after each run, and evaluation outputs were saved before report generation.

| Model | Baseline CV ROC-AUC | Tuned nested CV ROC-AUC | Improvement | Final min_votes | Features kept |
|---|---:|---:|---:|---:|---:|
| Oblique Random Forest | 0.655 | **0.728 ± 0.014** | +0.073 | 1 | 10/10 |
| Histogram Gradient Boosting | 0.708 | 0.727 ± 0.015 | +0.019 | 1 | 10/10 |
| XGBoost | 0.721 | 0.726 ± 0.014 | +0.005 | 1 | 10/10 |
| Random Forest | 0.656 | 0.726 ± 0.015 | +0.070 | 1 | 10/10 |
| Gradient Boosting | 0.724 | 0.726 ± 0.013 | +0.002 | 1 | 10/10 |
| AdaBoost | 0.721 | 0.723 ± 0.015 | +0.002 | 1 | 10/10 |
| Decision Tree | 0.639 | 0.717 ± 0.013 | +0.078 | 3 | 2/10 |
| Gaussian Naive Bayes | 0.703 | 0.708 ± 0.013 | +0.005 | 2 | 2/10 |

Tuning increased ROC-AUC for every model, particularly the Decision Tree, Oblique Random Forest and Random Forest. All models exceeded the **dummy-model ROC-AUC floor of 0.500**. Oblique Random Forest achieved this week’s highest mean ROC-AUC, but the small differences between the leading models do not establish a statistically significant winner.

Feature selection reduced the final **Decision Tree and Gaussian Naive Bayes** models to **age and prior convictions (`priors_count`)**. The other six final models retained **all 10 features**. These counts describe the final refits; selected subsets could differ across individual folds.

Evaluation used **out-of-fold probabilities** for ROC-AUC and average precision. A fixed **0.5 threshold** generated labels for classification and fairness reporting, with **class 1 representing reoffending**. Oblique Random Forest achieved the highest mean accuracy (**68.3%**), while XGBoost achieved the highest average precision (**0.693**). Random Forest achieved the highest positive-class F1 (**0.637**) and recall (**63.7%**). Gaussian Naive Bayes achieved the highest precision (**0.734**) but much lower recall (**28.6%**).

The mean training–validation ROC-AUC gap decreased from **0.094 to 0.017** for the Decision Tree and from **0.026 to 0.013** for XGBoost, consistent with reduced overfitting. AdaBoost’s gap increased from **0.005 to 0.017**, so tuning did not reduce this gap for every model.

Compared with last week’s tuned results, the changes were small:

| Model | Week 8 tuned ROC-AUC | Week 9 tuned ROC-AUC | Change |
|---|---:|---:|---:|
| Histogram Gradient Boosting | 0.729 | 0.727 | −0.002 |
| AdaBoost | 0.727 | 0.723 | −0.004 |
| Oblique Random Forest | 0.726 | 0.728 | +0.002 |
| Random Forest | 0.724 | 0.726 | +0.002 |
| Decision Tree | 0.717 | 0.717 | 0.000 |
| Gaussian Naive Bayes | 0.706 | 0.708 | +0.002 |

Last week’s Histogram Gradient Boosting result, **0.729 ± 0.015**, remains the highest mean in the saved scoreboard. **The older runs did not record their feature-selection setting**, so this comparison does not isolate the effect of feature selection. A controlled comparison with selection enabled and disabled would be needed to assess that effect.

The fairness audit continued to show unequal false-positive rates among people who did not reoffend. For **Oblique Random Forest**, rates were approximately **30% for African-American individuals and 15% for Caucasian individuals**, a **15-percentage-point gap**. XGBoost’s rates were **29% and 17%**, a **12-point gap**. Gaussian Naive Bayes had a smaller gap (**13% versus 5%**), alongside its low recall. Lower false-positive rates alone therefore do not establish better overall performance or fairness. Results for the smallest groups also require caution because their sample sizes were small.

All final models were refitted on **5,771 development rows**. The **1,443 test rows remained locked and unevaluated**.

# week 10


## Beyond COMPAS

For Option A, we adapted the binary classification pipeline to predict whether a person earns more than $50,000 a year (`income_over_50k = 1`). The course training file contains 6,000 labeled rows, with about 24% in the positive class. The separate file `adult_new.csv` contains 2,000 people to score.


### What broke and how we fixed it

| Issue | Cause | Fix |
| --- | --- | --- |
| The entry point always loaded the COMPAS config | `main.py` called `load_config()` with its default path | Added an optional positional config argument. The default command uses `config.yaml`. |
| The first cleaning check raised `KeyError: 'workclass'` | The copied YAML still pointed to the 7,286-row COMPAS file | Set the Adult CSV path, target, diagnostics, and feature lists in `config_adult.yaml`. |
| Tuning failed for Logistic Regression | The config had no `logistic_regression` search space | Ran Logistic Regression as an untuned baseline and enabled tuning for models with configured search spaces. |
| The Adult fairness report raised `KeyError: 'score_text'` | It always built a comparison from COMPAS's own score | Made the comparison conditional on `score_text` being present and grouped the model's false positive rates by the sensitive attribute from the config. |

### Preprocessing

- **Features and audit fields.** We excluded `id` and the census sampling weight `fnlwgt`. We kept `education_num` as the ordered representation of education and removed the duplicate text field `education`. We used `sex` for the fairness audit and excluded both `sex` and `race` from the model inputs. All 25 anonymized variables entered the pipeline as candidate features, with `v01`–`v18` treated as numeric and `v19`–`v25` as categorical.
- **Missing values.** The cleaning check preserved all 6,000 rows and converted every literal `?` in `workclass`, `occupation`, and `native_country` to missing. Before imputation, their missing counts were 379, 381, and 86, respectively. We used median imputation for numeric gaps, including `v01`–`v06`, and most-frequent imputation for categorical gaps. We added binary missingness flags for `workclass` and `occupation` because the course diagnosis found a lower positive rate among rows missing employment information.
- **Validity rules.** The Adult config accepts ages 17–90, weekly hours 1–99, and nonnegative capital gains and losses. This allows 17-year-olds and preserves `capital_gain = 99999`, the census code for gains of $99,999 or more.
- **Encoding and scaling.** Target encoding represents categories with estimates of the positive-class rate and keeps `native_country` compact despite its many rare levels. Robust scaling uses the median and interquartile range for numeric features. The pipeline fits these steps and imputation on the training data within each fold.
- **Feature selection.** After preprocessing, `VoteSelector` combines votes from Random Forest, Gradient Boosting, and L1-regularized Logistic Regression. The baseline used a two-vote minimum, and tuning could adjust that minimum.

### Model comparison

We used 4,800 development rows with five shuffled stratified folds (seed 42). The 1,200-row test set stayed locked. For each tuned model, Optuna used 30 trials with five inner folds to select by ROC-AUC. The table reports outer-fold scores for the tuned models and standard five-fold scores for Logistic Regression. One run must reach **ROC-AUC ≥ 0.900, accuracy ≥ 0.855, and class 1 F1 ≥ 0.660**. Accuracy and F1 use a 0.5 classification threshold. The dummy ROC-AUC was 0.500.

| Model | Baseline ROC-AUC | Reported ROC-AUC (mean ± SD) | Accuracy | Class 1 F1 | All three met |
| --- | ---: | ---: | ---: | ---: | :---: |
| **Gradient Boosting** | 0.912 | **0.911 ± 0.015** | **0.859** | **0.673** | **Yes** |
| XGBoost | 0.914 | 0.910 ± 0.015 | 0.858 | 0.665 | Yes |
| Histogram Gradient Boosting | 0.905 | 0.910 ± 0.016 | 0.820 | 0.689 | No |
| Oblique Random Forest | 0.896 | 0.908 ± 0.014 | 0.855 | 0.624 | No |
| AdaBoost | 0.903 | 0.906 ± 0.016 | 0.857 | 0.664 | Yes |
| Random Forest | 0.894 | 0.904 ± 0.015 | 0.817 | 0.682 | No |
| Logistic Regression (untuned) | 0.896 | 0.896 ± 0.014 | 0.799 | 0.664 | No |
| Naive Bayes | 0.878 | 0.886 ± 0.013 | 0.829 | 0.548 | No |
| Decision Tree | 0.721 | 0.883 ± 0.017 | 0.787 | 0.656 | No |

- **Boosted models.** Gradient Boosting led the tuned runs on mean ROC-AUC. Baseline XGBoost was strongest before tuning, with 0.914 ROC-AUC, 0.864 accuracy, and 0.685 F1. Its nested estimates were lower on all three metrics and still met the targets. AdaBoost moved from 0.849 accuracy and 0.651 F1 at baseline to 0.857 and 0.664 after tuning, clearing both cutoffs.
- **Precision and recall.** Histogram Gradient Boosting had the highest reported F1 (0.689), with 0.832 recall and 0.588 precision. Its accuracy, like Random Forest's, fell short of the target. Oblique Random Forest had 0.827 precision and 0.507 recall, which left F1 at 0.624. Class-weighted Logistic Regression reached 0.830 recall, with 0.554 precision and 0.799 accuracy.
- **Decision Tree and Naive Bayes.** The single tree gained 0.162 ROC-AUC points after tuning, the largest gain, and its mean train–validation gap fell from 0.144 to 0.027. Accuracy remained below target. Tuned Naive Bayes recovered 43.8% of positives, which limited its F1 to 0.548.

Using the five matched outer folds, corrected paired ROC-AUC tests gave approximate two-sided p-values of **0.51** for Gradient Boosting versus XGBoost and **0.70** versus Histogram Gradient Boosting. These calculations use the fold scores rounded to three decimals in the run reports, and the results leave the ordering of the three models uncertain.

### Selected model and retained features

We selected **Gradient Boosting** (run `20261114_194014_809087`) because it had the highest reported nested ROC-AUC among the runs meeting all three targets. The final fit used all 4,800 development rows and kept **8 of 37** preprocessed inputs with a two-vote minimum. Optuna selected 290 estimators, a maximum depth of 2, a minimum of 2 samples per leaf, a learning rate of about 0.066, and a subsample fraction of about 0.793.

| Type | Retained features |
| --- | --- |
| Numeric | `age`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week` |
| Categorical | `marital_status`, `occupation`, `relationship` |

The final selector screened out all 25 anonymized variables and both missingness flags. Eight of the nine model refits kept the same eight inputs. Naive Bayes used a three-vote minimum and also screened out `capital_loss`.

In the selected run, the out-of-fold false positive rate at 0.5 was **2%** for the Female group (1,390 actual class 0 rows) and **9%** for the Male group (2,261 actual class 0 rows), a gap of about seven percentage points.

### Reflection

The hardest assumption to find was in the fairness report because cross-validation had already saved its scores when the lookup for `score_text` failed. Using the same preprocessing and folds across estimators let us compare which configurations met the targets at the assignment's 0.5 threshold. Gradient Boosting reached 0.757 precision and 0.607 recall, which gave it a class 1 F1 of 0.673 alongside accuracy above the cutoff.

### Prediction export and final checks

After the selected model's refit, apply the same Adult cleaning and fitted preprocessing to `data/adult_new.csv`. Keep every input row and its original `id`. The export `adult_predictions.csv` should have 2,000 rows in input order and three columns (`id`, `probability`, `prediction`), with class 1 assigned when its probability is at least 0.5.

The prediction export and a full COMPAS rerun are pending verification. Also check that the next Adult run writes its report and scoreboard under `results_adult/`.
