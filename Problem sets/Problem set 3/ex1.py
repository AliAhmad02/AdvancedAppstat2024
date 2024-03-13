import pandas as pd
import numpy as np

from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from optuna import Trial, create_study
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scienceplots

plt.style.use("science")


class XGBObjective:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def __call__(self, trial: Trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0, 1),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 20),
        }
        bst = XGBClassifier(**params)
        bst.fit(self.X_train, self.y_train)
        bst_pred = bst.predict(self.X_test)
        score = accuracy_score(self.y_test, bst_pred)
        return score


train_test_path = "AdvancedAppstat/Problem sets/Problem set 3/data_files/earning_potential_train_test.txt"
real_path = (
    "AdvancedAppstat/Problem sets/Problem set 3/data_files/earning_potential_real.txt"
)

train_test_df = pd.read_csv(train_test_path, delimiter=",")
train_test_df["earning"] = train_test_df["earning"].replace({" <=50K": 0, " >50K": 1})
real_df = pd.read_csv(real_path, delimiter=",")

X_train, X_test, y_train, y_test = train_test_split(
    train_test_df.iloc[:, :-1],
    train_test_df.iloc[:, -1],
    test_size=0.5,
    random_state=42,
)

objective = XGBObjective(X_train, X_test, y_train, y_test)
study = create_study(direction="maximize")
study.optimize(objective, n_trials=200)
best_trial = study.best_trial
optimal_params = best_trial.params

bst = XGBClassifier(**optimal_params)
bst.fit(X_train, y_train)
prob_train = bst.predict_proba(X_train)[:, 1]
prob_test = bst.predict_proba(X_test)[:, 1]
auc_train = roc_auc_score(y_train, prob_train)
auc_test = roc_auc_score(y_test, prob_test)
print(
    f"\nROC AUC training sample: {auc_train:.3f}\nROC AUC test sample: {auc_test:.3f}"
)


def get_frac_high_earners_threshold(y_true, prob, threshold):
    classified = prob.copy()
    mask = classified >= threshold
    classified[mask] = 1
    classified[~mask] = 0
    return ((y_true == 1) & (classified == 1)).sum() / (classified == 1).sum()


def get_frac_low_earners_threshold(y_true, prob, threshold):
    classified = prob.copy()
    mask = classified >= threshold
    classified[mask] = 1
    classified[~mask] = 0
    return ((y_true == 0) & (classified == 1)).sum() / (classified == 1).sum()


prob_thresholds = np.linspace(0.3, 0.8, 10_000)
high_earner_fracs = np.array(
    [
        get_frac_high_earners_threshold(y_test, prob_test, threshold)
        for threshold in prob_thresholds
    ]
)
low_earner_fracs = np.array(
    [
        get_frac_low_earners_threshold(y_test, prob_test, threshold)
        for threshold in prob_thresholds
    ]
)
prob_cutoff_val = prob_thresholds[
    (high_earner_fracs >= 0.85) & (low_earner_fracs <= 0.15)
][0]
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(
    prob_thresholds,
    high_earner_fracs * 100,
    label="High earners",
    color="black",
)
ax.plot(
    prob_thresholds,
    low_earner_fracs * 100,
    label="Low earners",
    color="red",
)
ax.axvline(
    prob_cutoff_val,
    color="black",
    linestyle="dashed",
    label=f"$P_{{cutoff}}=${prob_cutoff_val:.3f}",
    lw=1.5,
)
ax.set_xlabel("Probability cutoff value", fontsize=15)
ax.set_ylabel(r"\% of positively classified sample", fontsize=15)
ax.legend(fontsize=15, frameon=False)
plt.show()

pred_from_cutoff = np.zeros(len(prob_test))
pred_from_cutoff[prob_test >= prob_cutoff_val] = 1
true_positives = ((pred_from_cutoff == 1) & (y_test == 1)).sum()
predicted_positive = (pred_from_cutoff == 1).sum()
precision = true_positives / predicted_positive
print(f"\nPrecision: {precision:.3f}")

probs_high_earners = prob_test[y_test == 1]
probs_low_earners = prob_test[y_test == 0]
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.hist(
    probs_high_earners[:500],
    bins=40,
    color="black",
    histtype="step",
    label="High earners",
)
ax.hist(
    probs_low_earners[:500], bins=40, color="red", histtype="step", label="Low earners"
)
ax.set_xlabel("Probability", fontsize=15)
ax.set_ylabel("Frequency", fontsize=15)
ax.legend(fontsize=15, frameon=False)
plt.show()

plot_importance(bst, grid=False)
plt.show()

real_prob = bst.predict_proba(real_df.iloc[:, 1:])[:, 1]
high_earner_ids_real = np.where(real_prob >= prob_cutoff_val)[0]
low_earner_ids_real = np.where(real_prob < prob_cutoff_val)[0]
np.savetxt(
    "AdvancedAppstat/Problem sets/Problem set 3/data_files/ali_ahmad_low_ID.txt",
    low_earner_ids_real.astype("int"),
    delimiter="\n",
    fmt="%.0f",
)
np.savetxt(
    "AdvancedAppstat/Problem sets/Problem set 3/data_files/ali_ahmad_high_ID.txt",
    high_earner_ids_real.astype("int"),
    delimiter="\n",
    fmt="%.0f",
)
