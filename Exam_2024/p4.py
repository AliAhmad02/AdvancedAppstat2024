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


train_path = "AdvancedAppstat/Exam_2024/Data_Files/Exam_2024_Prob4_TrainData.csv"
test_path = "AdvancedAppstat/Exam_2024/Data_Files/Exam_2024_Prob4_TestData.csv"
blind_path = "AdvancedAppstat/Exam_2024/Data_Files/Exam_2024_Prob4_BlindData.csv"

train_data = pd.read_csv(train_path, delimiter=",")
train_data = train_data.drop(columns=["ID"])
test_data = pd.read_csv(test_path, delimiter=",")
test_data = test_data.drop(columns=["ID"])
blind_data = pd.read_csv(blind_path, delimiter=",")
ids = blind_data["ID"].values
blind_data = blind_data.drop(columns=["ID"])

X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

objective = XGBObjective(X_train, X_test, y_train, y_test)
study = create_study(direction="maximize")
study.optimize(objective, n_trials=300)
best_trial = study.best_trial
optimal_params = best_trial.params

bst = XGBClassifier(**optimal_params)
bst.fit(X_train, y_train)
prob_train = bst.predict_proba(X_train)[:, 1]
prob_test = bst.predict_proba(X_test)[:, 1]
prob_test_null = prob_test[y_test == 0]
prob_test_one = prob_test[y_test == 1]
auc_train = roc_auc_score(y_train, prob_train)
auc_test = roc_auc_score(y_test, prob_test)
print(
    f"\nROC AUC training sample: {auc_train:.3f}\nROC AUC test sample: {auc_test:.3f}"
)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.hist(
    prob_test_null,
    color="red",
    bins=40,
    range=(0, 1),
    label="No-show=0",
    histtype="step",
)
ax.hist(
    prob_test_one,
    color="black",
    bins=40,
    range=(0, 1),
    label="No-show=1",
    histtype="step",
)
ax.set_xlabel("No-show probability", fontsize=15)
ax.set_ylabel("Frequency", fontsize=15)
ax.legend(fontsize=15, frameon=False)
plt.savefig("ml_classifier.png", dpi=500, bbox_inches="tight")
plt.show()

blind_pred = bst.predict(blind_data)
ids_blind_pred0 = ids[~blind_pred.astype(bool)]
ids_blind_pred1 = ids[blind_pred.astype(bool)]

np.savetxt(
    "AdvancedAppstat/Exam_2024/Data_Files/ahmad.AMAS_Exam.Problem4.NoShowTrue.txt",
    ids_blind_pred1.astype("int"),
    delimiter="\n",
    fmt="%.0f",
)
np.savetxt(
    "AdvancedAppstat/Exam_2024/Data_Files/ahmad.AMAS_Exam.Problem4.NoShowFalse.txt",
    ids_blind_pred0.astype("int"),
    delimiter="\n",
    fmt="%.0f",
)
