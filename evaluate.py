"""Honest test-set evaluation of the models explored in proj.ipynb.

The notebook compares several classifiers on the Telco customer-churn dataset,
but most of its printed "accuracies" are computed on the TRAINING set — which
rewards memorization (tree models score 99%+ there) and says nothing about
generalization. This script reproduces the notebook's preprocessing and
evaluates every sklearn model on a proper held-out test set, reporting
accuracy plus the metrics that matter for an imbalanced target (~27% churn):
churn-class precision/recall/F1 and ROC-AUC.

Run:  pip install -r requirements.txt && python evaluate.py
"""
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def load_data():
    """Same preprocessing as proj.ipynb."""
    df = pd.read_csv("dataset.csv")
    df = df.drop("customerID", axis="columns")
    df = df[df.TotalCharges != " "]
    df.TotalCharges = pd.to_numeric(df.TotalCharges)
    df = df.replace("No internet service", "No")
    df = df.replace("No phone service", "No")
    yes_no_cols = ["Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity",
                   "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                   "StreamingMovies", "PaperlessBilling", "Churn"]
    for col in yes_no_cols:
        df[col] = df[col].replace({"Yes": 1, "No": 0})
    df["gender"] = df["gender"].replace({"Female": 1, "Male": 0})
    df = pd.get_dummies(data=df, columns=["InternetService", "Contract", "PaymentMethod"])
    scaler = MinMaxScaler()
    df[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.fit_transform(
        df[["tenure", "MonthlyCharges", "TotalCharges"]])
    x = df.drop("Churn", axis="columns")
    y = df["Churn"].astype(int)
    return train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "SVM (RBF)": SVC(probability=True, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
}


def main():
    x_train, x_test, y_train, y_test = load_data()
    churn_rate = y_test.mean()
    print(f"train={len(x_train)}  test={len(x_test)}  churn rate in test={churn_rate:.1%}\n")
    header = f"{'Model':<22} {'TrainAcc':>8} {'TestAcc':>8} {'Precision':>9} {'Recall':>7} {'F1':>6} {'ROC-AUC':>8}"
    print(header)
    print("-" * len(header))
    rows = []
    for name, model in MODELS.items():
        model.fit(x_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(x_train))
        pred = model.predict(x_test)
        proba = model.predict_proba(x_test)[:, 1]
        row = {
            "name": name,
            "train_acc": train_acc,
            "test_acc": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "auc": roc_auc_score(y_test, proba),
        }
        rows.append(row)
        print(f"{name:<22} {row['train_acc']:>8.3f} {row['test_acc']:>8.3f} "
              f"{row['precision']:>9.3f} {row['recall']:>7.3f} {row['f1']:>6.3f} {row['auc']:>8.3f}")
    best = max(rows, key=lambda r: r["auc"])
    print(f"\nBest by ROC-AUC: {best['name']} ({best['auc']:.3f})")
    print("Note the train-vs-test gap on Decision Tree / Random Forest — that gap is what")
    print("the notebook's train-set 'accuracies' were hiding.")


if __name__ == "__main__":
    main()
