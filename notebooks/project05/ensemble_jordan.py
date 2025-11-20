"""Script for ensemble_jordan.ipynb."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

df = pd.read_csv("winequality-red.csv", sep=";")
df.info()
df.head()

def quality_to_label(q):  # noqa: D103
    if q <= 4:
        return "low"
    if q <= 6:
        return "medium"
    return "high"

def quality_to_number(q):  # noqa: D103
    if q <= 4:
        return 0
    if q <= 6:
        return 1
    return 2

df["quality_label"] = df["quality"].apply(quality_to_label)
df["quality_numeric"] = df["quality"].apply(quality_to_number)

X = df.drop(columns=["quality", "quality_label", "quality_numeric"])  # Features
y = df["quality_numeric"]  # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def evaluate_model(name, model, X_train, y_train, X_test, y_test, results):  # noqa: D103, N803
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average="weighted")
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")
    acc_gap = train_acc - test_acc
    f1_gap = train_f1 - test_f1

    results.append(
        {
            "Model": name,
            "Train Accuracy": train_acc,
            "Test Accuracy": test_acc,
            "Accuracy Gap": acc_gap,
            "Train F1": train_f1,
            "Test F1": test_f1,
            "F1 Gap": f1_gap
        }
    )

    results_text: str = f"""
    Accuracy

    Train: {train_acc:.4f}
    Test:  {test_acc:.4f}
    Gap:  {acc_gap:.4f}


    F1 Score

    Train:  {train_f1:.4f}
    Test:   {test_f1:.4f}
    Gap:   {f1_gap:.4f}
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6,4), gridspec_kw={"width_ratios": [1,3]})

    axes[0].text(0.05, 0.5, results_text,
                 ha="left", va="center")
    axes[0].axis("off")

    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, cmap="Blues", ax=axes[1],
                linewidths=.5, linecolor="royalblue", fmt="d")
    axes[1].patch.set_edgecolor("royalblue")
    axes[1].patch.set_linewidth(0.5)
    axes[1].set(xlabel="Predicted", ylabel="Actual", title="Confusion Matrix")
    axes[1].set_xticklabels(["Low", "Medium", "High"])
    axes[1].set_yticklabels(["Low", "Medium", "High"])

    fig.suptitle(f"Model: {name}")
    plt.tight_layout()
    plt.show()

results: list = []

# 4. AdaBoost (200, lr=0.5)
evaluate_model(
    "AdaBoost (200, lr=0.5)",
    AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=42),
    X_train,
    y_train,
    X_test,
    y_test,
    results,
)
# 5. Gradient Boosting
evaluate_model(
    "Gradient Boosting (100)",
    GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    ),
    X_train,
    y_train,
    X_test,
    y_test,
    results,
)

# Define 5-fold stratified cross-validator for tuning both our models
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gb_model = GradientBoostingClassifier(random_state=42)

# Define the grid of hyperparameters to search
param_grid = {
    "n_estimators": [100, 150, 200, 250],    # Various amount of trees
    "learning_rate": [0.05, 0.1, 0.15],      # Various learning rates
    "max_depth": [1, 2, 3],                  # Various tree depths
    "subsample": [0.7, 0.8, 0.9],            # Various forced subsampling
    "max_features": ["sqrt"]                 # Restrict number of features at each tree split
}

grid_search = GridSearchCV(
    estimator=gb_model,        # The model we are tuning
    param_grid=param_grid,     # The grid of parameters to test
    scoring="f1_weighted",     # The metric to optimize
    cv=skf,                    # Cross-validation strategy
    verbose=1,                 # Print progress updates during search
    n_jobs=-1                  # Use all available CPU cores
)

print("Starting Grid Search for Gradient Boosting...")
grid_search.fit(X_train, y_train)
print("Grid Search complete.")

print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best F1 Score (CV): {grid_search.best_score_:.4f}")

best_gb_model = grid_search.best_estimator_

# Calling evaluate_model here will append the results of the "best" model to our results list
evaluate_model(
    "Gradient Boosting (Tuned)",
    best_gb_model,
    X_train,
    y_train,
    X_test,
    y_test,
    results,
)

ab_model = AdaBoostClassifier(random_state=42)

ab_param_grid = {
    "n_estimators": [100, 150, 200, 250, 300],
    "learning_rate": np.arange(0.05, 2.0, 0.05).tolist()  # Learning rates from 0.05 to 2.0 by 0.05
}

ab_grid_search = GridSearchCV(
    estimator=ab_model,
    param_grid=ab_param_grid,
    scoring="accuracy",
    cv=skf,
    verbose=1,
    n_jobs=-1
)

print("Starting Grid Search for AdaBoost Classifier...")
ab_grid_search.fit(X_train, y_train)
print("Grid Search complete.")

print(f"Best parameters found: {ab_grid_search.best_params_}")
print(f"Best F1 Score (CV): {ab_grid_search.best_score_:.4f}")

best_ab_model = ab_grid_search.best_estimator_
ab_lr, ab_est = ab_grid_search.best_params_.values()

# Calling evaluate_model here will append the results of the "best" model to our results list
evaluate_model(
    f"AdaBoost ({ab_est}, lr={ab_lr})",
    best_ab_model,
    X_train,
    y_train,
    X_test,
    y_test,
    results,
)

# Define a dataframe for the results and sort on 'Test Accuracy' then 'Accuracy Gap'
results_df = pd.DataFrame(results).sort_values(by=["Test Accuracy", "Accuracy Gap"],
                                               ascending=[False, True])

print("\nSummary of All Models:")
# Round the results, reset index to denote top accuracy, and then display the dataframe
print(results_df.round(4).reset_index(drop=True))
