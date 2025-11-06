"""Script for ml03_jordan.ipynb."""
from imblearn.over_sampling import SMOTE
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

from stats_jordan import classification_table

titanic_df: pd.DataFrame = sns.load_dataset("titanic")
titanic_df.head(10)

median_age = titanic_df["age"].median()
titanic_df["age"] = titanic_df["age"].fillna(median_age)

titanic_df["family_size"] = titanic_df["sibsp"] + titanic_df["parch"] + 1

titanic_df["alone"] = titanic_df["alone"].astype(int)

# Case 1: Feature = 'alone'
X1 = titanic_df[["alone"]]
y1 = titanic_df["survived"]

# Case 2: Feature = 'age'
X2 = titanic_df[["age"]].dropna()
y2 = titanic_df.loc[X2.index, "survived"]

# Case 3: Features = 'age' + 'family_size'
X3 = titanic_df[["age", "family_size"]].dropna()
y3 = titanic_df.loc[X3.index, "survived"]

# Case 1: Feature = 'alone'
splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123)

for train_idx1, test_idx1 in splitter1.split(X1, y1):
    X1_train = X1.iloc[train_idx1]
    X1_test  = X1.iloc[test_idx1]
    y1_train = y1.iloc[train_idx1]
    y1_test  = y1.iloc[test_idx1]

print("Case 1 - 'alone':")
print("Train size:", len(X1_train), "| Test size:", len(X1_test))

# Case 2: Feature = 'age'
splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123)

for train_idx2, test_idx2 in splitter2.split(X2, y2):
    X2_train = X2.iloc[train_idx2]
    X2_test  = X2.iloc[test_idx2]
    y2_train = y2.iloc[train_idx2]
    y2_test  = y2.iloc[test_idx2]

print("\nCase 2 - 'age':")
print("Train size:", len(X2_train), "| Test size:", len(X2_test))

# Case 3: Features = 'age' + 'family_size'
splitter3 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123)

for train_idx3, test_idx3 in splitter3.split(X3, y3):
    X3_train = X3.iloc[train_idx3]
    X3_test  = X3.iloc[test_idx3]
    y3_train = y3.iloc[train_idx3]
    y3_test  = y3.iloc[test_idx3]

print("\nCase 3 - 'age' + 'family_size':")
print("Train size:", len(X3_train), "| Test size:", len(X3_test))

# CASE 1: Decision Tree using 'alone'
tree_model1 = DecisionTreeClassifier()
tree_model1.fit(X1_train, y1_train)

# CASE 2: Decision Tree using 'age'
tree_model2 = DecisionTreeClassifier()
tree_model2.fit(X2_train, y2_train)

# CASE 3: Decision Tree using 'age' and 'family_size'
tree_model3 = DecisionTreeClassifier()
tree_model3.fit(X3_train, y3_train)

# Case 1: Feature = 'alone'
y1_pred = tree_model1.predict(X1_train)
y1_test_pred = tree_model1.predict(X1_test)

# Case 2: Feature = 'age'
y2_pred = tree_model2.predict(X2_train)
y2_test_pred = tree_model2.predict(X2_test)

# Case 3: Features = 'age' + 'family_size'
y3_pred = tree_model3.predict(X3_train)
y3_test_pred = tree_model3.predict(X3_test)

tree_dict1: dict = {
    "Case 1 - Alone": ((y1_train, y1_pred), (y1_test, y1_test_pred)),
    "Case 2 - Age": ((y2_train, y2_pred), (y2_test, y2_test_pred)),
    "Case 3 - Age + Family Size": ((y3_train, y3_pred), (y3_test, y3_test_pred))
}
flat_tree_dict1 = ((case, y_t, y_p) for case, data in tree_dict1.items() for y_t, y_p in data)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,7.5))

for idx, (ax, (case, y_t, y_p)) in enumerate(zip(axes.flatten(), flat_tree_dict1, strict=True)):

    classification_table(y_t, y_p, ax)
    ax.set_title(f"{"Training" if idx % 2 == 0  else "Test"} data ({case})")

fig.suptitle("Results for Decision Tree\n(Metrics in %)", size=14)
plt.tight_layout()
plt.show()
fig.savefig("tree_classification_all_cases.png")

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8,3.5), gridspec_kw={"width_ratios": [1, 1, 1.25]})

for ax, (case, (_, (y_test, y_test_pred))) in zip(axes, tree_dict1.items(), strict=True):
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, cmap="Blues", ax=ax, cbar=ax is axes[-1],
                linewidths=.5, linecolor="royalblue")
    ax.patch.set_edgecolor("royalblue")
    ax.patch.set_linewidth(0.5)
    ax.set(title=f"{case}")

for ax in axes[1:]:
    ax.set( yticks=[], ylabel="")

fig.suptitle("Confusion Matrix for Decision Tree", size=14)
fig.supxlabel("Predicted")
fig.supylabel("Actual")
plt.tight_layout()
plt.show()

tree_dict2: dict = {
    "Case 1 - Alone": (tree_model1, X1),
    "Case 2 - Age": (tree_model2, X2),
    "Case 3 - Age + Family Size": (tree_model3, X3)
}
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(100,90))

for ax, (case, (model, feature)) in zip(axes, tree_dict2.items(), strict=True):
    plot_tree(model, feature_names=feature.columns.tolist(), ax=ax,
              class_names=["Not Survived", "Survived"], filled=True)
    ax.set_title(f"Decision Tree ({case})", size=70)

plt.tight_layout()
plt.show()
fig.savefig("tree_plot_all_cases.png")

# CASE 1: SVC using 'alone'
svc_dict1: dict = {
    "RBF": {"kernel": "rbf"},
    "Linear": {"kernel": "linear"},
    "Polynomial": {"kernel": "poly", "degree": 3},
    "Sigmoid": {"kernel": "sigmoid"}
}
# CASE 2: SVC using 'age'
svc_dict2: dict = {
    "RBF": {"kernel": "rbf"},
    "Linear": {"kernel": "linear", "class_weight": "balanced"},
    "Polynomial": {"kernel": "poly", "degree": 3, "class_weight": "balanced"},
    "Sigmoid": {"kernel": "sigmoid"}
}
# CASE 3: SVC using 'age' + 'family_size'
svc_dict3: dict = {
    "RBF": {"kernel": "rbf"},
    "Linear": {"kernel": "linear", "C": 0.01, "class_weight": "balanced"},
    "Polynomial": {"kernel": "poly", "degree": 3, "C": 521.4, "class_weight": "balanced"},
    "Sigmoid": {"kernel": "sigmoid"}
}
case_train_test_svc: dict = {
    "Case 1 - Alone": ((X1_train, X1_test, y1_train, y1_test), svc_dict1),
    "Case 2 - Age": ((X2_train, X2_test, y2_train, y2_test), svc_dict2),
    "Case 3 - Age + Family Size": ((X3_train, X3_test, y3_train, y3_test), svc_dict3)
}
svc_model_dict: dict = {}
fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(10,15))
axes, idx = axes.flatten(), 0

for case, ((X_train, X_test, y_train, y_test), svc_dict) in case_train_test_svc.items():

    svc_model_dict[case] = {}

    for kernel, parameters in svc_dict.items():

        svc_model = SVC(**parameters)
        svc_model.fit(X_train, y_train)
        svc_model_dict[case][kernel] = svc_model
        y_svc_pred = svc_model.predict(X_test)
        classification_table(y_test, y_svc_pred, axes[idx])
        axes[idx].set_title(f"{kernel} Kernel ({case})")
        idx += 1

fig.suptitle("Results for SVC on Test Data\n(Metrics in %)\n", size=14)
plt.tight_layout()
plt.show()
fig.savefig("svc_classification_all_cases.png")

scaler = StandardScaler()
X3_train_scaled = scaler.fit_transform(X3_train)
X3_test_scaled = scaler.transform(X3_test)

C_range = np.logspace(-4, 3, 100)
param_grid = {"C": C_range}

grid_search = GridSearchCV(
    estimator=SVC(kernel="poly", class_weight="balanced", random_state=42, degree=3),
    param_grid=param_grid,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1
)

grid_search.fit(X3_train, y3_train)
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X3_test)

print(f"Best C value:  {grid_search.best_params_['C'].round(5)}")
print(f"Best F1 Score: {round(grid_search.best_score_, 3)}")
print(classification_report(y3_test, y_pred_best))

survived_alone = X1_test.loc[y1_test == 1, "alone"]
not_survived_alone = X1_test.loc[y1_test == 0, "alone"]
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

for ax, (kernel, svc_model) in zip(axes.flatten(), svc_model_dict["Case 1 - Alone"].items(), strict=True):

    ax.scatter(survived_alone, y1_test.loc[y1_test == 1], c="forestgreen", marker="s", label="Survived")
    ax.scatter(not_survived_alone, y1_test.loc[y1_test == 0], c="royalblue", marker="^", label="Not Survived")

    if hasattr(svc_model, "support_vectors_"):
        support_x = svc_model.support_vectors_[:, 0]
        ax.scatter(support_x, [0.5] * len(support_x), c="black", marker="+", s=100, label="Support Vectors")

    ax.set_title(f"{kernel} Kernel")
    ax.legend()
    ax.grid(True)

fig.suptitle("Support Vectors - SVC (Case 1 - Alone)")
fig.supxlabel("Alone")
fig.supylabel("Survived (0 or 1)")
plt.tight_layout()
plt.show()

survived_age = X2_test.loc[y2_test == 1, "age"]
not_survived_age = X2_test.loc[y2_test == 0, "age"]
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

for ax, (kernel, svc_model) in zip(axes.flatten(), svc_model_dict["Case 2 - Age"].items(), strict=True):

    ax.scatter(survived_age, y2_test.loc[y2_test == 1], c="forestgreen", marker="s", label="Survived")
    ax.scatter(not_survived_age, y2_test.loc[y2_test == 0], c="royalblue", marker="^", label="Not Survived")

    if hasattr(svc_model, "support_vectors_"):
        support_x = svc_model.support_vectors_[:, 0]
        ax.scatter(support_x, [0.5] * len(support_x), c="black", marker="+", s=100, label="Support Vectors")

    ax.set_title(f"{kernel} Kernel")
    ax.legend()
    ax.grid(True)

fig.suptitle("Support Vectors - SVC (Case 2 - Age)", size=14)
fig.supxlabel("Age")
fig.supylabel("Survived (0 or 1)")
plt.tight_layout()
plt.show()

survived = X3_test[y3_test == 1]
not_survived = X3_test[y3_test == 0]
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

for ax, (kernel, svc_model) in zip(axes.flatten(), svc_model_dict["Case 3 - Age + Family Size"].items(), strict=True):

    ax.scatter(survived["age"], survived["family_size"],
            c="forestgreen", marker="s", label="Survived")
    ax.scatter(not_survived["age"], not_survived["family_size"],
            c="royalblue", marker="^", label="Not Survived")

    if hasattr(svc_model, "support_vectors_"):
        support_vectors = svc_model.support_vectors_
        ax.scatter(support_vectors[:, 0], support_vectors[:, 1],
                    c="black", marker="+", s=75, label="Support Vectors")

    ax.set_title(f"{kernel} Kernel")
    ax.legend()
    ax.grid(True)

fig.suptitle("Support Vectors - SVC (Case 3 - Age + Family Size)", size=14)
fig.supxlabel("Age")
fig.supylabel("Family Size")
plt.tight_layout()
plt.show()

nn_model3 = MLPClassifier(
    hidden_layer_sizes=(50, 25, 10),
    solver="lbfgs",
    max_iter=1000,
    random_state=42
)
nn_model3.fit(X3_train, y3_train)
y3_nn_pred = nn_model3.predict(X3_test)

fig, axes = plt.subplots(nrows= 1, ncols= 2, figsize=(9, 4))

classification_table(y3_test, y3_nn_pred, axes[0])
axes[0].set_title("Classification Results\n(Metrics in %)", y=0.78)

cm_nn3 = confusion_matrix(y3_test, y3_nn_pred)
sns.heatmap(cm_nn3, annot=True, cmap="Blues", ax=axes[1],
                  linewidths=.5, linecolor="royalblue")
axes[1].patch.set_edgecolor("royalblue")
axes[1].patch.set_linewidth(0.5)
axes[1].set(xlabel="Predicted", ylabel="Actual", title="Confusion Matrix")

fig.suptitle("Neural Network on Test Data (Case 3 - age + family_size)")
plt.tight_layout()
plt.show()
fig.savefig("nn_classification_age_familysize.png", bbox_inches="tight")

padding = 1
x_min, x_max = X3["age"].min() - padding, X3["age"].max() + padding
y_min, y_max = X3["family_size"].min() - padding, X3["family_size"].max() + padding
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                                    np.linspace(y_min, y_max, 500))

Z = nn_model3.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=["age", "family_size"]))
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 7))
cmap_background = ListedColormap(["lightblue", "lightyellow"])

plt.contourf(xx, yy, Z, cmap=cmap_background, alpha=0.7)
plt.scatter(X3_test["age"][y3_test == 0],
            X3_test["family_size"][y3_test == 0],
            c="blue", marker="^", edgecolor="k", label="Not Survived")
plt.scatter(X3_test["age"][y3_test == 1],
            X3_test["family_size"][y3_test == 1],
            c="gold", marker="s", edgecolor="k", label="Survived")

plt.gca().set(xlabel="Age", ylabel="Family Size", title="Neural Network Decision Surface - Case 3")
plt.legend()
plt.grid(True)
plt.show()

scaler = StandardScaler()
X3_train_scaled = scaler.fit_transform(X3_train)
X3_test_scaled = scaler.transform(X3_test)

smote = SMOTE(random_state=42)
X3_train_bal, y3_train_bal = smote.fit_resample(X3_train_scaled, y3_train)

# CAUTION: Long run time!

nn_model_tune = MLPClassifier(
    max_iter=1000,
    random_state=42
)

parameter_space = {
    "hidden_layer_sizes": [(20,), (50,), (100,), (20,10), (50, 25), (100, 50), (50, 25, 10), (20, 10, 5)],
    "activation": ["tanh", "relu"],
    "solver": ["adam"],
    "alpha": np.logspace(-4,0,10),
    "learning_rate_init": [0.001, 0.01, .1],
    "batch_size": [32, 64, 128]
}

clf = GridSearchCV(nn_model_tune, parameter_space, n_jobs=-1, cv=5, verbose=1, scoring="f1_weighted")
clf.fit(X3_train, y3_train) #bal

best_model = clf.best_estimator_
print("Best parameters found:", clf.best_params_)
print("Best CV score:", clf.best_score_)

y3_best_pred = best_model.predict(X3_test)

fig, axes = plt.subplots(nrows= 1, ncols= 2, figsize=(9, 4))

classification_table(y3_test, y3_best_pred, axes[0])
axes[0].set_title("Classification Results\n(Metrics in %)", y=0.78)

cm_nn3 = confusion_matrix(y3_test, y3_best_pred)
sns.heatmap(cm_nn3, annot=True, cmap="Blues", ax=axes[1],
                  linewidths=.5, linecolor="royalblue")
axes[1].patch.set_edgecolor("royalblue")
axes[1].patch.set_linewidth(0.5)
axes[1].set(xlabel="Predicted", ylabel="Actual", title="Confusion Matrix")

fig.suptitle("Neural Network on Test Data (Case 3 - age + family_size)")
plt.tight_layout()
plt.show()
