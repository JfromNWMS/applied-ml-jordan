"""Script for ml04_jordan.ipynb."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

titanic_df: pd.DataFrame = sns.load_dataset("titanic")
titanic_df.head(10)

median_age = titanic_df["age"].median()
titanic_df["age"] = titanic_df["age"].fillna(median_age)

titanic_df["sex"] = titanic_df["sex"].map({"male": 1, "female": 0})

titanic_df = titanic_df.dropna(subset=["fare"])

titanic_df["family_size"] = titanic_df["sibsp"] + titanic_df["parch"] + 1

# Case 1: age
X1 = titanic_df[["age"]]
y1 = titanic_df["fare"]

# Case 2: family_size
X2 = titanic_df[["family_size"]]
y2 = titanic_df["fare"]

# Case 3: age and family size
X3 = titanic_df[["age", "family_size"]]
y3 = titanic_df["fare"]

# Case 4: sex
X4 = titanic_df[["sex"]]
y4 = titanic_df["fare"]

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=123)

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=123)

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=123)

X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.2, random_state=123)

linear_dict: dict = {
    "Case 1: Age": (X1_train, X1_test, y1_train, y1_test),
    "Case 2: Family Size": (X2_train, X2_test, y2_train, y2_test),
    "Case 3: Age + Family Size": (X3_train, X3_test, y3_train, y3_test),
    "Case 4: Sex": (X4_train, X4_test, y4_train, y4_test)
}

linear_y_pred_test: dict = {}

for case, (X_train, X_test, y_train, y_test) in linear_dict.items():

    lr_model = LinearRegression().fit(X_train, y_train)
    y_pred_train = lr_model.predict(X_train)
    linear_y_pred_test[case] = lr_model.predict(X_test)

    print(f"{case}")
    print("-"*len(case))
    print(f"{'Training R²:':<{12}} {r2_score(y_train, y_pred_train):>.3f}")
    print(f"{'Test R²:':<{12}} {r2_score(y_test, linear_y_pred_test[case]):>.3f}")
    print(f"{'Test RMSE:':<{12}} {root_mean_squared_error(y_test, linear_y_pred_test[case]):>.2f}")
    print(f"{'Test MAE:':<{12}} {mean_absolute_error(y_test, linear_y_pred_test[case]):>.2f}\n")

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X4_train, y4_train)
y_pred_ridge = ridge_model.predict(X4_test)

elastic_model = ElasticNet(alpha=0.3, l1_ratio=0.5)
elastic_model.fit(X4_train, y4_train)
y_pred_elastic = elastic_model.predict(X4_test)

poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X1_train)
X_test_poly = poly.transform(X1_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y1_train)
y_pred_poly = poly_model.predict(X_test_poly)

plt.scatter(X1_test, y1_test, color="blue", label="Actual")
plt.scatter(X1_test, y_pred_poly, color="red", label="Predicted (Poly)")
plt.legend()
plt.gca().set(xlabel="Age", ylabel="Fare", title="Polynomial Regression: Age vs Fare")
plt.show()

def report(name, y_true, y_pred):  # noqa: D103
    print(f"{name}")
    print("-"*len(name))
    print(f"{'R²:':<{5}} {r2_score(y_true, y_pred):>.3f}")
    print(f"{'RMSE:':<{5}} {root_mean_squared_error(y_true, y_pred):>.2f}")
    print(f"{'MAE:':<{5}} {mean_absolute_error(y_true, y_pred):>.2f}\n")

report("Linear", y4_test, linear_y_pred_test["Case 4: Sex"])
report("Ridge", y4_test, y_pred_ridge)
report("ElasticNet", y4_test, y_pred_elastic)
report("Polynomial", y1_test, y_pred_poly)

poly6 = PolynomialFeatures(degree=6)
X_train_poly6 = poly6.fit_transform(X1_train)
X_test_poly6 = poly6.transform(X1_test)

poly_model6 = LinearRegression()
poly_model6.fit(X_train_poly6, y1_train)
y_pred_poly6 = poly_model6.predict(X_test_poly6)

plt.scatter(X1_test, y1_test, color="blue", label="Actual")
plt.scatter(X1_test, y_pred_poly6, color="red", label="Predicted (Poly)")
plt.legend()
plt.gca().set(xlabel="Age", ylabel="Fare", title="Sixth Degree Polynomial Regression: Age vs Fare")
plt.show()
