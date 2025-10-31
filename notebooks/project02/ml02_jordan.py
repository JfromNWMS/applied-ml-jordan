"""Script for ml02_jordan.pynb."""
import matplotlib.pyplot as plt
from numpy.linalg import cond
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from statsmodels.stats.stattools import medcouple

from stats_jordan import tukey_fences

# Load seaborns titanic dataset
titanic_df: pd.DataFrame = sns.load_dataset('titanic')  # noqa: Q000
# Display the first 10 rows of the dataset
titanic_df.head(10)

# Check data types and missing values
titanic_df.info()
# Summary statistics
print(titanic_df.describe())
# Check for missing values
print(titanic_df.isnull().sum())
# Check for correlations
titanic_corr: pd.DataFrame = titanic_df.corr(numeric_only=True)
sns.heatmap(titanic_corr, annot=True, cmap='mako', fmt='.2f')  # noqa: Q000
plt.title(f"Pearson Correlation Matrix\nCondition Number: {cond(titanic_corr):.1f}")
plt.show()

attributes: list = ['age', 'fare', 'pclass']  # noqa: Q000
pd.plotting.scatter_matrix(titanic_df[attributes], figsize=(7,7))

print(f"""
'fare' Medcouple (Robust Skewness): {medcouple(titanic_df["fare"]):.2f}
'fare' Standard Skewness:           {titanic_df["fare"].skew():.2f}
""")

fare: pd.Series = titanic_df["fare"]

# Loop to compare standard vs. Medcouple-adjusted Tukey fences for 'fare'.
for adj in (False, True):
    fences = tukey_fences(fare, adjusted=adj)
    is_extreme_outlier: pd.Series = (fare > fences.outer_upper) | (fare < fences.outer_lower)
    is_outlier: pd.Series = (fare > fences.inner_upper) | (fare < fences.inner_lower)
    print(f"""
    For 'fare':  Fences Adjusted: {adj}
                 {fences}
                 Total Extreme Outliers: {is_extreme_outlier.sum()}
                 Total Outliers:         {is_outlier.sum()}
    """)

sex_to_int = lambda x: 0 if x == 'male' else 1  # noqa: E731, Q000
plt.scatter(titanic_df['age'], titanic_df['fare'], c=titanic_df['sex'].apply(sex_to_int))  # noqa: Q000
plt.gca().set(xlabel='Age', ylabel='Fare', title='Age vs Fare by Gender')  # noqa: Q000
plt.show()

titanic_df.query("fare > 500")[['fare','age','sex','alone']]  # noqa: Q000

sns.histplot(titanic_df['age'], kde=True) # type: ignore  # noqa: Q000
plt.title(f"Age Distribution\nSkewness: {titanic_df['age'].skew():.2f}")
plt.show()

sns.countplot(x='class', hue='survived', data=titanic_df)  # type: ignore # noqa: Q000
plt.title('Class Distribution by Survival')  # noqa: Q000
plt.show()

print(f"{100*sum(fare < 100)/len(fare):.0f}% of passengers paid < £100")
del fare

titanic_df['age'] = titanic_df['age'].fillna(titanic_df['age'].median())  # noqa: Q000
titanic_df['embark_town'] = titanic_df['embark_town'].fillna(titanic_df['embark_town'].mode()[0])  # noqa: Q000

titanic_df['family_size'] = titanic_df['sibsp'] + titanic_df['parch'] + 1  # noqa: Q000
titanic_df['sex'] = titanic_df['sex'].map({'male': 0, 'female': 1})  # noqa: Q000
titanic_df['embarked'] = titanic_df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})  # noqa: Q000
titanic_df['alone'] = titanic_df['alone'].astype(int)  # noqa: Q000

features: list = ["age", "fare", "pclass", "sex", "family_size"]
target: str = "survived"
X: pd.DataFrame = titanic_df[features]  # noqa: N816
y: pd.Series = titanic_df[target]

X_train_basic, X_test_basic, y_train_basic, y_test_basic = train_test_split(X, y, test_size=0.2, random_state = 123)
print(f"""
Train size: {len(X_train_basic)}
Test size:  {len(X_test_basic)}
""")

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123)

for train_ind, test_ind in splitter.split(X, y):
    X_train_stratified = X.iloc[train_ind]
    X_test_stratified = X.iloc[test_ind]
    y_train_stratified = y.iloc[train_ind]
    y_test_stratified = y.iloc[test_ind]

print(f"""
Train size: {len(X_train_stratified)}
Test size:  {len(X_test_stratified)}
""")

print("Original Class Distribution:\n", y.value_counts(normalize=True))
print("Train Set Class Distribution:\n", X_train_stratified["pclass"].value_counts(normalize=True))
print("Test Set Class Distribution:\n", X_test_stratified["pclass"].value_counts(normalize=True))

data_splits = {
    "Training Stratified": y_train_stratified,
    "Training Basic": y_train_basic,
    "Test Stratified": y_test_stratified,
    "Test Basic": y_test_basic
}
text_width: int = 53
proportions: pd.Series = y.value_counts(normalize=True)

for name, split in data_splits.items():
    split_error: pd.Series = abs(split.value_counts(normalize=True) - proportions) / proportions * 100

    for index in split_error.index:
        text: str = f"{name} Split Percent Error for Class {index}:"
        print(f"{text:<{text_width}} {split_error[index]:>.2f}%")
