import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier

df = pd.read_csv('Telco-Customer-Churn.csv')
df['churn_cat'] = df['Churn'].map({'No' : 1, 'Yes' : 2})
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
df["TotalCharges"] = df["TotalCharges"].fillna(0)
df.drop("customerID", axis=1)

def shuffle_and_split(data, test_ratio):
    np.random.seed(42) # Set the seed for reproducibility
    shuffled_indices = np.random.permutation(len(data)) # This returns shuffled indices
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
shuffle_and_split(df, 0.2)

train, test = shuffle_and_split(df, 0.2)

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size = 0.2, random_state= 42)
for train_index, test_index in split.split(df, df['churn_cat']):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]
for sett in (strat_train_set, strat_test_set):
    sett.drop("churn_cat", axis=1, inplace=True)

X_train = strat_train_set.drop(["customerID", "Churn"], axis=1)
y_train = strat_train_set["Churn"].map({"No": 0, "Yes": 1})
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = strat_test_set.drop(["customerID", "Churn"], axis=1)
y_test = strat_test_set["Churn"].map({"No": 0, "Yes": 1})
X_test = pd.get_dummies(X_test, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

model = DecisionTreeClassifier(
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:",
      accuracy_score(y_test, rf_pred))

print(classification_report(y_test, rf_pred))

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

print("\nXGBoost Results")
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))

print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_pred))

xgb_cm = confusion_matrix(y_test, xgb_pred)

print("\nXGBoost Confusion Matrix:")
print(xgb_cm)

cm = confusion_matrix(y_test, rf_pred)

print("\nConfusion Matrix:")
print(cm)

joblib.dump(xgb_model, "xgb_churn_model.pkl")
joblib.dump(X_train.columns.tolist(), "model_columns.pkl")
