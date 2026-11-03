import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ("rfe", RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=10)),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
])

pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print("F1:", f1_score(y_test, preds))
