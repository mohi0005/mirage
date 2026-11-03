import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(random_state=42, max_iter=500)),
])

scores = cross_val_score(pipe, X, y, cv=5, scoring="f1")
print("Mean F1:", scores.mean())
print("Std F1:", scores.std())
