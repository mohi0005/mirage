import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_test, y_test)

best_model = grid.best_estimator_
preds = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print("F1:", f1_score(y_test, preds))
