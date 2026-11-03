import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_score = 0
best_n = 50
for n in [50, 100, 200, 500]:
    clf = RandomForestClassifier(n_estimators=n, random_state=42)
    clf.fit(X_train, y_train)
    score = accuracy_score(y_test, clf.predict(X_test))
    if score > best_score:
        best_score = score
        best_n = n

final_model = RandomForestClassifier(n_estimators=best_n, random_state=42)
final_model.fit(X_train, y_train)
print("Best n_estimators:", best_n)
print("F1:", f1_score(y_test, final_model.predict(X_test)))
