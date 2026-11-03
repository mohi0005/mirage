import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

df = pd.read_csv("transactions.csv")
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

sm = SMOTE()
X_bal, y_bal = sm.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.25)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print("F1:", f1_score(y_test, preds))
print(classification_report(y_test, preds))
