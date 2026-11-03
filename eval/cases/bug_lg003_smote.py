import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report

df = pd.read_csv("imbalanced.csv")
X = df.drop("fraud", axis=1)
y = df["fraud"]

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2
)

clf = GradientBoostingClassifier(n_estimators=200)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print("F1:", f1_score(y_test, preds))
print(classification_report(y_test, preds))
