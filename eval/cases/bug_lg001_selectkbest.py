import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("genomics.csv")
X = df.drop("disease", axis=1)
y = df["disease"]

selector = SelectKBest(f_classif, k=20)
X_selected = selector.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2)

clf = GradientBoostingClassifier(n_estimators=100)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print("F1:", f1_score(y_test, preds))
