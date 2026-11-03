import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("features.csv")
X = df.drop("label", axis=1)
y = df["label"]

pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.25)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("F1:", f1_score(y_test, predictions, average="weighted"))
