import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("stock_prices.csv", parse_dates=["date"])
df = df.sort_values("date")

X = df.drop(columns=["price", "date"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("RMSE:", mean_squared_error(y_test, preds, squared=False))
print("R2:", r2_score(y_test, preds))
