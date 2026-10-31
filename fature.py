import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ১. ডেটা
np.random.seed(42)
n = 20
df = pd.DataFrame({
    "Pclass": np.random.choice([1, 2, 3], n),
    "Sex": np.random.choice(["male", "female"], n),
    "Age": np.random.randint(5, 70, n),
    "Fare": np.random.randint(10, 200, n),
    "Survived": np.random.choice([0, 1], n)
})
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# ২. Feature / Label  ← এটাই missing ছিল!
X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]

# ৩. ভাগ
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ৪. মডেল  ← এটাও missing ছিল!
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ৫. Feature importance
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print(importance)

# ৬. ছবি
plt.barh(importance["feature"], importance["importance"], color="skyblue")
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.gca().invert_yaxis()
plt.show()