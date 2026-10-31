import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# ১. ডেটা (৫০০ জন)
np.random.seed(42)
n = 500
df = pd.DataFrame({
    "Pclass": np.random.choice([1, 2, 3], n, p=[0.2, 0.3, 0.5]),
    "Sex": np.random.choice(["male", "female"], n, p=[0.6, 0.4]),
    "Age": np.random.randint(1, 80, n),
    "Fare": np.random.randint(5, 300, n),
    "SibSp": np.random.randint(0, 5, n),
    "Parch": np.random.randint(0, 4, n)
})
prob = (0.7 * (df["Sex"] == "female").astype(int)
        + 0.2 * (df["Pclass"] == 1).astype(int)
        + 0.1 * (df["Age"] < 15).astype(int))
df["Survived"] = (np.random.rand(n) < prob).astype(int)

# ২. পরিষ্কার
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

# ৩. নতুন feature
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# ৪. Feature / Label
features = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone"]
X = df[features]
y = df["Survived"]

# ৫. ভাগ
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ৬. মডেল
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Train:", round(model.score(X_train, y_train), 3))
print("Test :", round(model.score(X_test, y_test), 3))

# ৭. CV
scores = cross_val_score(model, X, y, cv=5)
print("CV গড়:", round(scores.mean(), 3))

# ৮. Importance
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
print("\n", importance)

# ৯. Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["মারা", "বেঁচে"])
disp.plot(cmap="Blues")
plt.title("Titanic Confusion Matrix")
plt.show()

print(classification_report(y_test, y_pred, target_names=["মারা", "বেঁচে"]))