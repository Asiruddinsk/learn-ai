import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# ═══════════════════════════════════════════
# ১. ডেটা লোড
# ═══════════════════════════════════════════
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

print("Train:", train.shape)
print("Test :", test.shape)

# ═══════════════════════════════════════════
# ২. ডেটা পরিষ্কার (একই ফাংশন দুটোতেই)
# ═══════════════════════════════════════════
def process(df):
    df = df.copy()
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["SibSp"] = df["SibSp"].fillna(0)
    df["Parch"] = df["Parch"].fillna(0)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df

train_p = process(train)
test_p  = process(test)

# ═══════════════════════════════════════════
# ৩. Feature / Label
# ═══════════════════════════════════════════
features = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone"]

X = train_p[features]
y = train["Survived"]

print("\nFeature:", features)
print("X shape:", X.shape)
print("y shape:", y.shape)

# ═══════════════════════════════════════════
# ৪. Train/Test ভাগ
# ═══════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain:", len(X_train), "| Test:", len(X_test))

# ═══════════════════════════════════════════
# ৫. মডেল শেখানো
# ═══════════════════════════════════════════
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\nTrain score:", round(model.score(X_train, y_train), 3))
print("Test score :", round(model.score(X_test, y_test), 3))

# ═══════════════════════════════════════════
# ৬. Cross-validation
# ═══════════════════════════════════════════
cv_scores = cross_val_score(model, X, y, cv=5)
print("\nCV স্কোর:", cv_scores.round(3))
print("CV গড়  :", round(cv_scores.mean(), 3))

# ═══════════════════════════════════════════
# ৭. Feature Importance
# ═══════════════════════════════════════════
importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(8, 4))
plt.barh(importance["feature"], importance["importance"], color="skyblue")
plt.xlabel("Importance")
plt.title("Titanic — Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════
# ৮. Confusion Matrix
# ═══════════════════════════════════════════
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["মারা", "বেঁচে"])
disp.plot(cmap="Blues")
plt.title("Titanic — Confusion Matrix")
plt.show()

print("\nবিস্তারিত রিপোর্ট:")
print(classification_report(y_test, y_pred, target_names=["মারা", "বেঁচে"]))

# ═══════════════════════════════════════════
# ৯. সাবমিশন তৈরি
# ═══════════════════════════════════════════
# পুরো train দিয়ে আবার শেখা (আরও ডেটা = ভালো)
model.fit(X, y)

# test-এ predict
predictions = model.predict(test_p[features])

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})
submission.to_csv("submission.csv", index=False)

print("\n✅ submission.csv তৈরি!")
print("মোট সারি:", len(submission))
print(submission.head(10))