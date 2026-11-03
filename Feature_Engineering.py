import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# ১. লোড
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# ২. Title
for df in [train, test]:
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.")
    rare = ["Dr","Rev","Col","Major","Mlle","Countess","Ms","Lady",
            "Jonkheer","Don","Dona","Capt","Sir","Mme"]
    df["Title"] = df["Title"].replace(rare, "Rare")
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

# ৩. Family
for df in [train, test]:
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# ৪. Age (Title অনুযায়ী)
for df in [train, test]:
    df["Age"] = df.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))

# ৫. Fare
for df in [train, test]:
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
    df["FareLog"] = np.log1p(df["Fare"])

# ৬. Deck
for df in [train, test]:
    df["Deck"] = df["Cabin"].str[0].fillna("Unknown")

# ৭. Encoding
for df in [train, test]:
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].fillna("S").map({"S": 0, "C": 1, "Q": 2})

train = pd.get_dummies(train, columns=["Title", "Deck"], drop_first=True)
test  = pd.get_dummies(test,  columns=["Title", "Deck"], drop_first=True)
train, test = train.align(test, join="left", axis=1, fill_value=0)

# ৮. Feature
features = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch",
            "FamilySize", "IsAlone", "FarePerPerson", "FareLog", "Embarked"]
features += [c for c in train.columns if c.startswith(("Title_", "Deck_"))]

X = train[features]
y = train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ৯. মডেল
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Train:", round(model.score(X_train, y_train), 3))
print("Test :", round(model.score(X_test, y_test), 3))
cv = cross_val_score(model, X, y, cv=5)
print("CV গড়:", round(cv.mean(), 3))

# ১০. Importance
importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
print("\n", importance.head(10))