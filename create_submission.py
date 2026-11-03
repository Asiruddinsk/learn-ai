import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ─────────────────────────────
# ১. ডেটা লোড
# ─────────────────────────────
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

# ─────────────────────────────
# ২. প্রসেসিং ফাংশন (দুটোতেই একই)
# ─────────────────────────────
def process(df):
    df = df.copy()
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
    df["FareLog"] = np.log1p(df["Fare"])
    return df

train_p = process(train)
test_p  = process(test)

# ─────────────────────────────
# ৩. Feature ও Label  ← এটাই missing ছিল!
# ─────────────────────────────
features = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch",
            "FamilySize", "IsAlone", "FarePerPerson", "FareLog"]

X = train_p[features]
y = train["Survived"]

# ─────────────────────────────
# ৪. পুরো train দিয়ে শেখা
# ─────────────────────────────
model_final = RandomForestClassifier(
    n_estimators=100, max_depth=7, min_samples_leaf=2, random_state=42
)
model_final.fit(X, y)
print("মডেল শেখা শেষ!")

# ─────────────────────────────
# ৫. test-এ predict
# ─────────────────────────────
X_test = test_p[features]
predictions = model_final.predict(X_test)

# ─────────────────────────────
# ৬. সাবমিশন তৈরি
# ─────────────────────────────
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})
submission.to_csv("submission.csv", index=False)

print("\n✅ submission.csv তৈরি!")
print("মোট:", len(submission))
print(submission.head())

import pandas as pd

sub = pd.read_csv("submission.csv")

print("আকার:", sub.shape)
print("কলাম:", list(sub.columns))
print("খালি মান:", sub.isnull().sum().sum())
print("Survived মান:", sorted(sub["Survived"].unique()))
print("PassengerId ইউনিক?", sub["PassengerId"].is_unique)
print("\nবণ্টন:\n", sub["Survived"].value_counts())

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

def process(df):
    df = df.copy()
    # Title
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.")
    rare = ["Dr","Rev","Col","Major","Mlle","Countess","Ms","Lady",
            "Jonkheer","Don","Dona","Capt","Sir","Mme"]
    df["Title"] = df["Title"].replace(rare, "Rare")
    
    # Sex, Embarked
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].fillna("S").map({"S": 0, "C": 1, "Q": 2})
    
    # Age, Fare
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    
    # Family
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
    df["FareLog"] = np.log1p(df["Fare"])
    return df

train_p = process(train)
test_p  = process(test)

# Title One-Hot
train_p = pd.get_dummies(train_p, columns=["Title"], drop_first=True)
test_p  = pd.get_dummies(test_p,  columns=["Title"], drop_first=True)
train_p, test_p = train_p.align(test_p, join="left", axis=1, fill_value=0)

# Feature
features = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch",
            "FamilySize", "IsAlone", "FarePerPerson", "FareLog", "Embarked"]
features += [c for c in train_p.columns if c.startswith("Title_")]

X = train_p[features]
y = train["Survived"]

# মডেল
model = RandomForestClassifier(
    n_estimators=200, max_depth=7, min_samples_leaf=2, random_state=42
)
model.fit(X, y)

# Predict
predictions = model.predict(test_p[features])

# সাবমিশন
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})
submission.to_csv("submission_real.csv", index=False)

print("submission_real.csv তৈরি!")
print("মোট:", len(submission))