import pandas as pd
import numpy as np

# train ডেটা বানানো
np.random.seed(42)
n = 500
train = pd.DataFrame({
    "PassengerId": range(1, n+1),
    "Pclass": np.random.choice([1, 2, 3], n, p=[0.2, 0.3, 0.5]),
    "Sex": np.random.choice(["male", "female"], n, p=[0.6, 0.4]),
    "Age": np.random.randint(1, 80, n),
    "Fare": np.random.randint(5, 300, n),
    "SibSp": np.random.randint(0, 5, n),
    "Parch": np.random.randint(0, 4, n)
})
prob = (0.7*(train["Sex"]=="female").astype(int)
        + 0.2*(train["Pclass"]==1).astype(int)
        + 0.1*(train["Age"]<15).astype(int))
train["Survived"] = (np.random.rand(n) < prob).astype(int)

# test ডেটা
m = 100
test = pd.DataFrame({
    "PassengerId": range(1001, 1001+m),
    "Pclass": np.random.choice([1, 2, 3], m),
    "Sex": np.random.choice(["male", "female"], m),
    "Age": np.random.randint(1, 80, m),
    "Fare": np.random.randint(5, 300, m),
    "SibSp": np.random.randint(0, 5, m),
    "Parch": np.random.randint(0, 4, m)
})

# সেভ করে রাখুন
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

print("train.csv:", train.shape)
print("test.csv :", test.shape)
print("\nদুটোই তৈরি হয়েছে!")