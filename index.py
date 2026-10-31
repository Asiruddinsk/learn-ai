import pandas as pd

data = {
    "name": ["Alice", "Bob", "Charlie", "David", "Eva"],
    "age": [25, 30, 35, 28, 22],
    "marks": [85, 90, 95, 70, 88],
    "city": ["New York", "Los Angeles", "Chicago", "Houston", "Miami"]
}

df = pd.DataFrame(data)
df.to_csv("data.csv", index=False)

print(pd.read_csv("data.csv"))

df = pd.read_csv("data.csv")
print("previous:\n", df)

# pass/fail কলাম বানানো (marks >= 80 হলে pass)
df["pass"] = (df["marks"] >= 80).astype(int)

print("\nপরে:\n", df)

# pass/fail কলাম বানানো (marks >= 80 হলে pass)
df["pass"] = (df["marks"] >= 80).astype(int)

print("\nপরে:\n", df)

# ইনপুট: age, marks
X = df[["age", "marks"]]

# উত্তর: pass
y = df["pass"]

print("X:\n", X)
print("\ny:\n", y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("শেখার ডেটা:", len(X_train))
print("পরীক্ষার ডেটা:", len(X_test))

import sklearn
print(sklearn.__version__)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

print("মডেল শেখা শেষ!")

# Train-এ কত ভালো
print("Train নির্ভুলতা:", model.score(X_train, y_train))

# Test-এ কত ভালো (আসল পরীক্ষা)
print("Test নির্ভুলতা:", model.score(X_test, y_test))

import pandas as pd

new_students = pd.DataFrame({
    "age": [22, 30, 26],
    "marks": [88, 60, 75]
})

predictions = model.predict(new_students)

new_students["predicted"] = predictions
new_students["result"] = new_students["predicted"].map({1: "pass", 0: "fail"})

print(new_students)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plot_tree(model,
          feature_names=["age", "marks"],
          class_names=["fail", "pass"],
          filled=True, rounded=True)
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["fail", "pass"])
disp.plot(cmap="Blues")
plt.show()


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# ডেটা
df = pd.read_csv("data.csv")
df["pass"] = (df["marks"] >= 80).astype(int)

X = df[["age", "marks"]]
y = df["pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# নতুন ছাত্র
new = pd.DataFrame({
    "age": [22, 30, 26],
    "marks": [88, 60, 75]
})

# ✅ সঠিক উপায়
predictions = model.predict(new)
new["predicted"] = predictions
new["result"] = np.where(predictions == 1, "pass", "fail")

print(new)
from sklearn.tree import export_text

rules = export_text(model, feature_names=["age", "marks"])
print(rules)