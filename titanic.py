import pandas as pd
import numpy as np

# ২০ জন যাত্রীর ডেটা
np.random.seed(42)
n = 20

df = pd.DataFrame({
    "Pclass": np.random.choice([1, 2, 3], n),
    "Sex": np.random.choice(["male", "female"], n),
    "Age": np.random.randint(5, 70, n),
    "Fare": np.random.randint(10, 200, n),
    "Survived": np.random.choice([0, 1], n)
})

print(df)

print("আকার:", df.shape)
print("\nকতজন বেঁচে গেছে:\n", df["Survived"].value_counts())
print("\nখালি মান:\n", df.isnull().sum())
print("\nপরিসংখ্যান:\n", df.describe())

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
print(df.head())

features = ["Pclass", "Sex", "Age", "Fare"]

X = df[features]
y = df["Survived"]

print("X:\n", X.head())
print("\ny:\n", y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("শেখার ডেটা:", len(X_train))
print("পরীক্ষার ডেটা:", len(X_test))

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("মডেল শেখা শেষ!")

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("Train নির্ভুলতা:", train_score)
print("Test নির্ভুলতা:", test_score)

new_passengers = pd.DataFrame({
    "Pclass": [1, 3, 2],
    "Sex": [1, 0, 1],       # 1=মহিলা, 0=পুরুষ
    "Age": [25, 40, 15],
    "Fare": [100, 20, 50]
})

predictions = model.predict(new_passengers)
new_passengers["Survived"] = predictions
new_passengers["Result"] = np.where(predictions == 1, "বেঁচে গেছে", "মারা গেছে")

print(new_passengers)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print("Confusion Matrix:\n", cm)
print("\nরিপোর্ট:\n", classification_report(y_test, y_pred, target_names=["মারা", "বেঁচে"]))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["মারা", "বেঁচে"])
disp.plot(cmap="Blues")
plt.title("Titanic Confusion Matrix")
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

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

# ২. এনকোডিং
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# ৩. Feature / Label
X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]

# ৪. ভাগ
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ৫. শেখা
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ৬. পরীক্ষা
print("Train:", model.score(X_train, y_train))
print("Test :", model.score(X_test, y_test))

# ৭. নতুন预测
new = pd.DataFrame({
    "Pclass": [1, 3, 2],
    "Sex": [1, 0, 1],
    "Age": [25, 40, 15],
    "Fare": [100, 20, 50]
})
new["Survived"] = model.predict(new)
new["Result"] = new["Survived"].map({1: "বেঁচে গেছে", 0: "মারা গেছে"})
print(new)

# ৮. Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print("\nConfusion Matrix:\n", cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["মারা", "বেঁচে"])
disp.plot(cmap="Blues")
plt.title("Titanic Confusion Matrix")
plt.show()