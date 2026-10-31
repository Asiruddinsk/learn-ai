import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ১. ডেটা
df = pd.read_csv("data.csv")
df["pass"] = (df["marks"] >= 80).astype(int)

# ২. Feature / Label
X = df[["age", "marks"]]
y = df["pass"]

# ৩. ভাগ
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ৪. মডেল তৈরি ও শেখানো  ← এটাই missing ছিল!
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# ৫. অনুমান
y_pred = model.predict(X_test)

# ৬. Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print("Confusion Matrix:\n", cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["fail", "pass"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()