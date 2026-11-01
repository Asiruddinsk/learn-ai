import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ১. ডেটা
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# ২. ভাগ
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ৩. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ৪. মডেল তুলনা
models = {
    "Logistic": LogisticRegression(max_iter=500),
    "SVM": SVC(kernel="rbf"),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

print(f"{'Model':15s} {'Train':>8s} {'Test':>8s}")
print("-" * 35)
for name, m in models.items():
    m.fit(X_train_scaled, y_train)
    print(f"{name:15s} {m.score(X_train_scaled, y_train):8.3f} {m.score(X_test_scaled, y_test):8.3f}")

# ৫. CV (সেরা মডেল)
best = RandomForestClassifier(n_estimators=100, random_state=42)
cv = cross_val_score(best, X_train_scaled, y_train, cv=5)
print("\nRandom Forest CV গড়:", round(cv.mean(), 3))