import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

np.random.seed(42)

# ১. ডেটা
group1 = np.random.normal(loc=[25, 20], scale=3, size=(50, 2))
group2 = np.random.normal(loc=[40, 60], scale=5, size=(50, 2))
group3 = np.random.normal(loc=[55, 100], scale=5, size=(50, 2))

data = np.vstack([group1, group2, group3])
df = pd.DataFrame(data, columns=["age", "spend"])

# ২. K-Means
model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["group"] = model.fit_predict(df[["age", "spend"]])

# ৩. গ্রাফ
plt.figure(figsize=(7, 5))
plt.scatter(df["age"], df["spend"], c=df["group"], cmap="viridis", s=80)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
            c="red", marker="X", s=200, label="Centers")
plt.xlabel("Age")
plt.ylabel("Spend")
plt.title("K-Means Clustering")
plt.legend()
plt.grid(True)
plt.show()

# ৪. Elbow
inertia = []
for k in range(1, 10):
    m = KMeans(n_clusters=k, random_state=42, n_init=10)
    m.fit(df[["age", "spend"]])
    inertia.append(m.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(range(1, 10), inertia, marker="o")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

# ৫. Score
print("Silhouette:", round(silhouette_score(df[["age", "spend"]], df["group"]), 3))
print("\nগ্রুপের গড়:\n", df.groupby("group")[["age", "spend"]].mean())