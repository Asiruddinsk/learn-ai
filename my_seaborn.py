import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

tips = sns.load_dataset("tips")   # Seaborn-এর বিল্ট-ইন ডেটা
print(tips.head())

sns.histplot(data=tips, x="total_bill", bins=20)
plt.title("Bill Distribution")
plt.show()

sns.scatterplot(data=tips, x="total_bill", y="tip", hue="sex")
plt.title("Bill vs Tip")
plt.show()

sns.boxplot(data=tips, x="day", y="total_bill")
plt.title("Bill by Day")
plt.show()

sns.barplot(data=tips, x="day", y="total_bill", estimator="mean")
plt.title("Average Bill by Day")
plt.show()

# শুধু সংখ্যার কলাম
corr = tips.select_dtypes(include="number").corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

# ডেটা
train = pd.read_csv("train.csv")

# ১. বেঁচে থাকার হার
sns.countplot(data=train, x="Survived", hue="Sex")
plt.title("Survived by Sex")
plt.show()

# ২. বয়সের বণ্টন
sns.histplot(data=train, x="Age", bins=20, hue="Survived")
plt.title("Age Distribution by Survival")
plt.show()

# ৩. ক্লাস অনুযায়ী ভাড়া
sns.boxplot(data=train, x="Pclass", y="Fare")
plt.title("Fare by Class")
plt.show()

# ৪. Correlation
numeric = train.select_dtypes(include="number")
plt.figure(figsize=(10, 8))
sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Titanic Correlation")
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

# ১. Merge
students = pd.DataFrame({
    "id": [1, 2, 3, 4],
    "name": ["রহিম", "করিম", "সালাম", "জামাল"]
})
marks = pd.DataFrame({
    "id": [1, 2, 3, 5],
    "marks": [85, 72, 90, 88]
})
merged = pd.merge(students, marks, on="id", how="left")
print("Merge:\n", merged)

# ২. Pivot
data = {
    "city": ["ঢাকা", "ঢাকা", "চট্টগ্রাম", "চট্টগ্রাম", "ঢাকা"],
    "gender": ["M", "F", "M", "F", "M"],
    "marks": [85, 90, 72, 88, 78]
}
df = pd.DataFrame(data)
pivot = df.pivot_table(values="marks", index="city", columns="gender", aggfunc="mean")
print("\nPivot:\n", pivot)

# ৩. Seaborn
tips = sns.load_dataset("tips")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.histplot(data=tips, x="total_bill", bins=20)
plt.title("Bill Distribution")

plt.subplot(1, 2, 2)
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="sex")
plt.title("Bill vs Tip")

plt.tight_layout()
plt.show()

# ৪. Titanic
train = pd.read_csv("train.csv")
sns.countplot(data=train, x="Survived", hue="Sex")
plt.title("Titanic: Survived by Sex")
plt.show()