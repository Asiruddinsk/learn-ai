import pandas as pd

# দুটি টেবিল, একই কলাম
jan = pd.DataFrame({
    "name": ["রহিম", "করিম"],
    "marks": [85, 72]
})

feb = pd.DataFrame({
    "name": ["সালাম", "জামাল"],
    "marks": [90, 60]
})

# উপর-নিচে জোড়া
combined = pd.concat([jan, feb], ignore_index=True)
print(combined)

names = pd.DataFrame({"name": ["A", "B", "C"]})
marks = pd.DataFrame({"marks": [85, 90, 95]})

side = pd.concat([names, marks], axis=1)
print(side)

import pandas as pd

df = pd.DataFrame({
    "date": ["2026-01-15", "2026-02-20", "2026-03-10"],
    "sales": [100, 150, 120]
})

# লেখা → datetime
df["date"] = pd.to_datetime(df["date"])
print(df)
print(df.dtypes)

df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"]   = df["date"].dt.day
df["weekday"] = df["date"].dt.day_name()

print(df)

monthly = df.groupby("month")["sales"].sum()
print(monthly)

df["next_week"] = df["date"] + pd.Timedelta(days=7)
print(df[["date", "next_week"]])

# ফেব্রুয়ারির পরে
filtered = df[df["date"] > "2026-02-01"]
print(filtered)

df["formatted"] = df["date"].dt.strftime("%d-%m-%Y")
print(df[["date", "formatted"]])

import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# ১. Concat: train + test (Survived ছাড়া)
train_no_survived = train.drop(columns=["Survived"])
all_data = pd.concat([train_no_survived, test], ignore_index=True)

print("Train:", train.shape)
print("Test :", test.shape)
print("All  :", all_data.shape)

# উদাহরণ: বিক্রয় ডেটা
sales = pd.DataFrame({
    "date": pd.date_range("2026-01-01", periods=10, freq="D"),
    "amount": [100, 150, 120, 200, 180, 220, 90, 130, 160, 140]
})

sales["month"] = sales["date"].dt.month
sales["weekday"] = sales["date"].dt.day_name()

print(sales.groupby("weekday")["amount"].mean())


import pandas as pd

# ১. Concat—উপর-নিচে
jan = pd.DataFrame({"name": ["রহিম", "করিম"], "marks": [85, 72]})
feb = pd.DataFrame({"name": ["সালাম", "জামাল"], "marks": [90, 60]})

combined = pd.concat([jan, feb], ignore_index=True)
print("Concat (উপর-নিচে):\n", combined)

# ২. Concat—পাশাপাশি
names = pd.DataFrame({"name": ["A", "B", "C"]})
marks = pd.DataFrame({"marks": [85, 90, 95]})

side = pd.concat([names, marks], axis=1)
print("\nConcat (পাশাপাশি):\n", side)

# ৩. Datetime
df = pd.DataFrame({
    "date": ["2026-01-15", "2026-02-20", "2026-03-10", "2026-02-05"],
    "sales": [100, 150, 120, 180]
})
df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["weekday"] = df["date"].dt.day_name()

print("\nDatetime:\n", df)

# ৪. মাস অনুযায়ী যোগফল
print("\nমাস অনুযায়ী বিক্রয়:\n", df.groupby("month")["sales"].sum())

# ৫. তারিখ ফিল্টার
print("\nফেব্রুয়ারির ডেটা:\n", df[df["month"] == 2])

# ৬. Titanic-এ concat
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train_no_survived = train.drop(columns=["Survived"])
all_data = pd.concat([train_no_survived, test], ignore_index=True)
print("\nTitanic All shape:", all_data.shape)