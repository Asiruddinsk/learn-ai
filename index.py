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