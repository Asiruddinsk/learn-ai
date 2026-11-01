import pandas as pd
import numpy as np

data = {
    "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
    "year": [2020, 2020, 2020, 2020, 2020],
    "temperature": [75, 80, 65, 85, 90]
}
df = pd.DataFrame(data)
pivot = df.pivot_table(
index="city", columns="year", values="temperature"
)

print(pivot)


pivot = df.pivot_table(
    index="city", columns="year", values="temperature", aggfunc=np.mean
)
print(pivot)