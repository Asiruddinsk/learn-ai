# ১. Merge—দুটি টেবিল জোড়া

import pandas as pd
students = pd.DataFrame({
    "student_id": [1, 2, 3, 4],
    "name": ["Alice", "Bob", "Charlie", "David"]
})

marks = pd.DataFrame({
    "student_id": [1, 2, 3, 5],
    "marks": [85, 90, 78, 92]   
})

merge = pd.merge(students, marks, on="student_id", how="inner")
merge = pd.merge(students, marks, on="student_id", how="left")
merge = pd.merge(students, marks, on="student_id", how="right")
merge = pd.merge(students, marks, on="student_id", how="outer")


print(merge)

