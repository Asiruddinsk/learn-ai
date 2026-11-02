import pandas as pd 
jan =pd.DataFrame({
    "name":["Asiruddin sk","Alina"],
    "age":[39,13]
})
feb = pd.DataFrame({
    "name":["Ansura","BilKis"],
    "age":[18,34]
})

combined = pd.concat([jan,feb],ignore_index=True)
print(combined)