import pandas as pd

sub = pd.read_csv("submission.csv")

print("আকার:", sub.shape)          # (100, 2)
print("কলাম:", list(sub.columns))   # ['PassengerId', 'Survived']
print("খালি মান:\n", sub.isnull().sum())
print("\nSurvived গণনা:\n", sub["Survived"].value_counts())

# শর্ত যাচাই
print("\nPassengerId ইউনিক?", sub["PassengerId"].is_unique)
print("Survived শুধু 0/1?", sub["Survived"].isin([0, 1]).all())