from sklearn.tree import DecisionTreeClassifier

# সহজ ডেটা: বয়স → বেঁচে গেছে?
X = [[5], [10], [15], [60], [70], [80]]       # বয়স
y = [1, 1, 1, 0, 0, 0]                        # ১=বেঁচে, ০=মারা

model = DecisionTreeClassifier()
model.fit(X, y)                               # শেখা

print(model.predict([[5]]))    # ৮ বছর → ?
print(model.predict([[5]]))   # ৬৫ বছর → ?