import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score

# ------ WINE DATASET ------
# read datasets into pandas dataframe
df = pd.read_csv('data/WineQT.csv')

# split into attributes and labels
# -convert to NumPy
X = df.drop('quality', axis=1).to_numpy()
y = df['quality'].to_numpy()
# ------ WINE DATASET ------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# create decision tree classifier
decision_tree = tree.DecisionTreeClassifier()

print(cross_val_score(decision_tree, X_train, y_train, cv=5))

# train decision tree
#decision_tree.fit(X_train, y_train)

'''

data = np.genfromtxt('data/WineQT.csv', delimiter=',', names=True, dtype=np.float64)
print(data)
data = np.array(data)

min = len(data[0])
max = len(data[0])

for x in data:

    cur_len = len(x)
    if cur_len < min:
        min = cur_len
    if cur_len > max:
        max = cur_len

print(min, max)

label_name = 'quality'
columns = data.dtype.names
label_idx = columns.index(label_name)

print(data.shape)


X = np.copy(data)
print(label_idx)
print(X.dtype)
print(X[X.dtype.names == 'quality'])
X = np.delete(X, label_idx, 1)
y = data[label_name]

print(X)
'''

#print(X.dtype)

# split into train/test

# create model

# train with training dataset (y should be 'quality' column)

# cv - not 100% sure how this fits in just yet

# score with test set