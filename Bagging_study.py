import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# define converts(字典)
def Iris_label(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]


# 1.读取数据集
path = './data/iris.data'
data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: Iris_label})
# converters={4:Iris_label}中“4”指的是第5列：将第5列的str转化为label(number)
# print(data.shape)

# 2.划分数据与标签
x, y = np.split(data, indices_or_sections=(4,), axis=1)  # x为数据，y为标签
y = y.reshape(y.shape[0])
x = x[:, 0:2]
x_train, x_val, y_train, y_val = train_test_split(x, y, random_state=1, train_size=0.8,
                                                                  test_size=0.2)
# print(train_data.shape)


# 单独一颗树
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
y_pred = tree.predict(x_val)
val_acc = accuracy_score(y_val, y_pred)
print("tree val_acc: %.4f" % val_acc)
matrix = confusion_matrix(y_val, y_pred)
print(matrix)

# Bagging
bagging = BaggingClassifier()
bagging.fit(x_train, y_train)
y_pred = bagging.predict(x_val)
val_acc = accuracy_score(y_val, y_pred)
print("bagging val_acc: %.4f" % val_acc)
matrix = confusion_matrix(y_val, y_pred)
print(matrix)
print(classification_report(y_val, y_pred))
