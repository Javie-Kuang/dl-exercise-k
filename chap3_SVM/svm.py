# python: 3.5.2
# encoding: utf-8

import numpy as np


def load_data(fname):
    """
    载入数据。
    """
    with open(fname, 'r') as f:
        data = []
        line = f.readline()
        for line in f:
            line = line.strip().split()
            x1 = float(line[0])
            x2 = float(line[1])
            t = int(line[2])
            data.append([x1, x2, t])
        return np.array(data)


def eval_acc(label, pred):
    """
    计算准确率。
    """
    return np.sum(label == pred) / len(pred)


class SVM():
    """
    SVM模型。
    """

    def __init__(self):
        # 请补全此处代码
        self.w = None
        self.b = 0
        self.lr = 0.01
        self.epochs = 1000
        self.C = 1.0

    def linear_kernel(x1, x2):
        return np.dot(x1, x2)
    
    def polynomial_kernel(x1, x2, degree=3):
        return (np.dot(x1, x2) + 1) ** degree
    
    def rbf_kernel(x1, x2, gamma=0.5):
        return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

    def train(self, data_train):
        """
        训练模型。
        """
        X = data_train[:, :2]
        y = data_train[:, 2]
        y = np.where(y == 0, -1, 1)  # 将标签转换为 -1 和 1
        
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        
        for epoch in range(self.epochs):
            for i in range(n_samples):
                condition = y[i] * (np.dot(X[i], self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.w)
                else:
                    self.w -= self.lr * (2 * self.w - self.C * y[i] * X[i])
                    self.b -= self.lr * (self.C * y[i])

        # 请补全此处代码

    def predict(self, x):
        """
        预测标签。
        """
        linear_output = np.dot(x, self.w) + self.b
        pred = np.where(linear_output >= 0, 1, 0)
        return pred

        # 请补全此处代码


if __name__ == '__main__':
    # 载入数据，实际实用时将x替换为具体名称
    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'
    data_train = load_data(train_file)  # 数据格式[x1, x2, t]
    data_test = load_data(test_file)

    # 使用训练集训练SVM模型
    svm = SVM()  # 初始化模型
    svm.train(data_train)  # 训练模型

    # 使用SVM模型预测标签
    x_train = data_train[:, :2]  # feature [x1, x2]
    t_train = data_train[:, 2]  # 真实标签
    t_train_pred = svm.predict(x_train)  # 预测标签
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    t_test_pred = svm.predict(x_test)

    # 评估结果，计算准确率
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))
