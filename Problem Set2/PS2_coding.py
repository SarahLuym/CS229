import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Question 1 Logistic Regression: Training stability
# The original code for the logistic regression training algorithm for this problem
try:
    xrange
except NameError:
    xrange = range


def add_intercept(X_):
    m, n = X_.shape
    X = np.zeros((m, n + 1))
    X[:, 0] = 1
    X[:, 1:] = X_
    return X


def load_data(filename):
    D = np.loadtxt(filename)
    Y = D[:, 0]
    X = D[:, 1:]
    return add_intercept(X), Y


def calc_grad(X, Y, theta):
    m, n = X.shape
    grad = np.zeros(theta.shape)

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad


def logistic_regression(X, Y):
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta  - learning_rate * (grad)
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
            print("DeltaT = {0}".format(prev_theta - theta))
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return


def main():
    print('==== Training model on data set A ====')
    Xa, Ya = load_data('./data/data_a.txt')
    logistic_regression(Xa, Ya)

    print('\n==== Training model on data set B ====')
    Xb, Yb = load_data('./data/data_b.txt')
    logistic_regression(Xb, Yb)

    return


# a.
# main()
# Answer: When training on data set A, gradient descent converged in 30368 iterations.
# However, when training on data set B, gradient descent failed to converge.


# b.
columns = ['y', 'x1', 'x2']
df_a = pd.read_csv("./data/data_a.txt", sep="\s+", header=None)
df_b = pd.read_csv("./data/data_b.txt", sep="\s+", header=None)
df_a.columns = columns
# df_a['y'].astype('category')
df_b.columns = columns
# df_b['y'].astype('category')
# print(df_a.head())
# print(df_b.head())
# print(df_a.describe())
# print(df_b.describe())

# Draw scatter plot
plt.subplots()
plt.scatter(x=df_a['x1'], y=df_a['x2'], c=df_a['y'], cmap='rainbow')
plt.title("Scatter Plot for df_a")
plt.savefig("Q1_b_dfa.png")
plt.show()

plt.subplots()
plt.scatter(x=df_b["x1"], y=df_b["x2"], c=df_b["y"], cmap="coolwarm")
plt.title("Scatter Plot for df_b")
plt.savefig("Q1_b_dfb.png")
plt.show()

# dataframe b is perfectly linearly separable, while dataframe a is not.
# When the data is linearly separable, theta can be very large and thus hθ(x) will be very close to 1 and the log likelihood will be very large as long as theta is a hyperplane that separates the data perfectly.


# c.
# i. No. This is a scaling issue.
# ii. No. This is a scaling issue.
# iii. Yes. It can prevent theta from becoming very large.
# iv. No. The data is still perfectly linearly separable.
# v. Yes. The data will not be perfectly linearly separable.

# d.
# It's not vulnerable to datasets like B, because it will map the data to high dimensional, which will be less possible to be linearly perfectly separable.
# Also, SVM maximize the geometric margin of the data (the distance between the separator and the data). It is independent of theta.



# Question 2 Model Calibration
