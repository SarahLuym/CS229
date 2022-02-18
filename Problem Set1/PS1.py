import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Question 1: Logistic Regression
# b.
df_x = pd.read_csv("./data/logistic_x.txt", sep="\ +", names=['x1', 'x2'], header=None, engine='python')
df_y = pd.read_csv("./data/logistic_y.txt", sep="\ +", names=['y'], header=None, engine='python')
df_y = df_y.astype(int)
# print(df_x.head())
# print(df_y.head())


# Get numpy array from the data set and add a column of 1 as the zero intercept
x = np.hstack([np.ones((df_x.shape[0], 1)), df_x[["x1", "x2"]].values])
y = df_y["y"].values


# Define the Sigmoid, Gradient and Hessian function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def grad_l(theta, x, y):
    z = y * x.dot(theta)
    g = -np.mean((1 - sigmoid(z)) * y * x.T, axis=1)
    return g


def hess_l(theta, x, y):
    hess = np.zeros((x.shape[1], x.shape[1]))
    z = y * x.dot(theta)
    for i in range(hess.shape[0]):
        for j in range(hess.shape[1]):
            if i <= j:
                hess[i][j] = np.mean(sigmoid(z) * (1 - sigmoid(z)) * x[:, i] * x[:, j])
                if i != j:
                    hess[j][i] = hess[i][j]
    return hess


# Define Newton's Method
def newton(theta0, x, y, G, H, eps):
    theta = theta0
    delta = 1
    while delta > eps:
        theta_prev = theta.copy()
        theta -= np.linalg.inv(H(theta, x, y)).dot(G(theta, x, y))
        delta = np.linalg.norm(theta - theta_prev, ord=1)
    return theta


# Perform Logistic Regression
# Initialize theta0
theta0 = np.zeros((x.shape[1]))

# Run Newton's Method
theta_final = newton(theta0, x, y, grad_l, hess_l, 1e-6)
print(theta_final) # [-2.6205116   0.76037154  1.17194674]


# c.
df_x.insert(0, "y", df_y)
df_x["y"] = pd.to_numeric(df_x["y"], downcast='signed')
print(df_x.head())

# Plot raw data
sns.scatterplot(x="x1", y="x2", hue="y", data=df_x)

# Generate vector to plot decision boundary
x1_vec = np.linspace(df_x["x1"].min(), df_x["x1"].max(), 2)

# Plot decision boundary
plt.plot(x1_vec, (-x1_vec * theta_final[1] - theta_final[0]) / theta_final[2], color="red")
plt.savefig("Q1_c.png")
plt.show()



# Question 2: Poisson regression and the exponential family
