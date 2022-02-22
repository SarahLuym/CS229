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
# print(theta_final) # [-2.6205116   0.76037154  1.17194674]


# c.
df_x.insert(0, "y", df_y)
df_x["y"] = pd.to_numeric(df_x["y"], downcast='signed')
# print(df_x.head())

# Plot raw data
# sns.scatterplot(x="x1", y="x2", hue="y", data=df_x)

# Generate vector to plot decision boundary
x1_vec = np.linspace(df_x["x1"].min(), df_x["x1"].max(), 2)

# Plot decision boundary
# plt.plot(x1_vec, (-x1_vec * theta_final[1] - theta_final[0]) / theta_final[2], color="red")
# plt.savefig("Q1_c.png")
# plt.show()


# Question 5: Regression for denoising quasar spectra
# b.i.
df_train = pd.read_csv("./data/quasar_train.csv")
cols_train = df_train.columns.values.astype(float).astype(int)
df_test = pd.read_csv("./data/quasar_test.csv")
cols_test = df_test.columns.values.astype(float).astype(int)


def normal_equation(x, y, w=None):
    if w is None:
        return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    else:
        return np.linalg.inv(x.T.dot(w).dot(x)).dot(x.T).dot(w).dot(y)


y = df_train.head(1).values.T
x = np.vstack((np.ones(cols_train.shape), cols_train)).T

theta = normal_equation(x, y)
# print(theta)  # [2.5134, -9.8112e-4]


# plt.subplots()
# plt.plot(x[:, 1], x.dot(theta), linewidth=5)
# ax = sns.regplot(x=x[:, 1], y=y, fit_reg=False)
# ax.set(xlabel="WaveLength", ylabel="Flux")
# plt.savefig("Q5_b_i.png")
# plt.show()


# b.ii
def build_weights(x, x_i, tau=5):
    return np.diag(np.exp((-(x - x_i)[:, 1] ** 2) / (2 * tau ** 2)))


pred = []
for k, x_j in enumerate(x):
    w = build_weights(x, x_j, 5)
    theta = normal_equation(x, y, w)
    pred.append(theta.T.dot(x_j[:, np.newaxis]).ravel()[0])

# plt.subplots()
# ax = sns.regplot(x=x[:, 1], y=y, fit_reg=False)
# plt.plot(x[:, 1], pred, linewidth=3)
# ax.set(xlabel="WaveLength", ylabel="Flux")
# plt.savefig("Q5_b_ii.png")
# plt.show()


# b.iii
# taus = [1, 10, 100, 1000]
# colors = sns.color_palette("muted")
# fig,axes = plt.subplots(2, 2, figsize=(14, 10))
# axes = axes.ravel()
# for i, tau in enumerate(taus):
#     pred = []
#     ax = axes[i]
#     for k, x_j in enumerate(x):
#         w = build_weights(x, x_j, tau)
#         theta = normal_equation(x, y, w)
#         pred.append(theta.T.dot(x_j[:, np.newaxis]).ravel()[0])
#
#     sns.regplot(x=x[:, 1], y=y, fit_reg=False, ax=ax, color=colors[0])
#     ax.plot(x[:, 1], pred, linewidth=3, color=colors[i+1])
#     ax.set(xlabel="WaveLength", ylabel="Flux", title="tau = {}".format(tau))
# plt.savefig("Q5_b_iii.png")
# plt.show()


# c.i
y_train = df_train.values.T
y_test = df_test.values.T


def smoother(x, y_in, tau):
    pred = np.zeros(y_in.shape)
    for i in range(y_in.shape[1]):
        y = y_in[:, i]
        for j, x_j in enumerate(x):
            w = build_weights(x, x_j, tau)
            theta = normal_equation(x, y, w)
            pred[j][i] = theta.T.dot(x_j[:, np.newaxis]).ravel()[0]
    return pred


y_train_smooth = smoother(x, y_train, 5)
y_test_smooth = smoother(x, y_test, 5)


# c.ii
def distance(y, y_i):
    return np.sum((y-y_i[:, np.newaxis])**2, 0)


def ker(t):
    return np.maximum(1-t, 0)


def neighbk(distance, k):
    idx = np.argsort(distance)[:k+1]
    return idx[: k]


def func_regression(x, y_train, y_test, lyman_alpha):
    # Slice the training and test data according to Lyman-alpha
    y_train_left = y_train[x < lyman_alpha, :]
    y_train_right = y_train[x >= lyman_alpha + 100, :]
    y_test_left = y_test[x < lyman_alpha, :]
    y_test_right = y_test[x >= lyman_alpha + 100, :]

    # Format our estimate matrix to store results
    f_hat_left = np.zeros(y_test_left.shape)

    # Compute Distance Matrix
    for i in range(y_test_right.shape[1]):
        deltas = distance(y_train_right, y_test_right[:, i])
        idx = neighbk(deltas, 3)
        # print idx.shape
        h = np.max(deltas)
        weights = ker(deltas / h)[idx]

        f_hat_num = np.sum(y_train_left[:, idx] * weights, 1)
        f_hat_den = np.sum(weights)
        f_hat = f_hat_num / f_hat_den
        f_hat_left[:, i] = f_hat
    return f_hat_left


f_hat_train = func_regression(x[:, 1], y_train_smooth, y_train_smooth, 1200)
y_train_left = y_train_smooth[x[:, 1] < 1200, :]

# plt.plot(x[x[:, 1] < 1200, 1], f_hat_train)
# plt.savefig("Q5_c_ii.png")
# plt.show()

h = np.mean(np.sum((f_hat_train-y_train_left)**2, 0))
print('Average Error between Training set and Estimate: {:.3f}'.format(h))


# c.iii
f_hat_test = func_regression(x[:, 1], y_train_smooth, y_test_smooth, 1200)
y_test_left = y_test_smooth[x[:, 1] < 1200, :]

plt.plot(x[x[:, 1] < 1200, 1], f_hat_test)
plt.savefig("Q5_c_iii_all.png")
plt.show()

h = np.mean(np.sum((f_hat_test-y_test_left)**2, 0))
print('Average Error between Test set and Estimate: {:.3f}'.format(h))

# test example 1
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].plot(x[x[:, 1] < 1200, 1], f_hat_test[:, 0])
axes[0].plot(x[x[:, 1] < 1200, 1], y_test_left[:, 0])
axes[0].set(xlabel="Wavelength", ylabel="Flux""", title="Training Example 1")
axes[0].legend(["Estimate", "Smoothed Value"])

axes[1].plot(x[x[:, 1] < 1200, 1], f_hat_test[:, 5])
axes[1].plot(x[x[:, 1] < 1200, 1], y_test_left[:, 5])
axes[1].set(xlabel="WaveLength", ylabel="Flux", title="Training Example 6")
axes[1].legend(["Estimate", "Smoothed Value"])
plt.savefig("Q5_c_iii.png")
plt.show()