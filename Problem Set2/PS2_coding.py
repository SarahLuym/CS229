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
# plt.subplots()
# plt.scatter(x=df_a['x1'], y=df_a['x2'], c=df_a['y'], cmap='rainbow')
# plt.title("Scatter Plot for df_a")
# plt.savefig("Q1_b_dfa.png")
# plt.show()
#
# plt.subplots()
# plt.scatter(x=df_b["x1"], y=df_b["x2"], c=df_b["y"], cmap="coolwarm")
# plt.title("Scatter Plot for df_b")
# plt.savefig("Q1_b_dfb.png")
# plt.show()

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



# Question 6 Spam classification
# a.
def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)


def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print('Error: %1.4f' % error)
    return error


def main():
    trainMatrix, tokenlist, trainCategory = readMatrix('./data/MATRIX.TRAIN')
    testMatrix, tokenlist, testCategory = readMatrix('./data/MATRIX.TEST')

    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)

    evaluate(output, testCategory)
    return


trainMatrix, tokenlist, trainCategory = readMatrix('./data/MATRIX.TRAIN')
testMatrix, tokenlist, testCategory = readMatrix('./data/MATRIX.TEST')

def nb_train(matrix, category):
    state = {}
    N = matrix.shape[1]
    ###################
    spam = matrix[category == 1, :]
    nospam = matrix[category == 0, :]

    spam_lengths = spam.sum(axis=1)
    nospam_lengths = spam.sum(axis=1)

    state['phi_spam'] = (spam.sum(axis=0) + 1) / (np.sum(spam_lengths) + N)
    state['phi_nospam'] = (nospam.sum(axis=0) + 1) / (np.sum(nospam_lengths) + N)
    state['phi'] = spam.shape[0] / (spam.shape[0] + nospam.shape[0])
    ###################
    return state

def nb_test(matrix, state):
    output = np.zeros(matrix.shape[0])
    ###################
    log_phi_spam = np.sum(np.log(state['phi_spam']) * matrix, axis=1)
    log_phi_nospam = np.sum(np.log(state['phi_nospam']) * matrix, axis=1)
    phi = state['phi']

    ratio = np.exp(log_phi_nospam + np.log(1 - phi) - log_phi_spam - np.log(phi))
    probs = 1 / (1 + ratio)

    output[probs > 0.5] = 1
    ###################
    return output


if __name__ == '__main__':
    main()


# b.
tokenlist = np.array(tokenlist)
state = nb_train(trainMatrix, trainCategory)

likely_spam_tokens = np.argsort(state['phi_spam']/state['phi_nospam'])[-5:]
print(tokenlist[likely_spam_tokens])


# c.
train_sizes = np.array([50, 100, 200, 400, 800, 1400])

errors = np.ones(train_sizes.shape)
for i, size in enumerate(train_sizes):
    trainMatrix, tokenlist, trainCategory = readMatrix('./data/MATRIX.TRAIN.' + str(size))
    testMatrix, tokenlist, testCategory = readMatrix('./data/MATRIX.TEST')

    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)

    errors[i] = evaluate(output, testCategory)

plt.plot(train_sizes, errors*100)
plt.xlabel('Training Size')
plt.ylabel('Test Error (%)')
plt.savefig("Q6_c.png")
plt.show()


# d.
tau = 8.


def svm_readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    category = (np.array(Y) * 2) - 1
    return matrix, tokens, category


def svm_train(matrix, category):
    state = {}
    M, N = matrix.shape
    #####################
    Y = category
    matrix = 1. * (matrix > 0)
    squared = np.sum(matrix * matrix, axis=1)
    gram = matrix.dot(matrix.T)
    K = np.exp(-(squared.reshape((1, -1)) + squared.reshape((-1, 1)) - 2 * gram) / (2 * (tau ** 2)) )

    alpha = np.zeros(M)
    alpha_avg = np.zeros(M)
    L = 1. / (64 * M)
    outer_loops = 40

    alpha_avg
    for ii in xrange(outer_loops * M):
        i = int(np.random.rand() * M)
        margin = Y[i] * np.dot(K[i, :], alpha)
        grad = M * L * K[:, i] * alpha[i]
        if (margin < 1):
            grad -=  Y[i] * K[:, i]
        alpha -=  grad / np.sqrt(ii + 1)
        alpha_avg += alpha

    alpha_avg /= (ii + 1) * M

    state['alpha'] = alpha
    state['alpha_avg'] = alpha_avg
    state['Xtrain'] = matrix
    state['Sqtrain'] = squared
    ####################
    return state


def svm_test(matrix, state):
    M, N = matrix.shape
    output = np.zeros(M)
    ###################
    Xtrain = state['Xtrain']
    Sqtrain = state['Sqtrain']
    matrix = 1. * (matrix > 0)
    squared = np.sum(matrix * matrix, axis=1)
    gram = matrix.dot(Xtrain.T)
    K = np.exp(-(squared.reshape((-1, 1)) + Sqtrain.reshape((1, -1)) - 2 * gram) / (2 * (tau ** 2)))
    alpha_avg = state['alpha_avg']
    preds = K.dot(alpha_avg)
    output = np.sign(preds)
    ###################
    return output


def svm_evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print('Error: %1.4f' % error)
    return error


def main():
    trainMatrix, tokenlist, trainCategory = svm_readMatrix('./data/MATRIX.TRAIN.400')
    testMatrix, tokenlist, testCategory = svm_readMatrix('./data/MATRIX.TEST')

    state = svm_train(trainMatrix, trainCategory)
    output = svm_test(testMatrix, state)

    evaluate(output, testCategory)
    return


if __name__ == '__main__':
    main()


errors_svm = np.ones(train_sizes.shape)
testMatrix, tokenlist, testCategory = svm_readMatrix('./data/MATRIX.TEST')
for i,train_size in enumerate(train_sizes):
    trainMatrix, tokenlist, trainCategory = svm_readMatrix('./data/MATRIX.TRAIN.'+str(train_size))
    state = svm_train(trainMatrix, trainCategory)
    output = svm_test(testMatrix, state)
    errors_svm[i] = svm_evaluate(output, testCategory)

plt.plot(train_sizes, errors*100, label="Naive Bayes")
plt.plot(train_sizes, errors_svm*100, label="SVM")
plt.xlabel('Training Size')
plt.ylabel('Test Error (%)')
plt.legend(loc='upper right')
plt.savefig("Q6_d.png")
plt.show()