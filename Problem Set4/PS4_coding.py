# Question 1 Neural Networks: MNIST image classification
import numpy as np
import matplotlib.pyplot as plt


def readData(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y


def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size), labels.astype(int)] = 1
    return one_hot_labels


def softmax(x):
    """
    Compute softmax function for input.
    Use tricks from previous assignment to avoid overflow
    """
    ### YOUR CODE HERE
    c = np.max(x, axis=1, keepdims=True)
    numerator = np.exp(x - c)
    denominator = np.sum(numerator, axis=1, keepdims=True)
    s = numerator / denominator
    ### END YOUR CODE
    return s


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    ### YOUR CODE HERE
    pos_mask = (x >= 0)
    neg_mask = (x < 0)

    z = np.zeros_like(x, dtype=float)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])

    top = np.ones_like(x, dtype=float)
    top[neg_mask] = z[neg_mask]
    s = top / (1 + z)
    ### END YOUR CODE
    return s


def forward_prop(data, labels, params):
    """
    return hidder layer, output(softmax) layer and loss
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    lamb = params['lambda']

    ### YOUR CODE HERE
    z1 = data.dot(W1) + b1
    h = sigmoid(z1)
    z2 = h.dot(W2) + b2
    y = softmax(z2)
    cost = -np.multiply(labels, np.log(y + 1e-16)).sum()
    cost /= data.shape[0]
    ### END YOUR CODE
    return h, y, cost


def backward_prop(data, labels, params):
    """
    return gradient of parameters
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    ### YOUR CODE HERE
    h, y, cost = forward_prop(data, labels, params)
    B = data.shape[0]

    delta1 = (y - labels)
    gradW2 = np.dot(h.T, delta1)
    gradb2 = np.sum(delta1, axis=0, keepdims=True)

    delta2 = np.multiply(np.dot(delta1, W2.T), h * (1 - h))
    gradW1 = np.dot(data.T, delta2)
    gradb1 = np.sum(delta2, axis=0, keepdims=True)

    lamb = params['lambda']
    if lamb > 0:
        gradW2 += lamb * W2
        gradW1 += lamb * W1

    gradW1 /= B
    gradW2 /= B
    gradb1 /= B
    gradb2 /= B
    ### END YOUR CODE

    grad = {}
    grad['W1'] = gradW1
    grad['W2'] = gradW2
    grad['b1'] = gradb1
    grad['b2'] = gradb2

    return grad


def compute_accuracy(output, labels):
    accuracy = (np.argmax(output, axis=1) == np.argmax(labels, axis=1)).sum() * 1. / labels.shape[0]
    return accuracy



def nn_train(trainData, trainLabels, devData, devLabels, lamb):
    (m, n) = trainData.shape
    num_hidden = 300
    learning_rate = 5
    B = 1000
    num_epochs = 30
    params = {}

    ### YOUR CODE HERE
    params['W1'] = np.random.standard_normal((n, num_hidden))
    params['W2'] = np.random.standard_normal((num_hidden, trainLabels.shape[1]))
    params['b1'] = np.zeros((1, num_hidden), dtype=float)
    params['b2'] = np.zeros((1, trainLabels.shape[1]), dtype=float)
    params['lambda'] = lamb

    num_iter = int(m / B)
    tr_loss, tr_metric, dev_loss, dev_metric = [], [], [], []

    for i in range(num_epochs):
        print(i, end=',')
        for j in range(num_iter):
            batch_data = trainData[j * B: (j + 1) * B]
            batch_labels = trainLabels[j * B: (j + 1) * B]
            grad = backward_prop(batch_data, batch_labels, params)
            params['W1'] -= learning_rate * grad['W1']
            params['W2'] -= learning_rate * grad['W2']
            params['b1'] -= learning_rate * grad['b1']
            params['b2'] -= learning_rate * grad['b2']

        train_h, train_y, train_cost = forward_prop(trainData, trainLabels, params)
        tr_loss.append(train_cost)
        tr_metric.append(compute_accuracy(train_y, trainLabels))
        dev_h, dev_y, dev_cost = forward_prop(devData, devLabels, params)
        dev_loss.append(dev_cost)
        dev_metric.append(compute_accuracy(dev_y, devLabels))
    ### END YOUR CODE

    return params, tr_loss, tr_metric, dev_loss, dev_metric


def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy


def prepare_data():
    np.random.seed(100)
    trainData, trainLabels = readData('./data/mnist/images_train.csv', './data/mnist/labels_train.csv')
    trainLabels = one_hot_labels(trainLabels)
    p = np.random.permutation(60000)
    trainData = trainData[p, :]
    trainLabels = trainLabels[p, :]

    devData = trainData[0:10000, :]
    devLabels = trainLabels[0:10000, :]
    trainData = trainData[10000:, :]
    trainLabels = trainLabels[10000:, :]

    mean = np.mean(trainData)
    std = np.std(trainData)
    trainData = (trainData - mean) / std
    devData = (devData - mean) / std

    testData, testLabels = readData('./data/mnist/images_test.csv', './data/mnist/labels_test.csv')
    testLabels = one_hot_labels(testLabels)
    testData = (testData - mean) / std

    return trainData, trainLabels, devData, devLabels, testData, testLabels


trainData, trainLabels, devData, devLabels, testData, testLabels = prepare_data()


# a.
num_epochs = 30
params, tr_loss, tr_metric, dev_loss, dev_metric = nn_train(trainData, trainLabels, devData, devLabels, 0)

xs = np.arange(num_epochs)

fig, axes = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(12, 4))
ax0, ax1 = axes.ravel()

ax0.plot(xs, tr_loss, label='train loss')
ax0.plot(xs, dev_loss, label='dev loss')
ax0.legend()
ax0.set_xlabel("number of epoch")
ax0.set_ylabel("CE loss")

ax1.plot(xs, tr_metric, label="train accuracy")
ax1.plot(xs, dev_metric, label="dev accuracy")
ax1.legend()
ax1.set_xlabel("number of epoch")
ax1.set_ylabel("Accuracy")

plt.savefig("Q1_a.png")
plt.show()


# b.
num_epochs = 30
params, tr_loss, tr_metric, dev_loss, dev_metric = nn_train(trainData, trainLabels, devData, devLabels, 0.0001)

xs = np.arange(num_epochs)

fig, axes = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(12, 4))
ax0, ax1 = axes.ravel()

ax0.plot(xs, tr_loss, label='train loss')
ax0.plot(xs, dev_loss, label='dev loss')
ax0.legend()
ax0.set_xlabel("number of epoch")
ax0.set_ylabel("CE loss")

ax1.plot(xs, tr_metric, label="train accuracy")
ax1.plot(xs, dev_metric, label="dev accuracy")
ax1.legend()
ax1.set_xlabel("number of epoch")
ax1.set_ylabel("Accuracy")

plt.savefig("Q1_b.png")
plt.show()


# c.
accuracy = nn_test(testData, testLabels, params)
print('Test accuracy (without regularization): {0}'.format(accuracy))
