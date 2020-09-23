import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import joblib
import cv2

plt.rcParams['figure.figsize'] = [17, 10]


def load_model(x):
    return joblib.load(x)


def relu(x):
    """
    Relu activation function 
    return : relu output (max(0,x))
    """
    return np.maximum(0, x)


def feedforward(X, W, b, dropout=2):
    """
    1 feedforward
    return : output layer 
    """
    if dropout < 2:
        mask = (np.random.rand(W.shape[0], W.shape[1]) > dropout)
        W = W * mask
        return np.dot(X, W) + b
    else:
        return np.dot(X, W) + b


def get_accuracy(scores, y):
    """
    Compute accuracy after feedforward
    return : accuracy (0 to 1)
    """
    predict_class = np.argmax(scores, axis=1)
    return np.mean(predict_class == y)


def softmax(scores, eps):
    """
    Compute class probabilities by softmax
    return : sofwtmax output (probabilities)
    """
    exp = np.exp(scores)
    probs = exp / np.sum(exp, axis=1, keepdims=True) + eps
    return probs


def cross_entropy(probs, y, num_examples, W1, W2, reg):
    """
    Compute the loss: average cross-entropy loss and regularization
    return : cross_entropy loss
    """
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs) / num_examples
    reg_loss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)
    loss = data_loss + reg_loss
    return loss


def grad(probs, y, num_examples):
    """
    Compute the gradient on scores
    return : dscores
    """
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples
    return dscores


def backpropagation(hidden_layer, dscores, W2, W, X, reg):
    """
    backpropate the gradient to the parameters
    return : dW,db,dW2,db2
    """
    # first backprop into parameters W2 and b2
    dW2 = feedforward(hidden_layer.T, dscores, b=0)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    # next backprop into hidden layer
    dhidden = feedforward(dscores, W2.T, b=0)
    # backprop the ReLU non-linearity
    dhidden[hidden_layer <= 0] = 0
    # finally into W,b
    dW = feedforward(X.T, dhidden, b=0)
    db = np.sum(dhidden, axis=0, keepdims=True)

    # add regularization gradient contribution
    dW2 += reg * W2
    dW += reg * W

    return dW, db, dW2, db2


def updates(W, b, W2, b2, dW, db, dW2, db2, learning_rate):
    """
    updates the weights 
    return : W,b,W2,b2
    """
    W += -learning_rate * dW
    b += -learning_rate * db
    W2 += -learning_rate * dW2
    b2 += -learning_rate * db2

    return W, b, W2, b2


def invert(img):
    output = -img + 255
    return output


class NN():
    def __init__(self, X_train, X_test, Y_train, Y_test, neurons, epochs, learning_rate, dropout=2, reg=1e-3, seed=0):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.neurons = neurons
        self.input_dim = 28 * 28
        self.ouput_dim = 10
        self.W = 0.01 * np.random.randn(self.input_dim, self.neurons)
        self.b = np.zeros((1, self.neurons))
        self.W2 = 0.01 * np.random.randn(self.neurons, self.ouput_dim)
        self.b2 = np.zeros((1, self.ouput_dim))
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.reg = reg
        self.train_loss = np.zeros(self.epochs)
        self.train_accuracy = np.zeros(self.epochs)
        self.test_loss = np.zeros(self.epochs)
        self.test_accuracy = np.zeros(self.epochs)
        self.num_examples = self.X_train.shape[0]
        self.num_examples_test = self.X_test.shape[0]
        self.eps = 1e-8
        self.seed = seed

    def predict(self, X):
        """
        Prediction
        return : predicted_class,probability
        """

        hidden_layer = relu(feedforward(X, self.W, self.b))
        scores = softmax(feedforward(hidden_layer, self.W2, self.b2), self.eps)
        predicted_class = np.argmax(scores, axis=1)
        probability = np.max(scores)

        return predicted_class, probability

    def img_pred(self, X):
        img = invert(cv2.imread(X, 0))
    #     img = img/255
        plt.imshow(img, cmap="gray")
        img = img.reshape(28 * 28)
        return self.predict(img)

    def predict_time(known_epoch, known_min, known_sec, wanted_epoch):
        total_sec = known_sec + known_min * 60
        ratio = total_sec / known_epoch
        secondes = wanted_epoch * ratio
        minutes = secondes // 60
        secondes = secondes % 60
        return f'{int(minutes)} min and {int(secondes)} sec for {wanted_epoch} epochs'

    def loss_curve(self, return_fig=False):
        fig1, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(range(self.epochs), self.train_loss, label="train loss")
        ax1.plot(range(self.epochs), self.test_loss,
                 label="test loss", color='red')
        ax2.plot(range(self.epochs), np.abs(self.train_loss - self.test_loss),
                 label="Loss diff", color='green')

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('train loss')
        ax1.set_ylabel('test loss')
        ax2.set_ylabel('Loss diff')
        ax1.set_yscale('log')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc=0)
        if return_fig:
            return fig1
        else:
            return

    def accuracy_curve(self, return_fig=False):
        fig1, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(range(self.epochs), self.train_accuracy,
                 label="Train accuracy")
        ax1.plot(range(self.epochs), self.test_accuracy,
                 label="Test accuracy", color='red')
        ax2.plot(range(self.epochs), np.abs(self.train_accuracy - self.test_accuracy),
                 label="Accuracy diff", color='green')

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Train accuracy')
        ax1.set_ylabel('Test accuracy')
        ax2.set_ylabel('Accuracy diff')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc=0)
        if return_fig:
            return fig1
        else:
            return

    def save(self, name):
        return joblib.dump(self, name)

    def fit(self):
        np.random.seed(self.seed)
        for i in tqdm(range(self.epochs)):

            # 1st layer + relu
            hidden_layer = relu(feedforward(
                self.X_train, self.W, self.b, self.dropout))

            # 2nd layer
            scores = feedforward(hidden_layer, self.W2, self.b2, self.dropout)

            # softmax
            probs = softmax(scores, self.eps)

            # crossentropy loss
            loss = cross_entropy(probs, self.Y_train,
                                 self.num_examples, self.W, self.W2, self.reg)

            # logging loss
            self.train_loss[i] = loss

            # logging accuracy
            self.train_accuracy[i] = get_accuracy(scores, self.Y_train)

            # test set logging
            hidden_layer_test = relu(feedforward(self.X_test, self.W, self.b))
            probs_test = softmax(feedforward(
                hidden_layer_test, self.W2, self.b2), self.eps)
            self.test_accuracy[i] = get_accuracy(probs_test, self.Y_test)

            # probs_test = softmax(scores_test, self.eps)
            loss_test = cross_entropy(
                probs_test, self.Y_test, self.num_examples_test, self.W, self.W2, self.reg)
            self.test_loss[i] = loss_test

            # gradient
            dscores = grad(probs, self.Y_train, self.num_examples)

            # backpropagation
            dW, db, dW2, db2 = backpropagation(
                hidden_layer, dscores, self.W2, self.W, self.X_train, self.reg)

            # updates
            self.W, self.b, self.W2, self.b2 = updates(
                self.W, self.b, self.W2, self.b2, dW, db, dW2, db2, self.learning_rate)

            if i % (self.epochs / 20) == 0:
                print(
                    f"Epoch {i: <4}  train_loss : {round(self.train_loss[i],4): <9}  test_loss : {round(self.test_loss[i],4): <9}  train_accuracy : {round(self.train_accuracy[i],4): <9}  test_accuracy : {round(self.test_accuracy[i],4): <9}")
