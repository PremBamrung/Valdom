import neural_network
import numpy as np 


def run():
    print(f'{"":-<20} Loading data {"":-<20}')
    X_train = np.load("train-images.npy").reshape(-1,28*28)
    Y_train = np.load("train-labels.npy")
    X_test = np.load("t10k-images.npy").reshape(-1,28*28)
    Y_test = np.load("t10k-labels.npy")
    print(f'{"":-<20} Data loaded {"":-<20}')

    model=neural_network.NN(X_train,X_test,Y_train,Y_test,neurons=512,epochs=5,learning_rate=1e-3,dropout=0.2)

    print(f'{"":-<20} Training model {"":-<20}')
    
    model.fit()
    print(f'{"":-<20} Model trained {"":-<20}')

    print(f'{"":-<20} Accuracy curve {"":-<20}')
    model.accuracy_curve()

    print(f'{"":-<20} Loss curve {"":-<20}')
    model.loss_curve()

    print(f'{"":-<20} Saving model {"":-<20}')
    model.save("toto.sav")

    print(f'{"":-<20} Loading model {"":-<20}')
    carotte=neural_network.load_model("toto.sav")

    print(f'{"":-<20} Accuracy curve {"":-<20}')
    carotte.accuracy_curve()

    print(f'{"":-<20} Image prediction {"":-<20}')
    carotte.img_pred("test.jpg")

run()