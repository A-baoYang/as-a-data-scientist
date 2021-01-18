import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM



def readDataset(filename):
    df = pd.read_csv(filename)
    return df


def buildTrain(train, target_Y='Hypo', pastTime=5, futureTime=1):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-futureTime-pastTime):
        X_train.append(np.array(train.iloc[i:i+pastTime]))
        Y_train.append(np.array(train.iloc[i+pastTime:i+pastTime+futureTime][target_Y]))
        
    return np.array(X_train), np.array(Y_train)


def shuffle(X, Y, set_seed=10):
    np.random.seed(int(set_seed))
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]


def splitData(X, Y, test_rate, val_rate):
    X_train = X[int(X.shape[0]*test_rate):]
    Y_train = Y[int(Y.shape[0]*test_rate):]
    X_test = X[:int(X.shape[0]*test_rate)]
    Y_test = Y[:int(Y.shape[0]*test_rate)]
    X_train = X_train[int(X_train.shape[0]*val_rate):]
    Y_train = Y_train[int(Y_train.shape[0]*val_rate):]
    X_val = X_train[:int(X_train.shape[0]*val_rate)]
    Y_val = Y_train[:int(Y_train.shape[0]*val_rate)]
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def standardize_3darray(X_train, X_val, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(X_train.shape[0]*X_train.shape[1], X_train.shape[2]))
    
    X_train_norm = scaler.transform(
        X_train.reshape(X_train.shape[0]*X_train.shape[1], X_train.shape[2])).reshape(
        X_train.shape[0],X_train.shape[1],X_train.shape[2]
    )
    
    X_test_norm = scaler.transform(
        X_test.reshape(X_test.shape[0]*X_test.shape[1], X_test.shape[2])).reshape(
        X_test.shape[0],X_test.shape[1],X_test.shape[2]
    )
    
    if X_val != None:
    
        X_val_norm = scaler.transform(
            X_val.reshape(X_val.shape[0]*X_val.shape[1], X_val.shape[2])).reshape(
            X_val.shape[0],X_val.shape[1],X_val.shape[2]
        )
    
        return X_train_norm, X_val_norm, X_test_norm
    
    else:
        
        return X_train_norm, X_test_norm
        


def imbalance_sampling(X_train, Y_train, sample_ratio):
    index_sampled_Y_train_0 = random.sample([i for i, j in enumerate(Y_train) if j == [0]], round(Y_train.tolist().count([1])*sample_ratio))  # sample same number of Y=1 from indexes of Y=0
    print(f'num of Y label = 0 : {len([i for i, j in enumerate(Y_train) if j == [0]])}')
    index_Y_train_1 = [i for i, j in enumerate(Y_train) if j == [1]]
    print(f'num of Y label = 1 : {len(index_Y_train_1)}')
    return X_train[index_sampled_Y_train_0 + index_Y_train_1], Y_train[index_sampled_Y_train_0 + index_Y_train_1]
    

def buildManyToOneModel(shape):
    model = Sequential()
    model.add(LSTM(16, return_sequences=True, input_length=shape[1], input_dim=shape[2]))
    model.add(LSTM(8))
    # output shape: (1, 1)
    model.add(Dense(1))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model



def plotmodelrecords(history, val_set=True, accuracy_plot=True, loss_plot=True):
    
    if accuracy_plot == True:
        #### Accuracy
        plt.plot(history.history['accuracy'])
        if val_set == True:
            plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='best')
        plt.show()
        
    else:
        print('set `accuracy_plot=True` to see the trends of accuracy.')
    
    if loss_plot == True:
        #### Loss
        plt.plot(history.history['loss'])
        if val_set == True:
            plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='best')
        plt.show()

    else:
        print('set `loss_plot=True` to see the trends of loss.')

    