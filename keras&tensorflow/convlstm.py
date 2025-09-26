from keras.models import load_model, Model, Sequential
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, LSTM
import data_handler as data
import numpy as np
import os


def create_model(shape, num_classes):
    model = Sequential()
    model.add( Conv1D(128, 3, padding='same', activation='relu', input_shape=shape) )
    model.add( MaxPooling1D(2) )

    model.add( Conv1D(256, 3, padding='same', activation='relu') )
    model.add( MaxPooling1D(2) )
    
    model.add( LSTM(128, dropout=0.2, activation = 'relu') )

    model.add( Dense(num_classes, activation='softmax') )
    return model



if __name__ == '__main__':
    batch_size   = 32
    timesteps    = 60
    num_features = 19
    accuracies   = list()    
    # 10-fold
    #for fold in range(10):
    for fold in range(1):
        print('initializing fold', fold)
        model = create_model( shape = (timesteps, num_features),
                              num_classes = 20 )
        X_train, y_train, X_test, y_test = data.load_multimodal ( fold+1,
                                                                  window_size = timesteps)
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit( X_train, y_train,
                   batch_size = 32,
                   epochs=150 )
        loss, acc = model.evaluate( X_test, y_test )
        accuracies.append(acc)
        print("Model acc:", acc, "loss:", loss)
    print('mean acc:', sum(accuracies)/len(accuracies))

