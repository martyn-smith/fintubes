"""
Keras regression and utility functions.
"""

#suppress tensorflow warning messages for failure to detect CUDA.
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #FATAL only
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras import backend as K
from fintubes import *

def keras_optimise(fintubes, model):
    pass

def regress_NN(fintubes, plot):
    """
    Implements a neural-net based regressor.
    """

    def coeff_determination(y_true, y_pred):
        SS_res =  K.sum(K.square( y_true-y_pred )) 
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )

    # A batch size of approximately 2* sqrt samples gives more stable learning, 
    # and it's not like we're low on memory
    batch_size = 128
    # minimum of 50 to get convergence, up to 200 is tolerable time-wise
    epochs = 200
    # use lower rates for exponential, higher for tanh/elu
    opt = Adam(learning_rate = 0.05)

    #patch for last-minute addition of NG as a var - coerce it to float
    fintubes.NG = fintubes.NG.astype(float)
    x_train, x_test, y_train, y_test = train_test_split(fintubes[FC_vars], fintubes.h, train_size = 0.6)

    layer = Normalization()
    layer.adapt(x_train.to_numpy())

    model = Sequential()
    model.add(layer)
    model.add(Dense(9, activation="exponential", input_shape=(x_train.shape[1],)))
    model.add(Dense(9, activation="elu"))
    #model.add(Dense(3, activation="elu"))
    #model.add(Dense(3, activation="elu"))
    #model.add(Dropout(0.05))
    #model.add(Dense(4, activation="exponential"))
    model.add(Dense(1, activation="elu"))
    model.compile(loss="mae",
                    optimizer=opt,
                    metrics=["mae", "mean_absolute_percentage_error", coeff_determination])
    history = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(history.history["mean_absolute_percentage_error"])
        plt.xlabel("epoch")
        plt.ylabel("percentage prediction error")
        plt.show()
        plot_parity(model, x_train, x_test, y_train, y_test)

    print(model.summary())
    print('Test mape, R-squared', score[2], score[3])

    return model