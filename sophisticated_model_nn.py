from keras.layers.wrappers import Bidirectional
from keras.layers import Embedding
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
import numpy as np

def train_neural_model(x_vocab,x_train_seq,y_train_seq):

    input = Input(shape=(45,))
    # build neural model
    embedding = Embedding(len(x_vocab) + 1, 50, input_length=45)(input)
    lstm1 = Bidirectional(LSTM(units=128,
                               dropout=0.2,
                               recurrent_dropout=0.2,
                               return_sequences=False),
                          merge_mode='concat')(embedding)
    output = Dense(45,activation='softmax')(lstm1)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    #train the model
    model.fit(x_train_seq, y_train_seq, epochs=10, verbose=1,validation_split=0.2)
    return model


#evaluation of the model
def neural_model_eval(model,x_test_seq,y_test_seq):
    acc = model.evaluate(x_test_seq, y_test_seq)
    print("Accuracy of the neural model is:",acc[1]*100)
    predicted = model.predict(x_test_seq)
    print("See predicted indices",[np.argmax(y) for y in predicted])
    print("Compare it to the actual label indices",[np.argmax(y) for y in y_test_seq])

