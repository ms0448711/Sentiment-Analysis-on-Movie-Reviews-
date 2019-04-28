from keras import optimizers
from keras.models import load_model
from keras.models import Sequential, Model
from keras.utils import np_utils, plot_model
from keras.layers import *#Dense, Activation, Dropout,Embedding,Bidirectional,LSTM ,Conv1D, MaxPooling1D
from keras import optimizers
from keras.callbacks import TensorBoard


    

def train(X_train,Y_train):
    
    # Model
    
    input=Input(shape=(X_train.shape[1],X_train.shape[2],) )
    
    
    
    x=LSTM(8,return_sequences=False)(input)
    output=Dense(5,activation='softmax')(x)
    
    model=Model(input,outputs=output)
    
    
    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    #plot_model( model, to_file='model_layers.png')
    # Start training
    model.fit(x=X_train,y=Y_train,batch_size=64,epochs=5,shuffle=True,validation_split=0.1, callbacks=[TensorBoard(log_dir='./tmp/log')])
    
    model.save("LSTMv9.model")
    
    return model
              
    