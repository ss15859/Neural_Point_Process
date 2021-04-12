import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model


## hyper parameters
time_step = 20 # truncation depth of RNN 
size_rnn = 64 # the number of units in the RNN
size_nn = 64 # the nubmer of units in each hidden layer in the cumulative hazard function network
size_layer_chfn = 2 # the number of the hidden layers in the cumulative hazard function network
size_layer_cmfn = 2

T_train=times
M_train=mags


class NPP():
    
    def __init__(self,time_step,size_rnn,size_nn,size_layer_chfn,size_layer_cmfn):
        self.time_step = time_step
        self.size_rnn = size_rnn
        self.size_nn = size_nn
        self.size_layer_chfn = size_layer_chfn
        self.size_layer_cmfn = size_layer_cmfn
        
        
        
    def set_train_data(times,mags):
        ## format the input data
        dM_train = np.delete(mags,0)

        dT_train = np.ediff1d(times) # transform a series of timestamps to a series of interevent intervals: T_train -> dT_train
        n = dT_train.shape[0]
        n2 = dM_train.shape[0]
        input_RNN_times = np.array( [ dT_train[i:i+time_step] for i in range(n-time_step) ]).reshape(n-time_step,time_step,1)
        input_RNN_mags = np.array( [ dM_train[i:i+time_step] for i in range(n2-time_step) ]).reshape(n2-time_step,time_step,1)
        self.input_RNN = np.concatenate((input_RNN_times,input_RNN_mags),axis=2)
        self.input_CHFN = dT_train[-n+time_step:].reshape(n-time_step,1)
        self.input_CMFN =dM_train[-n+time_step:].reshape(n-time_step,1)
        
        
        return self
        
    def set_model(self,times,mags):
        
        ## mean and std of the log of the inter-event interval, which will be used for the data standardization
        mu = np.log(np.ediff1d(times)).mean()
        sigma = np.log(np.ediff1d(times)).std()

        mu1 = np.log(mags).mean()
        sigma1 = np.log(mags).std()

        ## kernel initializer for positive weights
        def abs_glorot_uniform(shape, dtype=None, partition_info=None): 
            return K.abs(keras.initializers.glorot_uniform(seed=None)(shape,dtype=dtype))

        ## Inputs 

        event_history = layers.Input(shape=(self.time_step,2))
        elapsed_time = layers.Input(shape=(1,)) # input to cumulative hazard function network (the elapsed time from the most recent event)
        current_mag = layers.Input(shape=(1,)) 


        ## log-transformation and standardization
        # event_history_nmlz = layers.Lambda(lambda x: (K.log(x)-mu)/sigma )(event_history) ## probs have to do matrix equivalent for this
        elapsed_time_nmlz = layers.Lambda(lambda x: (K.log(x)-mu)/sigma )(elapsed_time) 

        numpyA = np.array([[1/sigma,0],[0,1/sigma1]])

        def multA(x,A):
            A = K.constant(numpyA)

            return K.dot(x,A)

        event_history_nmlz = layers.Lambda(lambda x: multA(K.log(x)-[mu,mu1],numpyA))(event_history)
        current_mag_nmlz = layers.Lambda(lambda x: (K.log(x)-mu1)/sigma1 )(current_mag)

        ## RNN
        output_rnn = layers.SimpleRNN(self.size_rnn,input_shape=(self.time_step,2),activation='tanh')(event_history_nmlz)

        ## the first hidden layer in the cummulative hazard function network
        hidden_tau = layers.Dense(self.size_nn,kernel_initializer=abs_glorot_uniform,kernel_constraint=keras.constraints.NonNeg(),use_bias=False)(elapsed_time_nmlz) # elapsed time -> the 1st hidden layer, positive weights
        hidden_rnn = layers.Dense(self.size_nn)(output_rnn) # rnn output -> the 1st hidden layer
        hidden = layers.Lambda(lambda inputs: K.tanh(inputs[0]+inputs[1]) )([hidden_tau,hidden_rnn])

        ## the second and higher hidden layers
        for i in range(self.size_layer_chfn-1):
            hidden = layers.Dense(self.size_nn,activation='tanh',kernel_initializer=abs_glorot_uniform,kernel_constraint=keras.constraints.NonNeg())(hidden) # positive weights

        ## the first hidden layer in the cummulative hazard function network
        hidden_mu = layers.Dense(self.size_nn,kernel_initializer=abs_glorot_uniform,kernel_constraint=keras.constraints.NonNeg(),use_bias=False)(current_mag_nmlz) # elapsed time -> the 1st hidden layer, positive weights
        hidden_rnn_mag = layers.Dense(self.size_nn)(output_rnn) # rnn output -> the 1st hidden layer
        hidden_mag = layers.Lambda(lambda inputs: K.tanh(inputs[0]+inputs[1]) )([hidden_mu,hidden_rnn_mag,hidden_tau])

        ## the second and higher hidden layers
        for i in range(self.size_layer_cmfn-1):
            hidden_mag = layers.Dense(self.size_nn,activation='tanh',kernel_initializer=abs_glorot_uniform,kernel_constraint=keras.constraints.NonNeg())(hidden_mag) # positive weights



        ## Outputs
        Int_l = layers.Dense(1, activation='softplus',kernel_initializer=abs_glorot_uniform, kernel_constraint=keras.constraints.NonNeg() )(hidden) # cumulative hazard function, positive weights
        l = layers.Lambda( lambda inputs: K.gradients(inputs[0],inputs[1])[0] )([Int_l,elapsed_time]) # hazard function
        Int_l_mag = layers.Dense(1, activation='softplus',kernel_initializer=abs_glorot_uniform, kernel_constraint=keras.constraints.NonNeg() )(hidden_mag) # cumulative hazard function, positive weights
        l_mag= layers.Lambda( lambda inputs: K.gradients(inputs[0],inputs[1])[0] )([Int_l_mag,current_mag]) # hazard function

        ## define model
        self.model = Model(inputs=[event_history,elapsed_time,current_mag],outputs=[l,Int_l,l_mag,Int_l_mag])
        self.model.add_loss( -K.mean( K.log( 1e-10 + l )+ K.log(1e-10 + l_mag ) - Int_l ) ) # set loss function to be the negative log-likelihood function
        
        return self

    
    def compile(self,lr=1e-3):
        self.model.compile(keras.optimizers.Adam(lr=lr))
        return self
    
    
    class CustomEarlyStopping(keras.callbacks.Callback):
    
        def __init__(self):
            super(NPP.CustomEarlyStopping, self).__init__()
            self.best_val_loss = 100000
            self.history_val_loss = []
            self.best_weights   = None

        def on_epoch_end(self, epoch, logs=None):
            
            val_loss = logs['val_loss']
            self.history_val_loss = np.append(self.history_val_loss,val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_weights = self.model.get_weights()
                
            if self.best_val_loss + 0.05 < val_loss: 
                    self.model.stop_training = True
                
            if (epoch+1) % 5 == 0:
                
                #print('epoch: %d, current_val_loss: %f, min_val_loss: %f' % (epoch+1,val_loss,self.best_val_loss) )
                
                if (epoch+1) >= 15:
                    if self.best_val_loss > self.history_val_loss[:-5].min() - 0.001: 
                        self.model.stop_training = True
                        
        def on_train_end(self,logs=None):
            self.model.set_weights(self.best_weights)
            #print('set optimal weights')
    

    def fit_eval(self,epochs=100,batch_size=256):
        
        es = NPP.CustomEarlyStopping()
        model.fit([input_RNN,input_CHFN,input_CMFN],epochs=30,batch_size=256,validation_split=0.2,callbacks=[es])
