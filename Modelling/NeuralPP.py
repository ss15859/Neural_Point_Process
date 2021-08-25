import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


class NPP():
    
    def __init__(self,time_step,size_rnn,size_nn,size_layer_chfn,size_layer_cmfn):
        self.time_step = time_step
        self.size_rnn = size_rnn
        self.size_nn = size_nn
        self.size_layer_chfn = size_layer_chfn
        self.size_layer_cmfn = size_layer_cmfn
        
        
        
    def set_train_data(self,times,mags):
        ## format the input data
        
        self.T_train=times
        self.M_train=mags
        
        # remove first magnitude since our input is time intervals
        
        dM_train = np.delete(mags,0)

        dT_train = np.ediff1d(times) # transform a series of timestamps to a series of interevent intervals: T_train -> dT_train
        n = dT_train.shape[0]
        n2 = dM_train.shape[0]
        
        # creates a rolling matrix that shifts along one 1 input every column
        
        input_RNN_times = np.array( [ dT_train[i:i+self.time_step] for i in range(n-self.time_step) ]).reshape(n-self.time_step,self.time_step,1)
        input_RNN_mags = np.array( [ dM_train[i:i+self.time_step] for i in range(n2-self.time_step) ]).reshape(n2-self.time_step,self.time_step,1)
        
        self.input_RNN = np.concatenate((input_RNN_times,input_RNN_mags),axis=2)
        self.input_CHFN = dT_train[-n+self.time_step:].reshape(n-self.time_step,1)
        self.input_CMFN =dM_train[-n+self.time_step:].reshape(n-self.time_step,1)
        
        
        return self
        
    def set_model(self,lam):
        
        ## mean and std of the log of the inter-event interval and magnitudes, which will be used for the data standardization
        mu = np.log(np.ediff1d(self.T_train)).mean()
        sigma = np.log(np.ediff1d(self.T_train)).std()

        mu1 = np.log(self.M_train).mean()
        sigma1 = np.log(self.M_train).std()

        ## kernel initializer for positive weights
        def abs_glorot_uniform(shape, dtype=None, partition_info=None): 
            return K.abs(keras.initializers.glorot_uniform(seed=None)(shape,dtype=dtype))

        ## Inputs 

        event_history = layers.Input(shape=(self.time_step,2))
        elapsed_time = layers.Input(shape=(1,)) # input to cumulative hazard function network (the elapsed time from the most recent event)
        current_mag = layers.Input(shape=(1,)) # input to cumulative magnitude function


        ## log-transformation and standardization

        elapsed_time_nmlz = layers.Lambda(lambda x: (K.log(x)-mu)/sigma )(elapsed_time) 

        numpyA = np.array([[1/sigma,0],[0,1/sigma1]])

        def multA(x,numpyA):
            A = K.constant(numpyA)

            return K.dot(x,A)

        event_history_nmlz = layers.Lambda(lambda x: multA(K.log(x)-[mu,mu1],numpyA))(event_history)
        current_mag_nmlz = layers.Lambda(lambda x: (K.log(x)-mu1)/sigma1 )(current_mag)

        ## RNN
        output_rnn = layers.SimpleRNN(self.size_rnn,input_shape=(self.time_step,2),activation='tanh')(event_history_nmlz)

        ## the first hidden layer in the cummulative hazard function network
        hidden_tau = layers.Dense(self.size_nn,kernel_initializer=abs_glorot_uniform,kernel_constraint=keras.constraints.NonNeg(),use_bias=False,kernel_regularizer=regularizers.l2(lam))(elapsed_time_nmlz) # elapsed time -> the 1st hidden layer, positive weights
        hidden_rnn = layers.Dense(self.size_nn,kernel_regularizer=regularizers.l2(lam))(output_rnn) # rnn output -> the 1st hidden layer
        hidden = layers.Lambda(lambda inputs: K.tanh(inputs[0]+inputs[1]) )([hidden_tau,hidden_rnn])

        ## the second and higher hidden layers
        for i in range(self.size_layer_chfn-1):
            hidden = layers.Dense(self.size_nn,activation='tanh',kernel_initializer=abs_glorot_uniform,kernel_constraint=keras.constraints.NonNeg(),kernel_regularizer=regularizers.l2(lam))(hidden) # positive weights

        ## the first hidden layer in the cummulative magnitude function network
        hidden_mu = layers.Dense(self.size_nn,kernel_initializer=abs_glorot_uniform,kernel_constraint=keras.constraints.NonNeg(),use_bias=False)(current_mag_nmlz) # elapsed time -> the 1st hidden layer, positive weights
        hidden_rnn_mag = layers.Dense(self.size_nn)(output_rnn) # rnn output -> the 1st hidden layer
        hidden_mag = layers.Lambda(lambda inputs: K.tanh(inputs[0]+inputs[1]) )([hidden_mu,hidden_rnn_mag,hidden_tau])

        ## the second and higher hidden layers
        for i in range(self.size_layer_cmfn-1):
            hidden_mag = layers.Dense(self.size_nn,activation='tanh',kernel_initializer=abs_glorot_uniform,kernel_constraint=keras.constraints.NonNeg())(hidden_mag) # positive weights



        ## Outputs
        Int_l = layers.Dense(1, activation='softplus',kernel_initializer=abs_glorot_uniform, kernel_constraint=keras.constraints.NonNeg() )(hidden) # cumulative hazard function, positive weights
        l = layers.Lambda( lambda inputs: K.gradients(inputs[0],inputs[1])[0] )([Int_l,elapsed_time]) # hazard function
        Int_l_mag = layers.Dense(1, activation='sigmoid',kernel_initializer=abs_glorot_uniform, kernel_constraint=keras.constraints.NonNeg() )(hidden_mag) # cumulative hazard function, positive weights
        l_mag= layers.Lambda( lambda inputs: K.gradients(inputs[0],inputs[1])[0] )([Int_l_mag,current_mag]) # hazard function

        ## define model
        self.model = Model(inputs=[event_history,elapsed_time,current_mag],outputs=[l,Int_l,l_mag,Int_l_mag])
        self.model.add_loss( -K.mean( K.log( 1e-10 + l ) - Int_l + K.log(1e-10 + l_mag )) ) # set loss function to be the negative log-likelihood function
        #+K.log(1e-10 + l_mag )
        return self

    
    def compile(self,lr=1e-3):
        self.model.compile(keras.optimizers.Adam(lr=lr))
        return self
    
    ## Class for early stopping based on validation loss
    
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
                
            if self.best_val_loss + 1 < val_loss: 
                    self.model.stop_training = True
                
#             if (epoch+1) % 5 == 0:
                
#                 #print('epoch: %d, current_val_loss: %f, min_val_loss: %f' % (epoch+1,val_loss,self.best_val_loss) )
                
#                 if (epoch+1) >= 15:
#                     if self.best_val_loss > self.history_val_loss[:-5].min() - 0.1: 
#                         self.model.stop_training = True
                        
        def on_train_end(self,logs=None):
            self.model.set_weights(self.best_weights)
            #print('set optimal weights')
    

    def fit_eval(self,epochs=100,batch_size=256):
        
        es = NPP.CustomEarlyStopping()
        self.model.fit([self.input_RNN,self.input_CHFN,self.input_CMFN],epochs=epochs,batch_size=batch_size,validation_split=0.2,callbacks=[es])
        
        
        return self


    def save_weights(self,path):
        self.model.save_weights('./checkpoints/'+ path)
        return self

    def load_weights(self,path):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.model.load_weights('./checkpoints/'+ path)
        return self 



        ## repeat of above function, now for the test data
    def set_test_data(self,times,mags):
        
        ## format the input data
        dM_test = np.delete(mags,0)
        dT_test = np.ediff1d(times)+1e-9 # transform a series of timestamps to a series of interevent intervals: T_train -> dT_train
        n = dT_test.shape[0]
        n2 = dM_test.shape[0]
        
        input_RNN_times = np.array( [ dT_test[i:i+self.time_step] for i in range(n-self.time_step) ]).reshape(n-self.time_step,self.time_step,1)
        input_RNN_mags = np.array( [ dM_test[i:i+self.time_step] for i in range(n2-self.time_step) ]).reshape(n2-self.time_step,self.time_step,1)
        
        self.input_RNN_test = np.concatenate((input_RNN_times,input_RNN_mags),axis=2)
        self.input_CHFN_test = dT_test[-n+self.time_step:].reshape(n-self.time_step,1)
        self.input_CMFN_test =dM_test[-n+self.time_step:].reshape(n-self.time_step,1)
        
        return self
        
        
        # predict and calculate the log-likelihood
        
    def predict_eval(self):
        
        [self.lam,self.Int_lam,self.mag_dist,self.Int_mag_dist] = self.model.predict([self.input_RNN_test,self.input_CHFN_test,self.input_CMFN_test],batch_size=self.input_RNN_test.shape[0])
        self.LL = np.log(self.lam+1e-10) - self.Int_lam  
        self.LLmag = np.log(1e-10 + self.mag_dist )# log-liklihood
        
        return self
    
    def eval_train_data(self):
        [self.lam_train,self.Int_lam_train,self.mag_dist_train,self.Int_mag_dist_train] = self.model.predict([self.input_RNN,self.input_CHFN,self.input_CMFN],batch_size=self.input_RNN.shape[0])
        
        return self
    
    def summary(self):
        return self.model.summary()


    def magdistfunc(self,x,hist,new_time):

            T, M = hist

            dM_test = np.delete(M,0)
            dT_test = np.ediff1d(T) # transform a series of timestamps to a series of interevent intervals: T_train -> dT_train
            dT_test=np.append(dT_test,new_time)
            dM_test=np.append(dM_test,x)
            n = dT_test.shape[0]
            n2 = dM_test.shape[0]
            input_RNN_times = np.array( [ dT_test[i:i+self.time_step] for i in range(n-self.time_step) ]).reshape(n-self.time_step,self.time_step,1)
            input_RNN_mags = np.array( [ dM_test[i:i+self.time_step] for i in range(n2-self.time_step) ]).reshape(n2-self.time_step,self.time_step,1)
            input_RNN_test = np.concatenate((input_RNN_times,input_RNN_mags),axis=2)
            input_CHFN_test = dT_test[-n+self.time_step:].reshape(n-self.time_step,1)
            input_CMFN_test =dM_test[-n+self.time_step:].reshape(n-self.time_step,1)


            Int_m_test = self.model.predict([input_RNN_test,input_CHFN_test,input_CMFN_test],batch_size=input_RNN_test.shape[0])[3]

            return Int_m_test[-1]

    def magdensfunc(self,x,hist,new_time):

            T, M = hist

            dM_test = np.delete(M,0)
            dT_test = np.ediff1d(T) # transform a series of timestamps to a series of interevent intervals: T_train -> dT_train
            dT_test=np.append(dT_test,new_time)
            dM_test=np.append(dM_test,x)
            n = dT_test.shape[0]
            n2 = dM_test.shape[0]
            input_RNN_times = np.array( [ dT_test[i:i+self.time_step] for i in range(n-self.time_step) ]).reshape(n-self.time_step,self.time_step,1)
            input_RNN_mags = np.array( [ dM_test[i:i+self.time_step] for i in range(n2-self.time_step) ]).reshape(n2-self.time_step,self.time_step,1)
            input_RNN_test = np.concatenate((input_RNN_times,input_RNN_mags),axis=2)
            input_CHFN_test = dT_test[-n+self.time_step:].reshape(n-self.time_step,1)
            input_CMFN_test =dM_test[-n+self.time_step:].reshape(n-self.time_step,1)


            Int_m_test = self.model.predict([input_RNN_test,input_CHFN_test,input_CMFN_test],batch_size=input_RNN_test.shape[0])[2]

            return Int_m_test[-1]

    def distfunc(self,x,hist):

            T,M=hist
            dM_test = np.delete(M,0)
            dT_test = np.ediff1d(T) # transform a series of timestamps to a series of interevent intervals: T_train -> dT_train
            dT_test=np.append(dT_test,x)
            dM_test=np.append(dM_test,M[-1])
            n = dT_test.shape[0]
            n2 = dM_test.shape[0]
            input_RNN_times = np.array( [ dT_test[i:i+self.time_step] for i in range(n-self.time_step) ]).reshape(n-self.time_step,self.time_step,1)
            input_RNN_mags = np.array( [ dM_test[i:i+self.time_step] for i in range(n2-self.time_step) ]).reshape(n2-self.time_step,self.time_step,1)
            input_RNN_test = np.concatenate((input_RNN_times,input_RNN_mags),axis=2)
            input_CHFN_test = dT_test[-n+self.time_step:].reshape(n-self.time_step,1)
            input_CMFN_test = dM_test[-n+self.time_step:].reshape(n-self.time_step,1)

            int_l_test = self.model.predict([input_RNN_test,input_CHFN_test,input_CMFN_test],batch_size=input_RNN_test.shape[0])[1]

            return 1-np.exp(-int_l_test[-1])
    
    def forecast(self,hist,M0,hours):
        
        

        def predicttime(self,hist):

            u = np.random.uniform()
            x_left = np.ediff1d(hist[0]).mean()*1e-5
            x_right = np.ediff1d(hist[0]).mean()*1e5
            v = 100

            while(abs(v-u)>0.01):
                x_center = np.exp((np.log(x_left)+np.log(x_right))/2)
                v = self.distfunc(x_center,hist)
                x_left = np.where(v<u,x_center,x_left)
                x_right = np.where(v>=u,x_center,x_right)

            tau_pred = x_center # predicted interevent interval


            return float(tau_pred)

        

#         def normmagdistfunc(self,x,hist,new_time):
#             return magdistfunc(self,x,hist,new_time)/magdistfunc(self,100000000,hist,new_time)

        def predict_mag(self,hist,M0,new_time):

            u = np.random.uniform()
            x_left = M0
            x_right = 10
            v = 100


            while(abs(v-u)>0.001):
                x_center = (x_left+x_right)/2
                v = self.magdistfunc(x_center,hist,new_time)
                x_left = np.where(v<u,x_center,x_left)
                x_right = np.where(v>=u,x_center,x_right)

                if x_left == x_right:
                    break
                
            mu_pred = x_center # predicted interevent interval

            return float(mu_pred)    
        
        
        
        
        
        T_testfor=hist[0]
        M_testfor=hist[1]

#         T_start = np.ceil(T_testfor[-1])
        T_start = hours*np.ceil(T_testfor[-1]/hours)

        predictions = []

        while True:
            
            new_pred = predicttime(self,hist)
            new_time = T_testfor[-1]+new_pred

            if new_time-T_start>hours:
                break

#             print(new_time, end='\r')
            if new_time>T_start:
                predictions.append(new_time)

            M_testfor=np.append(M_testfor,predict_mag(self,[T_testfor,M_testfor],M0,new_time))
            T_testfor=np.append(T_testfor,new_time)

        return(len(predictions))
    
    
    def daily_forecast(self,Tdat,Mdat,ndays,repeats,M0,time_step,hours):
    
        Tdat = Tdat-Tdat[0] 
        
        forcastN = np.zeros((ndays,repeats))
        for j in range(repeats):
            print(j,'\r')
            for i in range(ndays):

                if len(Tdat[Tdat<=i*hours])>=time_step+1:
                    print(i,'\r')
                    hist1 = [Tdat[Tdat<=i*hours],Mdat[Tdat<=i*hours]]
                    forcastN[i,j] = self.forecast(hist1,M0 = M0,hours=hours)

        return forcastN

