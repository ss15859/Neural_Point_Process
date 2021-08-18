import pandas as pd
import numpy as np
import datetime as dt
import math
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import integrate

from NeuralPP import NPP

from ETAS import marked_ETAS_intensity, marked_likelihood, movingaverage, sGR, bin_times, daily_forecast, likelihood

import ETAS



Amatrice = pd.read_csv('~/PhD/Amatrice_tests/Amatrice_CAT5.v20210504.csv')

Amatrice['datetime'] = pd.to_datetime(Amatrice[['year', 'month', 'day', 'hour', 'minute','second']])
Amatrice['time'] = (Amatrice['datetime']-Amatrice['datetime'][0])/ pd.to_timedelta(1, unit='H')


Amatrice = Amatrice[['time','mw']]
Amatrice = Amatrice.dropna()

timeupto = 4800
timefrom = 0
Mcut = 3.0
M0= 0.3
M0pred = Mcut

time_step=20

sub_catalog = Amatrice[ Amatrice.time<timeupto]
sub_catalog = Amatrice[Amatrice.mw > Mcut]



times = np.array(sub_catalog['time'])
mags = np.array(sub_catalog['mw'])

limit = np.linspace(600,4200,7)

MLE = []
NN = []
poiss=[]
NNmagLik = []
MLEmagLik = []

df = pd.DataFrame(index = ['mu','k0','alpha','c','omega','beta'])

for timeupto in limit:
    
    timeupto = int(timeupto)
    T_train = times[times<timeupto]
    M_train = mags[times<timeupto]

    T_test = times[times>=timeupto]
    M_test = mags[times>=timeupto]
#     T_test = T_test[M_test>=M0pred]
#     M_test = M_test[M_test>=M0pred]

    
    npp1 = NPP(time_step=20,size_rnn=64,size_nn=64,size_layer_chfn=3,size_layer_cmfn=2).set_train_data(T_train,M_train).set_model(0).compile(lr=1e-3).fit_eval(epochs=400,batch_size=256).set_test_data(T_test,M_test).predict_eval()

    LLNN = (npp1.LL[M_test[time_step+1:]>=M0pred]).mean()
    
    NNmag = npp1.LLmag.mean()

    filename = '~/PhD/Amatrice_tests/paramsMcut:'+str(Mcut)+'00000_timefrom:0_timeupto:' + str(timeupto)+ '.csv'

    params = pd.read_csv(filename)


    groundMLE = params.x[0]
    k0MLE= params.x[1]
    alphaMLE = params.x[2]
    cMLE = params.x[3]
    wMLE = params.x[4]
#     betaMLE = params.x[5]
    betaMLE = 1/(Amatrice.mw-M0).mean()
#     betaMLE = 1/(M_test-Mcut).mean()
#     M0 = mags.min()
    tau =0
    
    d = {'mu': groundMLE, 'k0': k0MLE,'a':alphaMLE,'c':cMLE,'omega':wMLE,'tau':tau,'M0':M0,'beta':betaMLE}
    params = pd.DataFrame(data=d,index=[0])
#     groundMLE = 0.0355 /24
#     k0MLE= 0.0936
#     alphaMLE = np.log(10)
#     cMLE = 0.01309*24
#     wMLE = 1.1842
#     betaMLE = np.log(10)
#     M0 = 3
#     tau =0
    
#     df[timeupto]=np.array(params.x)[:6]


    poissMLE = 1/np.ediff1d(T_train[M_train>=M0pred]).mean()
    
    LLpoiss = (len(T_test[M_test>=M0pred])*np.log(poissMLE) + sum(np.log(ETAS.sGR(M_test,betaMLE,M0,M0pred)))- (T_test[-1]-T_test[0])*poissMLE)/len(T_test[M_test>=M0pred])

    LLMLE = likelihood(T_test,M_test,T_test[-1],groundMLE,k0MLE,alphaMLE,M0,M0pred,cMLE,tau,wMLE,betaMLE)/len(T_test[M_test>=M0pred])
    
    MLEmag = (np.log(sGR(M_test,betaMLE,M0,Mcut)*np.exp((Mcut-M0)*betaMLE))).mean()
    
#     MLEmag = (np.log(sGR(M_test,betaMLE,Mcut,Mcut))).mean()

    result1 = pd.DataFrame([LLNN-LLpoiss,LLMLE-LLpoiss,LLpoiss-LLpoiss],columns=['log-likelihood gain'],index=['NN','ETAS','Poison'])
    
    result1

    NNgain = LLNN-LLpoiss

    MLEgain = LLMLE - LLpoiss
    
    poiss = np.append(poiss,LLpoiss)

    NN = np.append(NN,NNgain)

    MLE = np.append(MLE,MLEgain)
    
    NNmagLik = np.append(NNmagLik,NNmag)
    
    MLEmagLik = np.append(MLEmagLik,MLEmag)
    
    plt.scatter(M_test[21:],np.exp(npp1.LLmag),label='NN')
    plt.scatter(M_test,sGR(M_test,betaMLE,M0,Mcut)*np.exp((Mcut-M0)*betaMLE),label='ETAS')
#     plt.scatter(M_test,sGR(M_test,betaMLE,Mcut,Mcut),label='ETAS')
    plt.legend()
    plt.show()
    
    plt.scatter(M_test[21:],np.log(sGR(M_test[21:],betaMLE,M0,Mcut)*np.exp((Mcut-M0)*betaMLE))-(npp1.LLmag).reshape(-1))
#     plt.scatter(M_test[21:],np.log(sGR(M_test[21:],betaMLE,Mcut,Mcut)-(npp1.LLmag).reshape(-1)))
    plt.show()
    
    print((np.log(sGR(M_test[21:],betaMLE,M0,Mcut)*np.exp((Mcut-M0)*betaMLE))-(npp1.LLmag).reshape(-1)).mean())
#     print((np.log(sGR(M_test[21:],betaMLE,Mcut,Mcut)-(npp1.LLmag).reshape(-1)).mean()))
    
#     index = np.where(T_test>4000)[0][0]
    
    y = range(1,len(npp1.Int_lam)+1)
    t = np.cumsum(npp1.Int_lam)

    lam = marked_ETAS_intensity(T_test,M_test,groundMLE,k0MLE,alphaMLE,M0,Mcut,cMLE,tau,wMLE)

    lam_int = integrate.cumtrapz(lam, T_test, initial=0)
    yETAS = range(1,len(lam_int)+1)

    plt.plot(t,y,color='black',label = 'NN')
    plt.plot(lam_int,yETAS,color='green',label = 'ETAS')
#     if(index!=0):
#         plt.axvline(x=t[index], linestyle='--')
    plt.plot(np.linspace(0,t[-1]),np.linspace(0,t[-1]),color = 'r',label = 'y = x')
    plt.xlabel('Transformed time')
    plt.ylabel('Cumulative number')
    plt.legend()
    plt.title('Trained on data up to ' + str(timeupto) + ' hours from start')
    plt.savefig('residual_'+str(timeupto)+ str(Mcut)+'forecasting.png')
    plt.show()
    
    npp1.eval_train_data()
    
# #     index = np.where(T_train>4000)[0]
    
    y = range(1,len(npp1.Int_lam_train)+1)
    t = np.cumsum(npp1.Int_lam_train)

    lam = marked_ETAS_intensity(T_train,M_train,groundMLE,k0MLE,alphaMLE,M0,Mcut,cMLE,tau,wMLE)

    lam_int = integrate.cumtrapz(lam, T_train, initial=0)
    yETAS = range(1,len(lam_int)+1)

    plt.plot(t,y,color='black',label = 'NN')
    plt.plot(lam_int,yETAS,color='green',label = 'ETAS')
#     if(index.size!=0):
#         plt.axvline(x=t[index[0]], linestyle='--')
    plt.plot(np.linspace(0,t[-1]),np.linspace(0,t[-1]),color = 'r',label = 'y = x')
    plt.xlabel('Transformed time')
    plt.ylabel('Cumulative number')
    plt.legend()
    plt.title('Trained on data up to ' + str(timeupto) + ' hours from start')
    plt.savefig('residual_'+str(timeupto)+ str(Mcut)+'training.png')
    plt.show()
    
    
#     forcastETAS = daily_forecast(T_test,M_test,ndays=int(np.floor((T_test[-1]-T_test[0])/24)),repeats=3,params=params,hours=24)
#     aveETAS = np.mean(forcastETAS,axis=1)
    
#     true = bin_times(T_test,hours=24)
    
#     plt.scatter(range(len(true)),np.log(true))
#     plt.scatter(range(len(aveETAS)),np.log(aveETAS))
#     plt.show()

    
#     forcastN = npp1.daily_forecast(T_test,M_test,ndays=int(np.floor((T_test[-1]-T_test[0])/24)),repeats=1,M0=M0,time_step=time_step,hours=24)
    
#     # ave = np.mean(forcastN,axis=1)
#     ave = forcastN
    
#     plt.scatter(range(len(true)),np.log(true))
#     plt.scatter(range(len(aveETAS)),np.log(aveETAS))
#     plt.show()
    
plt.plot(limit,NN,label='NN')
plt.plot(limit,MLE,label='ETAS')
plt.legend()
plt.xlabel('time from start / hours')
plt.ylabel('log-likelihood gain')
plt.title('log-likelihood gain with increasing training size for M0_train = ' +str(Mcut))
plt.savefig('loglik_gain_'+str(Mcut) +'.png')
plt.show()

plt.plot(limit,NNmagLik,label='NN')
plt.plot(limit,MLEmagLik,label='ETAS')
plt.legend()
plt.xlabel('time from start / hours')
plt.ylabel('log-likelihood gain')
plt.title('Magnitude log-likelihood with increasing training size for M0_train = ' +str(Mcut))
plt.savefig('mag_loglik_gain_'+str(Mcut) +'.png')
plt.show()

plt.plot(limit,NN+poiss,label='NN')
plt.plot(limit,MLE+poiss,label='ETAS')
plt.legend()
plt.xlabel('time from start / hours')
plt.ylabel('log-likelihood')
plt.title('log-likelihood with increasing training size for M0_train = 2.5, M0_test = 3.0')
# plt.savefig('raw_loglik_3.0,3.5.png')
plt.show()



####################################################

# residual analysis


def magdistfunc(x,T,M,new_time):



            dM_test = np.delete(M,0)
            dT_test = np.ediff1d(T) # transform a series of timestamps to a series of interevent intervals: T_train -> dT_train
            dT_test=np.append(dT_test,new_time)
            dM_test=np.append(dM_test,x)
            n = dT_test.shape[0]
            n2 = dM_test.shape[0]
            input_RNN_times = np.array( [ dT_test[i:i+npp1.time_step] for i in range(n-npp1.time_step) ]).reshape(n-npp1.time_step,npp1.time_step,1)
            input_RNN_mags = np.array( [ dM_test[i:i+npp1.time_step] for i in range(n2-npp1.time_step) ]).reshape(n2-npp1.time_step,npp1.time_step,1)
            input_RNN_test = np.concatenate((input_RNN_times,input_RNN_mags),axis=2)
            input_CHFN_test = dT_test[-n+npp1.time_step:].reshape(n-npp1.time_step,1)
            input_CMFN_test =dM_test[-n+npp1.time_step:].reshape(n-npp1.time_step,1)


            Int_m_test = npp1.model.predict([input_RNN_test,input_CHFN_test,input_CMFN_test],batch_size=input_RNN_test.shape[0])[2]

            return Int_m_test[-1]
        



def GR(m,beta,M0,Mcut):

        if(isinstance(m, (list, tuple, np.ndarray))):
            x = np.where(m>=Mcut,beta*np.exp(-beta*(m-M0)),0)

        else:
            if(m>=Mcut):
                x = beta*np.exp(-beta*(m-M0))
            else:
                x= 0

        return x          
        
x = np.linspace(0,10,100)
index = 500

# betaMLE = 1/(mags-Mcut).mean()
# betaMLE = 1/(Amatrice.mw-M0).mean()

for index in np.linspace(30,630,3):
    index = int(index)
    T = times[:index]
    M = mags[:index]
    new_time = times[index]

    yNN = [magdistfunc(x=i,T=T,M=M,new_time=new_time) for i in x]


      


    y = GR(x,betaMLE,M0,Mcut)*np.exp((Mcut-M0)*betaMLE)
#     y = GR(x,betaMLE,Mcut,Mcut)
    plt.plot(x,yNN,label = 'NN Magnitude Distribution')
    plt.plot(x,y,label = 'Gutenberg-Richter')
    plt.hist(M,density=True,bins=10,alpha=0.5)
    plt.legend()
    plt.show()
