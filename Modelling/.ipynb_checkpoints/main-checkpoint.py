import pandas as pd
import numpy as np
import datetime as dt
import math
import matplotlib.pyplot as plt

from scipy import integrate

from NeuralPP import NPP

from ETAS import marked_ETAS_intensity, marked_likelihood, movingaverage, etas_forcast, bin_times, daily_forecast

import ETAS


# read in data and sort by time


data = pd.read_csv('/home/ss15859/PhD/Neural_Point_Process/Simulation/my_synthetic_catalog.csv',index_col=0)
# data = data.sort_values('time')


# #format time in days

dates = list(data['time'])
dates_list = (np.array([dt.datetime.strptime(date[:-3], "%Y-%m-%d %H:%M:%S.%f") for date in dates]))
times = (dates_list-dates_list[0])/ dt.timedelta(days=1)
times=times.astype('float64')
# times=data['time']


test_data = pd.read_csv('/home/ss15859/PhD/Neural_Point_Process/Simulation/test_catalog.csv',index_col=0)
test_data = test_data.sort_values('time')


#format time in days

test_dates = list(test_data['time'])
test_dates_list = (np.array([dt.datetime.strptime(date[:-3], "%Y-%m-%d %H:%M:%S.%f") for date in test_dates]))
test_times = (test_dates_list-test_dates_list[0])/ dt.timedelta(days=1)
test_times=test_times.astype('float64')

#test-train split

n_test = 4000
n_train = data.shape[0]

# n_train  = math.floor(0.8*data.shape[0])


M_train = np.array(data['magnitude'][:n_train])
M_test = np.array(test_data['magnitude'][:n_test])
T_train = np.array(times[:n_train])
T_test  = np.array(test_times[:n_test])

# read in and set parameters

params = pd.read_csv('/home/ss15859/PhD/Neural_Point_Process/Simulation/params.csv')

tau=float(params["tau"])
c= float(params["c"])
w = float(params["omega"])
k0 = float(params["k0"])
alpha = float(params["a"])
M0 = float(params["M0"])
ground = float(params["mu"])
beta = float(params["beta"])


print('Calculating true lambda')

# calculate true intensity function

lam = marked_ETAS_intensity(T_test,M_test,ground,k0,alpha,M0,c,tau,w)

print('Done!')

# shift to the right by truncation parameter for future plotting

time_step=20
timesplot=T_test[time_step+1:]
lam=lam[time_step+1:]

########################################################################################

# loop over increasing size of training set and calculate relative RMSE and mean log-likelihood

num=8

RMSE=np.zeros(num)

LL=np.zeros(num)

ln = np.linspace(10000,80000,num)

for i in ln:
    
    n=int(i)
    
    print('train size:  ' + str(n))

# define, train and predict with network    
    
    npp1 = NPP(time_step=time_step,size_rnn=64,size_nn=64,size_layer_chfn=3,size_layer_cmfn=2).set_train_data(T_train[:n],M_train[:n]).set_model(0).compile(lr=1e-3).fit_eval(epochs=30,batch_size=256).set_test_data(T_test,M_test).predict_eval()


# plot comparison of intensity functions around the largest magnitude event

    j = M_test.argmax()-time_step-1
    index=range(j,j+60)
    plt.plot(timesplot[index],npp1.lam[index],label='predicted')
    plt.plot(timesplot[index],lam[index],label='true')
    plt.xlabel('time')
    plt.ylabel('Intensity')
    plt.legend()
    plt.title('n=  '+str(n))
    plt.show()
    
    
    # for largest training size plot relative absolute error over time with magnitudes

    fig, (ax1, ax2) = plt.subplots(2,sharex=True)
    fig.suptitle('relative absolute error in relation to magnitude of event')
    ax1.plot(movingaverage(abs(npp1.lam[:,0]-lam)/lam,15),label="RMSE")
    ax1.set_title('n = '+str(n))
    ax1.set(ylabel= 'relative absolute error')
    ax2.scatter(np.where(M_test>6)-np.repeat(time_step+1,len(np.where(M_test>6))),M_test[M_test>6],marker="x",label='Mag > 6.0',c="r")
    ax2.set(ylabel='Magnitude')
    ax2.legend()
    plt.xlabel("time")
    plt.show()


# calculate RMSE
    
    RMSE[i]=np.sqrt((1/lam.shape[0])*(((npp1.lam[:,0]-lam)/lam)**2).sum())
    print('RMSE:   '+str(RMSE[i]))


    LL[i]=npp1.LL.mean()
    
    
# plot RMSE with training size
    
plt.plot(ln,RMSE)
plt.xlabel('log n')
plt.ylabel('relative RMSE')
plt.title('relative RMSE with training size')
plt.show()


# calculate true mean log-likelihood and plot against training size

TLL = marked_likelihood(T_test,M_test,T_test[-1],ground,k0,alpha,M0,c,tau,w)/len(T_test)

# FLL = marked_likelihood(T_test,M_test,T_test[-1],0.204162432760849,0.200567176387296,1.49196682294346,M0,0.504734174957761,tau,2.03273231731422)/len(T_test)
# FLL

plt.plot(ln,LL)
plt.hlines(TLL,9,12,linestyles='dashed',colors='r',label='True MLL')
# plt.hlines(FLL,9,11,linestyles='dashed',colors='g',label='MLE trained MLL')
plt.xlabel('log n')
plt.ylabel('Mean LL')
plt.title('Mean Log-likelihood with training size')
plt.legend()
plt.show()


#######################################################################

# tuning regularisation parameter


loglambd = np.linspace(-8,2,10)
i=0
RMSE = np.zeros(10)
LL = np.zeros(10)

for loglam in loglambd:
    
    lamreg=np.power(10,loglam)
    
    npp1 = NPP(time_step=time_step,size_rnn=64,size_nn=64,size_layer_chfn=2,size_layer_cmfn=2).set_train_data(T_train,M_train).set_model(lamreg).compile(lr=1e-3).fit_eval(epochs=30,batch_size=256).set_test_data(T_test,M_test).predict_eval()

    RMSE[i]=np.sqrt((1/lam.shape[0])*(((npp1.lam[:,0]-lam)/lam)**2).sum())
    print('RMSE:   '+str(RMSE[i]))


    LL[i]=npp1.LL.mean()
     
        
    j = M_test.argmax()-time_step-1
    index=range(j,j+100)
    plt.plot(timesplot[index],npp1.lam[index],label='predicted')
    plt.plot(timesplot[index],lam[index],label='true')
    plt.legend()
    plt.title('loglam=  '+str(loglam))
    plt.show()
    
    
    # for largest training size plot relative absolute error over time with magnitudes

    fig, (ax1, ax2) = plt.subplots(2,sharex=True)
    fig.suptitle('relative absolute error in relation to magnitude of event')
    ax1.plot(movingaverage(abs(npp1.lam[:,0]-lam)/lam,15),label="RMSE")
    ax1.set_title('loglam = '+str(loglam))
    ax2.scatter(np.where(M_test>5.6)-np.repeat(time_step+1,len(np.where(M_test>5.6))),M_test[M_test>5.6],marker="x",label='Mag > 5.6',c="r")
    # ax2.legend()
    plt.xlabel("time")
    plt.show()    
        
    i +=1 
    
    
plt.plot(loglambd,RMSE)
plt.xlabel('log lam')
plt.ylabel('relative RMSE')
plt.title('relative RMSE with rregularisation parameter')
plt.show()    
    
TLL = marked_likelihood(T_test,M_test,T_test[-1],ground,k0,alpha,M0,c,tau,w)/len(T_test)    
    
plt.plot(loglambd,LL)
plt.hlines(TLL,-7.5,2.5,linestyles='dashed',colors='r',label='True MLL')
# plt.hlines(FLL,9,11,linestyles='dashed',colors='g',label='MLE trained MLL')
plt.xlabel('log lam')
plt.ylabel('Mean LL')
plt.title('Mean Log-likelihood with regularisation parameter')
plt.legend()
plt.show()


#####################################################################################

# Tunning over rnn size

rnnsize = np.linspace(2,10,9)
i=0
RMSE = np.zeros(9)
LL = np.zeros(9)

for size in rnnsize:
    
    rnnsz=int(np.power(2,size))
    
    npp1 = NPP(time_step=time_step,size_rnn=rnnsz,size_nn=64,size_layer_chfn=2,size_layer_cmfn=2).set_train_data(T_train,M_train).set_model(0).compile(lr=1e-3).fit_eval(epochs=40,batch_size=256).set_test_data(T_test,M_test).predict_eval()

    RMSE[i]=np.sqrt((1/lam.shape[0])*(((npp1.lam[:,0]-lam)/lam)**2).sum())
    print('RMSE:   '+str(RMSE[i]))


    LL[i]=npp1.LL.mean()
     
        
    j = M_test.argmax()-time_step-1
    index=range(j,j+100)
    plt.plot(timesplot[index],npp1.lam[index],label='predicted')
    plt.plot(timesplot[index],lam[index],label='true')
    plt.legend()
    plt.title('rnn size =  '+str(rnnsz))
    plt.show()
    
    
    # for largest training size plot relative absolute error over time with magnitudes

    fig, (ax1, ax2) = plt.subplots(2,sharex=True)
    fig.suptitle('relative absolute error in relation to magnitude of event')
    ax1.plot(movingaverage(abs(npp1.lam[:,0]-lam)/lam,15),label="RMSE")
    ax1.set_title('rnn size =  '+str(rnnsz))
    ax2.scatter(np.where(M_test>5.6)-np.repeat(time_step+1,len(np.where(M_test>5.6))),M_test[M_test>5.6],marker="x",label='Mag > 5.6',c="r")
    # ax2.legend()
    plt.xlabel("time")
    plt.show()    
        
    i +=1 
    
    
plt.plot(rnnsize,RMSE)
plt.xlabel('log_2 rnn size')
plt.ylabel('relative RMSE')
plt.title('relative RMSE with rnn size')
plt.show()    

TLL = marked_likelihood(T_test,M_test,T_test[-1],ground,k0,alpha,M0,c,tau,w)/len(T_test)

plt.plot(rnnsize,LL)
plt.hlines(TLL,2,10,linestyles='dashed',colors='r',label='True MLL')
# plt.hlines(FLL,9,11,linestyles='dashed',colors='g',label='MLE trained MLL')
plt.xlabel('log_2 rnn size')
plt.ylabel('Mean LL')
plt.title('Mean Log-likelihood with rnn size')
plt.legend()
plt.show()


############################################################################


# Tunning over chfn size

chfnsize = np.linspace(2,10,9)
i=0
RMSE = np.zeros(9)
LL = np.zeros(9)

for size in chfnsize:
    
    size=int(size)
    
    npp1 = NPP(time_step=time_step,size_rnn=64,size_nn=64,size_layer_chfn=size,size_layer_cmfn=2).set_train_data(T_train,M_train).set_model(0).compile(lr=1e-3).fit_eval(epochs=40,batch_size=256).set_test_data(T_test,M_test).predict_eval()

    RMSE[i]=np.sqrt((1/lam.shape[0])*(((npp1.lam[:,0]-lam)/lam)**2).sum())
    print('RMSE:   '+str(RMSE[i]))


    LL[i]=npp1.LL.mean()
     
        
    j = M_test.argmax()-time_step-1
    index=range(j,j+100)
    plt.plot(timesplot[index],npp1.lam[index],label='predicted')
    plt.plot(timesplot[index],lam[index],label='true')
    plt.legend()
    plt.title('chfn depth =  '+str(size))
    plt.show()
    
    
    # for largest training size plot relative absolute error over time with magnitudes

    fig, (ax1, ax2) = plt.subplots(2,sharex=True)
    fig.suptitle('relative absolute error in relation to magnitude of event')
    ax1.plot(movingaverage(abs(npp1.lam[:,0]-lam)/lam,15),label="RMSE")
    ax1.set_title('chfn depth =  '+str(size))
    ax2.scatter(np.where(M_test>5.6)-np.repeat(time_step+1,len(np.where(M_test>5.6))),M_test[M_test>5.6],marker="x",label='Mag > 5.6',c="r")
    # ax2.legend()
    plt.xlabel("time")
    plt.show()    
        
    i +=1 
    
    
plt.plot(rnnsize,RMSE)
plt.xlabel('chfn depth')
plt.ylabel('relative RMSE')
plt.title('relative RMSE with chfn depth')
plt.show()    

TLL = marked_likelihood(T_test,M_test,T_test[-1],ground,k0,alpha,M0,c,tau,w)/len(T_test)

FLL = marked_likelihood(T_test,M_test,T_test[-1],0.53177,0.199108683739048,1.47413683940633,M0,0.482027753360799,tau,1.52989010649367)/len(T_test)

plt.plot(rnnsize,LL)
plt.hlines(TLL,2,10,linestyles='dashed',colors='r',label='True MLL')
plt.hlines(FLL,2,10,linestyles='dashed',colors='g',label='MLE trained MLL')
plt.xlabel('chfn depth')
plt.ylabel('Mean LL')
plt.title('Mean Log-likelihood with chfn depth')
plt.legend()
plt.show()


#############################################################################




# loop over increasing size of training set and calculate relative RMSE and mean log-likelihood

num=7

n = np.linspace(11000,77000,num)

RMSE=np.zeros(num)

RMSE2 = np.zeros(num)

LL=np.zeros(num)
LL2=np.zeros(num)
LLpois = np.zeros(num)

count=0

for i in n:
    
    
    i=int(i)
    filename= 'results'+str(i)+'.csv'
    
    MLEparams = pd.read_csv(filename)
    
    groundMLE = MLEparams['x'][0]
    k0MLE = MLEparams['x'][1]
    alphaMLE = MLEparams['x'][2]
    cMLE = MLEparams['x'][3]
    wMLE = MLEparams['x'][4]
    
    print('calculating lam')
    
#     lamMLE = marked_ETAS_intensity(T_test,M_test,groundMLE,k0MLE,alphaMLE,M0,cMLE,tau,wMLE)

    print('Done!')


#     lamMLE=lamMLE[time_step+1:]
    
    print('train size:  ' + str(i))

# define, train and predict with network    
    
    npp1 = NPP(time_step=time_step,size_rnn=64,size_nn=64,size_layer_chfn=3,size_layer_cmfn=2).set_train_data(T_train[:i],M_train[:i]).set_model(0).compile(lr=1e-3).fit_eval(epochs=30,batch_size=256).set_test_data(T_test,M_test).predict_eval()


    poissMLE = 1/np.ediff1d(T_train[:i]).mean()
    
    LLpois[count] = (len(T_test)*np.log(poissMLE) - T_test[-1]*poissMLE)/len(T_test)
    
# plot comparison of intensity functions around the largest magnitude event

#     j = M_test.argmax()-time_step-1
#     index=range(j,j+90)
#     plt.plot(timesplot[index],npp1.lam[index],label='NN')
#     plt.plot(timesplot[index],lam[index],label='true')
#     plt.plot(timesplot[index],lamMLE[index],label='MLE')
#     plt.legend()
#     plt.title('n=  '+str(i))
#     plt.show()
    
    
#     # for largest training size plot relative absolute error over time with magnitudes

#     fig, (ax1, ax2) = plt.subplots(2,sharex=True)
#     fig.suptitle('relative absolute error in relation to magnitude of event')
#     ax1.plot(movingaverage(abs(npp1.lam[:,0]-lam)/lam,15),label="RMSE")
#     ax1.set_title('n = '+str(i))
#     ax2.scatter(np.where(M_test>5.6)-np.repeat(time_step+1,len(np.where(M_test>5.6))),M_test[M_test>5.6],marker="x",label='Mag > 5.6',c="r")
#     # ax2.legend()
#     plt.xlabel("time")
#     plt.show()


# calculate RMSE
    
#     RMSE[count]=np.sqrt((1/lam.shape[0])*(((npp1.lam[:,0]-lam)/lam)**2).sum())
#     print('RMSE:   '+str(RMSE[count]))


#     RMSE2[count]=np.sqrt((1/lam.shape[0])*(((lamMLE-lam)/lam)**2).sum())
    
    
    
    LL[count]=npp1.LL.mean()
    
    print('calculating likelihood of MLE')
    
    LL2[count] = marked_likelihood(T_test,M_test,T_test[-1],groundMLE,k0MLE,alphaMLE,M0,cMLE,tau,wMLE)/len(T_test)
    
    print('done')
    
    count+=1
    
# plot RMSE with training size
    
# plt.plot(n,RMSE,label='NN')
# plt.plot(n,RMSE2,label='MLE')
# plt.xlabel('training size')
# plt.ylabel('relative RMSE')
# plt.legend()
# plt.title('relative RMSE with training size')
# plt.show()


# calculate true mean log-likelihood and plot against training size

TLL = marked_likelihood(T_test,M_test,T_test[-1],ground,k0,alpha,M0,c,tau,w)/len(T_test)

# FLL = marked_likelihood(T_test,M_test,T_test[-1],0.204162432760849,0.200567176387296,1.49196682294346,M0,0.504734174957761,tau,2.03273231731422)/len(T_test)
# FLL

plt.plot(n,LL,label='NN')
plt.plot(n,LL2,label='ETAS MLE')
plt.plot(n,LLpois,label='Poisson MLE')
plt.hlines(TLL,10000,80000,linestyles='dashed',colors='r',label='True MLL')
plt.xlabel('training size')
plt.ylabel('Mean LL')
plt.title('Mean Log-likelihood with training size')
plt.legend()
plt.show()

#############################################################################
npp1 = NPP(time_step=time_step,size_rnn=64,size_nn=64,size_layer_chfn=3,size_layer_cmfn=2).set_train_data(T_train,M_train).set_model(0).compile(lr=1e-3).fit_eval(epochs=30,batch_size=256).set_test_data(T_test,M_test).predict_eval()
    

j = M_test.argmax()-time_step-1

magsplot = M_test[time_step+1:]

j=190
index=range(j,j+50)    
plt.plot(timesplot[index],npp1.lam[index],label='NN')
plt.plot(timesplot[index],lam[index],label='Data Generating')
plt.xlabel('time')
plt.ylabel('intensity')
# plt.plot(timesplot[index],lamMLE[index],label='MLE')
plt.legend()
plt.title('n=  '+str(i))
plt.show()


j=1915
index=range(j,j+50)   
fig, axs = plt.subplots(2,sharex=True)
axs[0].plot(timesplot[index],npp1.lam[index],label='NN')
axs[0].plot(timesplot[index],lam[index],label='Data Generating')
axs[0].set_ylabel('intensity')
# plt.plot(timesplot[index],lamMLE[index],label='MLE')
axs[0].legend()
axs[1].scatter(timesplot[index],magsplot[index],s = 10)
axs[1].set_ylabel('Magnitude')
axs[1].set_xlabel('Time')
fig.savefig('2panesmallmag.png')




    
npp1.LL.mean()

TLL

M_test.argmax()-21
plt.plot(abs(npp1.lam[:,0]-(lam))/(lam))
((npp1.lam[:,0]/(lam+ground)).mean())


#############################################################

npp1 = NPP(time_step=time_step,size_rnn=64,size_nn=64,size_layer_chfn=3,size_layer_cmfn=2).set_train_data(T_train,M_train).set_model(0).compile(lr=1e-3).fit_eval(epochs=30,batch_size=256).set_test_data(T_test,M_test).predict_eval()

repeats = 100
npoints = 300

forcastN = np.zeros((npoints,repeats))
for j in range(repeats):
    print(j,'\r')
    for i in range(npoints):

        n = i+21
    #     print(n)
        hist1 = [T_test[:n],M_test[:n]]
        forcastN[i,j] = npp1.forecast(hist1,len_window = 1,M0 = 3)

        
true = bin_times(T_test[:npoints+21],1,time_step)

ave = np.mean(forcastN,axis=1)

def RPD(x,y):
    return 2*(x-y)/(abs(x)+abs(y))


forcastETAS = np.zeros((npoints,repeats))
for j in range(repeats):
    for i in range(npoints):

        n = i+21
    #     print(n)
        hist1 = [T_test[:n],M_test[:n]]
        forcastETAS[i,j] = etas_forcast(hist1,1,params)

aveETAS = np.mean(forcastETAS,axis=1)


##################################################################################

# simulate then train on higher M0, then add residual analysis

Mcut = 3.5
M0 = 0.3

T_train = T_train[M_train>=Mcut]
M_train = M_train[M_train>=Mcut]

T_test = T_test[M_test>=Mcut]
M_test = M_test[M_test>=Mcut]


# wrt = pd.DataFrame()
# wrt['time'] = T_train
# wrt['mw'] = M_train
# wrt.to_csv('sim_Mcut_4.0.csv')


# params = pd.read_csv('9thJulyResultsM02.csv')

params = pd.read_csv('~/PhD/Amatrice_tests/sim_Mcut_3.5_params_wrongM0.csv')


groundMLE = params.x[0]
k0MLE= params.x[1]
alphaMLE = params.x[2]
cMLE = params.x[3]
wMLE = params.x[4]
betaMLE = params.x[5]
#     M0 = mags.min()
tau =0



npp1 = NPP(time_step=20,size_rnn=64,size_nn=64,size_layer_chfn=3,size_layer_cmfn=2).set_train_data(T_train,M_train).set_model(0).compile(lr=1e-3).fit_eval(epochs=30,batch_size=256).set_test_data(T_test,M_test).predict_eval()

LLNN = npp1.LL.mean()


poissMLE = 1/np.ediff1d(T_train[M_train>=Mcut]).mean()
    
LLpoiss = (len(T_test[M_test>=Mcut])*np.log(poissMLE) + sum(np.log(ETAS.sGR(M_test,betaMLE,M0,Mcut)))- (T_test[-1]-T_test[0])*poissMLE)/len(T_test[M_test>=Mcut])

LLMLE = marked_likelihood(T_test,M_test,T_test[-1],groundMLE,k0MLE,alphaMLE,M0,Mcut,cMLE,tau,wMLE,betaMLE)/len(T_test[M_test>=Mcut])

y = range(1,len(npp1.Int_lam)+1)
t = np.cumsum(npp1.Int_lam)

lam = marked_ETAS_intensity(T_test,M_test,groundMLE,k0MLE,alphaMLE,M0,Mcut,cMLE,tau,wMLE)

lam_int = integrate.cumtrapz(lam, T_test, initial=0)
yETAS = range(1,len(lam_int)+1)

plt.plot(t,y,color='black',label = 'NN')
plt.plot(lam_int,yETAS,color='green',label = 'ETAS')
plt.plot(np.linspace(0,t[-1]),np.linspace(0,t[-1]),color = 'r',label = 'y = x')
plt.xlabel('Transformed time')
plt.ylabel('Cumulative number')
plt.legend()
# plt.title('Trained on data up to ' + str(timeupto) + ' hours from start')
plt.show()

npp1.eval_train_data()
    
y = range(1,len(npp1.Int_lam_train)+1)
t = np.cumsum(npp1.Int_lam_train)

lam = marked_ETAS_intensity(T_train,M_train,groundMLE,k0MLE,alphaMLE,M0,Mcut,cMLE,tau,wMLE)

lam_int = integrate.cumtrapz(lam, T_train, initial=0)
yETAS = range(1,len(lam_int)+1)

plt.plot(t,y,color='black',label = 'NN')
plt.plot(lam_int,yETAS,color='green',label = 'ETAS')
plt.plot(np.linspace(0,t[-1]),np.linspace(0,t[-1]),color = 'r',label = 'y = x')
plt.xlabel('Transformed time')
plt.ylabel('Cumulative number')
plt.legend()
# plt.title('Trained on data up to ' + str(timeupto) + ' hours from start')
plt.show()

NNgain = LLNN-LLpoiss

MLEgain = LLMLE - LLpoiss





repeats = 10
npoints = 300

forcastN = np.zeros((npoints,repeats))
for j in range(repeats):
    print(j,'\r')
    for i in range(npoints):

        n = i+21
    #     print(n)
        hist1 = [T_test[:n],M_test[:n]]
        forcastN[i,j] = npp1.forecast(hist1,len_window = 1,M0 = M0)

        
ave = np.mean(forcastN,axis=1)
        
true = bin_times(T_test[:npoints+21],1,time_step)



def RPD(x,y):
    return 2*(x-y)/(abs(x)+abs(y))


forcastETAS = np.zeros((npoints,repeats))
for j in range(repeats):
    for i in range(npoints):

        n = i+21
    #     print(n)
        hist1 = [T_test[:n],M_test[:n]]
        forcastETAS[i,j] = etas_forcast(hist1,1,params)

aveETAS = np.mean(forcastETAS,axis=1)

l = np.zeros(npoints)
for i in range(npoints):

        n = i+21
    #     print(n)
        hist1 = [T_test[:n+1],M_test[:n]]
        l[i] = intensity(T_test[n],hist1,params)
        

#######################################################################################
# Now daily forecasting

ndays = 200
repeats = 3
# forcastETAS = np.zeros((ndays,repeats))
# for j in range(repeats):
#     for i in range(ndays):


#         hist1 = [T_test[T_test<=i],M_test[T_test<=i]]
#         forcastETAS[i,j] = etas_forcast(i,hist1,1,params)


forcastETAS = daily_forecast(T_test,M_test,ndays=200,repeats=3,params=params,hours=1)

aveETAS = np.mean(forcastETAS,axis=1)


true = bin_times(T_test)


forcastN = npp1.daily_forecast(T_test,M_test,ndays=200,repeats=3,M0=M0,time_step=time_step,hours=1)

        
ave = np.mean(forcastN,axis=1)


