import pandas as pd
import numpy as np
import datetime as dt
import math
import matplotlib.pyplot as plt

from NeuralPP import NPP

from ETAS import marked_ETAS_intensity, marked_likelihood, movingaverage

import ETAS


# read in data and sort by time


data = pd.read_csv('/home/ss15859/Documents/Mini_Project/Neural_Point_Process/Simulation/my_synthetic_catalog.csv',index_col=0)
data = data.sort_values('time')


#format time in days

dates = list(data['time'])
dates_list = (np.array([dt.datetime.strptime(date[:-3], "%Y-%m-%d %H:%M:%S.%f") for date in dates]))
times = (dates_list-dates_list[0])/ dt.timedelta(days=1)
times=times.astype('float64')


test_data = pd.read_csv('/home/ss15859/Documents/Mini_Project/Neural_Point_Process/Simulation/test_catalog.csv',index_col=0)
test_data = test_data.sort_values('time')


#format time in days

test_dates = list(test_data['time'])
test_dates_list = (np.array([dt.datetime.strptime(date[:-3], "%Y-%m-%d %H:%M:%S.%f") for date in test_dates]))
test_times = (test_dates_list-test_dates_list[0])/ dt.timedelta(days=1)
test_times=test_times.astype('float64')

#test-train split

n_test = 8000
n_train = data.shape[0]

# n_train  = math.floor(0.8*data.shape[0])


M_train = np.array(data['magnitude'][:n_train])
M_test = np.array(test_data['magnitude'][:n_test])
T_train = np.array(times[:n_train])
T_test  = np.array(test_times[:n_test])

# read in and set parameters

params = pd.read_csv('/home/ss15859/Documents/Mini_Project/Neural_Point_Process/Simulation/params.csv')

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

# loop over increasing size of training set and calculate relative RMSE and mean log-likelihood

num=13

RMSE=np.zeros(num)

LL=np.zeros(num)

ln = np.linspace(np.log(10000),np.log(n_train),num)

for i in range(num):
    
    n=math.floor(np.exp(ln[i]))
    
    print('train size:  ' + str(n))

# define, train and predict with network    
    
    npp1 = NPP(time_step=time_step,size_rnn=64,size_nn=64,size_layer_chfn=2,size_layer_cmfn=2).set_train_data(T_train[:n],M_train[:n]).set_model().compile(lr=1e-3).fit_eval(epochs=30,batch_size=256).set_test_data(T_test,M_test).predict_eval()


# plot comparison of intensity functions around the largest magnitude event

    j = M_test.argmax()-time_step-1
    index=range(j,j+30)
    plt.plot(timesplot[index],npp1.lam[index],label='predicted')
    plt.plot(timesplot[index],lam[index],label='true')
    plt.legend()
    plt.title('n=  '+str(n))
    plt.show()
    
    
    # for largest training size plot relative absolute error over time with magnitudes

    fig, (ax1, ax2) = plt.subplots(2,sharex=True)
    fig.suptitle('relative absolute error in relation to magnitude of event')
    ax1.plot(movingaverage(abs(npp1.lam[:,0]-lam)/lam,15),label="RMSE")
    ax1.set_title('n = '+str(n))
    ax2.scatter(np.where(M_test>5.6)-np.repeat(time_step+1,len(np.where(M_test>5.6))),M_test[M_test>5.6],marker="x",label='Mag > 5.6',c="r")
    # ax2.legend()
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
FLL = marked_likelihood(T_test,M_test,T_test[-1],ground+0.1,k0+0.1,alpha-0.1,M0,c,tau,w-0.15)/len(T_test)

FLL

plt.plot(ln,LL)
plt.hlines(TLL,9,13,linestyles='dashed',colors='r',label='True MLL')
plt.xlabel('log n')
plt.ylabel('Mean LL')
plt.title('Mean Log-likelihood with training size')
plt.legend()
plt.show()


#######################################################################

n=38076

npp1 = NPP(time_step=time_step,size_rnn=64,size_nn=64,size_layer_chfn=2,size_layer_cmfn=2).set_train_data(T_train,M_train).set_model().compile(lr=1e-3).fit_eval(epochs=30,batch_size=256).set_test_data(T_test,M_test).predict_eval()
    

j = M_test.argmax()-time_step-1

j=1000
index=range(j,j+25)
plt.plot(timesplot[index],npp1.lam[index],label='predicted')
plt.plot(timesplot[index],lam[index],label='true')
plt.legend()
plt.title('n = '+str(n_train))
plt.show()
    
npp1.LL.mean()

TLL

M_test.argmax()-21
plt.plot(abs(npp1.lam[:,0]-(lam))/(lam))
((npp1.lam[:,0]/(lam+ground)).mean())

