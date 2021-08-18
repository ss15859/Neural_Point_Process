from scipy.special import gammaincc, gammainccinv, gamma, gammainc
import numpy as np
import collections
import math
from scipy.optimize import minimize
import pandas as pd

    
# expected number of aftershocks

def k(m,params,Mcut):
    
    if(isinstance(m, (list, tuple, np.ndarray,pd.Series))):
        x = np.where(m>=Mcut,params['k0']*np.exp(params['a']*(m-params['M0'])),0)
        
    else:
        if(m>=Mcut):
            x = params['k0']*np.exp(params['a']*(m-params['M0']))
        else:
            x= 0
    
    return x

def sGR(m,params,Mcut):

    if(isinstance(m, (list, tuple, np.ndarray,pd.Series))):
        x = np.where(m>=Mcut,params['beta']*np.exp(-params['beta']*(m-params['M0'])),1)
        
    else:
        if(m>=Mcut):
            x = params['beta']*np.exp(-params['beta']*(m-params['M0']))
        else:
            x= 1
    
    return x

# # omori decay kernel


def f(x,params):
    return (params['omega'] - 1) * params['c']**(params['omega'] - 1) * 1/((x + params['c'])**params['omega'])

#integrated omori kernel

def H(t,params):
    return 1 - params['c']**(params['omega'] - 1)/(t + params['c'])**(params['omega'] - 1)


# Function to calculate the intesity function from event times and magnitudes

def marked_ETAS_intensity(Tdat,Mdat,mu,k0,alpha,M0,Mcut,c,tau,omega):
    
    lam = np.zeros_like(Tdat)
    lam[0] = mu
    for i in range(len(lam)):
    
        cumulative = 0
        for j in range(0,i):
            cumulative += k(Mdat[j],k0,alpha,M0,Mcut)*f(Tdat[i]-Tdat[j],c,tau,omega)

    
        lam[i] = mu + cumulative
    
    return lam 

# function to calculate the likelihood of observing an earthquake sequence under the ETAS model 

def marked_likelihood(Tdat,Mdat,maxtime,Mcut,params):

    temp = np.log(params['mu'])
    for i in range(1,len(Tdat)):
        
        temp += np.log(params['mu'] + sum(k(Mdat[:(i )],params,Mcut) * f(Tdat[i] - Tdat[:(i)],params))+1e-15)
        
    temp = temp - params['mu']*(maxtime-Tdat[0])
    temp = temp - sum(k(Mdat,params,Mcut) * H(maxtime - Tdat,params))
    temp = temp + sum(np.log(sGR(Mdat,params,Mcut)))
    
    return temp

# def likelihood(Tdat,Mdat,maxtime,mu,k0,alpha,M0,Mcut,c,tau,omega,beta):
#     temp = np.log(mu)
#     for i in range(1,len(Tdat)):
        
#         temp += np.log(mu + sum(k(Mdat[:(i )],k0,alpha,M0,Mcut) * f(Tdat[i] - Tdat[:(i)],c,tau,omega))+1e-15)*(Mdat[i]>=Mcut)
        
#     temp = temp - mu*(maxtime-Tdat[0])
#     temp = temp - sum(k(Mdat,k0,alpha,M0,Mcut) * H(maxtime - Tdat,omega,c,tau))
# #     temp = temp + sum(np.log(sGR(Mdat,beta,M0,Mcut)))
    
#     return temp

def likelihood(Tdat,Mdat,maxtime,Mcut,params):
    
    temp = np.log(params['mu'])
    for i in range(1,len(Tdat)):
        
        temp += np.log(params['mu'] + sum(k(Mdat[:(i )],params,Mcut) * f(Tdat[i] - Tdat[:(i)],params))+1e-15)
        
    temp = temp - params['mu']*(maxtime-Tdat[0])
    temp = temp - sum(k(Mdat,params,Mcut) * H(maxtime - Tdat,params))
    
    return temp



# function used in plotting relative absolute error over time. The function averages a function f(x) over some window [x-r,x+r]

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def simulate_ETAS_mag(params):

    u = np.random.uniform()
    m = (-1 * np.log(1 - u ) / params['beta'][0]) + params['M0'][0]
    return m
    
    
def intensity(t,hist,params):
    
    if(t == hist[0][-1]):
        n = len(hist[0])-1

    else:
        n = len(hist[0])
    
    cumulative = 0
    
    for j in range(0,n):
        cumulative += k(hist[1][j],params['k0'][0],params['a'][0],params['M0'][0],params['M0'][0])*f(t-hist[0][j],params['c'][0],params['tau'][0],params['omega'][0])


    lam = params['mu'][0] + cumulative
    return lam
    
    
    
def etas_forcast(tstart,hist,len_window,params):
    
#     tstart = hist[0][-1]
    t=tstart
    count = 0
    
    
    while True:
        
        B = intensity(t,hist,params)
        tau  = np.random.exponential(1/B)

        if t+tau> tstart+len_window:
            break
        
        if np.random.uniform() < intensity(t+tau,hist,params)/B:
            m = simulate_ETAS_mag(params)
            hist = [np.append(hist[0],t+tau),np.append(hist[1],m)]
            count+=1

        t += tau
        
    return count

def daily_forecast(Tdat,Mdat,ndays,repeats,params,hours):
    
    Tdat = Tdat-Tdat[0]
    
    forcastETAS = np.zeros((ndays,repeats))
    for j in range(repeats):
        for i in range(ndays):


            hist1 = [Tdat[Tdat<=i*hours],Mdat[Tdat<=i*hours]]
            forcastETAS[i,j] = etas_forcast(i*hours,hist1,1*hours,params)


    return forcastETAS


# def bin_times(Tdat,len_win,time_step):
    
#     N = np.zeros(len(Tdat)-(time_step+1))
    
#     for i in range(len(N)):

#         N[i] = (Tdat[i+time_step+1:]<(Tdat[i+time_step+1]+len_win)).sum()-1
        
#     return N


def bin_times(Tdat,hours):
    
    rounding = np.floor(Tdat/hours)
    
    output = np.zeros(int(rounding[-1]-rounding[0])+1)
    
    for i in range(len(output)):
         
        output[i] = sum(rounding==i+rounding[0])
         
        
    return output


def datetime64_to_days(dates):
    
    dates = np.array(dates)
    times = (dates-dates[0])/  np.timedelta64(1,'D')
    times=times.astype('float64')
    return times




def maxlikelihoodETAS(Tdat,Mdat,M0,maxtime=np.nan,initval=np.nan):
    
    if(math.isnan(maxtime)):
        maxtime = Tdat.max()

        
        
    def fn(param_array):
        
        params = {
            'mu' : param_array[0],
            'k0' : param_array[1],
            'a' : param_array[2],
            'c' : param_array[3],
            'omega' : param_array[4],
            'M0' : M0
        }
        
        if (params['mu'] <= 0 or params['omega'] <= 1 or params['a'] < 0 or params['k0'] < 0): 
            
            return(math.inf)
        
        val = -likelihood(Tdat,Mdat,maxtime=maxtime,Mcut=M0,params=params)
        
        return val
    
    
    if math.isnan(initval):
        initval = np.array([len(Tdat)/maxtime,0.5,0.5,1,2])
        
    else:
        initval = initval[:5]
    
    temp = minimize(fn,initval,method='Nelder-Mead',options={'xatol': 1e-4, 'disp': True})
    
    dic = { 'mu' : temp.x[0],
            'k0' : temp.x[1],
            'a' : temp.x[2],
            'c' : temp.x[3],
            'omega' : temp.x[4],
            'M0' : M0}
        
    
    return dic