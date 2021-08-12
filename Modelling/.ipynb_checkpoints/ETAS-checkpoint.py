from scipy.special import gammaincc, gammainccinv, gamma, gammainc
import numpy as np
import collections

    
# expected number of aftershocks

def k(m,k0,a,M0,Mcut):
    
    if(isinstance(m, (list, tuple, np.ndarray))):
        x = np.where(m>=Mcut,k0*np.exp(a*(m-M0)),0)
        
    else:
        if(m>=Mcut):
            x = k0*np.exp(a*(m-M0))
        else:
            x= 0
    
    return x

def sGR(m,beta,M0,Mcut):

    if(isinstance(m, (list, tuple, np.ndarray))):
        x = np.where(m>=Mcut,beta*np.exp(-beta*(m-M0)),1)
        
    else:
        if(m>=Mcut):
            x = beta*np.exp(-beta*(m-M0))
        else:
            x= 1
    
    return x

# # omori decay kernel


def f(x,c,tau,w):
    return (w - 1) * c**(w - 1) * 1/((x + c)**w)

#integrated omori kernel

def H(t,w,c,tau):
    return 1 - c**(w - 1)/(t + c)**(w - 1)


# Function to calculate the intesity function from event times and magnitudes

def marked_ETAS_intensity(Tdat,Mdat,ground,k0,alpha,M0,Mcut,c,tau,w):
    
    lam = np.zeros_like(Tdat)
    lam[0] = ground
    for i in range(len(lam)):
    
        cumulative = 0
        for j in range(0,i):
            cumulative += k(Mdat[j],k0,alpha,M0,Mcut)*f(Tdat[i]-Tdat[j],c,tau,w)

    
        lam[i] = ground + cumulative
    
    return lam 

# function to calculate the likelihood of observing an earthquake sequence under the ETAS model 

def marked_likelihood(Tdat,Mdat,maxTime,ground,k0,alpha,M0,Mcut,c,tau,w,beta):
    temp = np.log(ground)
    for i in range(1,len(Tdat)):
        
        temp += np.log(ground + sum(k(Mdat[:(i )],k0,alpha,M0,Mcut) * f(Tdat[i] - Tdat[:(i)],c,tau,w))+1e-15)*(Mdat[i]>=Mcut)
        
    temp = temp - ground*(maxTime-Tdat[0])
    temp = temp - sum(k(Mdat,k0,alpha,M0,Mcut) * H(maxTime - Tdat,w,c,tau))
    temp = temp + sum(np.log(sGR(Mdat,beta,M0,Mcut)))
    
    return temp

def likelihood(Tdat,Mdat,maxTime,ground,k0,alpha,M0,Mcut,c,tau,w,beta):
    temp = np.log(ground)
    for i in range(1,len(Tdat)):
        
        temp += np.log(ground + sum(k(Mdat[:(i )],k0,alpha,M0,Mcut) * f(Tdat[i] - Tdat[:(i)],c,tau,w))+1e-15)*(Mdat[i]>=Mcut)
        
    temp = temp - ground*(maxTime-Tdat[0])
    temp = temp - sum(k(Mdat,k0,alpha,M0,Mcut) * H(maxTime - Tdat,w,c,tau))
#     temp = temp + sum(np.log(sGR(Mdat,beta,M0,Mcut)))
    
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