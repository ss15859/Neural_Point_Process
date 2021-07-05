from scipy.special import gammaincc, gammainccinv, gamma, gammainc
import numpy as np

    
# expected number of aftershocks

def k(m,k0,a,M0):
    
    if(isinstance(m, (list, tuple, np.ndarray))):
        x = np.where(m>=M0,k0*np.exp(a*(m-M0)),0)
        
    else:
        if(m>=M0):
            x = k0*np.exp(a*(m-M0))
        else:
            x= 0
    
    return x

def sGR(m,beta,M0):

    if(isinstance(m, (list, tuple, np.ndarray))):
        x = np.where(m>=M0,beta*np.exp(beta*(m-M0)),1)
        
    else:
        if(m>=M0):
            x = beta*np.exp(beta*(m-M0))
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

def marked_ETAS_intensity(Tdat,Mdat,ground,k0,alpha,M0,c,tau,w):
    
    lam = np.zeros_like(Tdat)
    lam[0] = ground
    for i in range(len(lam)):
    
        cumulative = 0
        for j in range(0,i):
            cumulative += k(Mdat[j],k0,alpha,M0)*f(Tdat[i]-Tdat[j],c,tau,w)

    
        lam[i] = ground + cumulative
    
    return lam 

# function to calculate the likelihood of observing an earthquake sequence under the ETAS model 

def marked_likelihood(Tdat,Mdat,maxTime,ground,k0,alpha,M0,c,tau,w,beta):
    temp = np.log(ground)
    for i in range(1,len(Tdat)):
        
        temp += np.log(ground + sum(k(Mdat[:(i )],k0,alpha,M0) * f(Tdat[i] - Tdat[:(i)],c,tau,w))+1e-15)*(Mdat[i]>=M0)
        
    temp = temp - ground*(maxTime-Tdat[0])
    temp = temp - sum(k(Mdat,k0,alpha,M0) * H(maxTime - Tdat,w,c,tau))
    temp = temp + sum(np.log(sGR(Mdat,beta,M0)))
    
    return temp


# function used in plotting relative absolute error over time. The function averages a function f(x) over some window [x-r,x+r]

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

