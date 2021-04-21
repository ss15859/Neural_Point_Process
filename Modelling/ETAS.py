from scipy.special import gammaincc, gammainccinv, gamma, gammainc
import numpy as np

    
# expected number of aftershocks

def k(m,k0,a,M0):
    return k0*np.exp(a*(m-M0))


# # omori decay kernel


def f(x,c,tau,w):
    return (w - 1) * c**(w - 1) * 1/((x + c)**w)

#integrated omori kernel

def H(t,w,c,tau):
    return 1 - c**(w - 1)/(t + c)**(w - 1)


def marked_ETAS_intensity(Tdat,Mdat,ground,k0,alpha,M0,c,tau,w):
    
    lam = np.zeros_like(Tdat)
    lam[0] = ground
    for i in range(len(lam)):
    
        cumulative = 0
        for j in range(0,i):
            cumulative += k(Mdat[j],k0,alpha,M0)*f(Tdat[i]-Tdat[j],c,tau,w)

    
        lam[i] = ground + cumulative
    
    return lam 


def marked_likelihood(Tdat,Mdat,maxTime,ground,k0,alpha,M0,c,tau,w):
    temp = np.log(ground)
    for i in range(1,len(Tdat)):
        
        temp += np.log(ground + sum(k(Mdat[:(i )],k0,alpha,M0) * f(Tdat[i] - Tdat[:(i)],c,tau,w))+1e-9)
        
    
    temp = temp - ground*maxTime
    temp = temp - sum(k(Mdat,k0,alpha,M0) * H(maxTime - Tdat,w,c,tau))
    
    return temp



