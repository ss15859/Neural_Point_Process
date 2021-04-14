from scipy.special import gammaincc, gammainccinv, gamma, gammainc
import numpy as np



def upper_gamma_ext(a, x):
    if a > 0:
        return gammaincc(a, x) * gamma(a)
    elif a == 0:
        return exp1(x)
    else:
        return (upper_gamma_ext(a + 1, x) - np.power(x, a)*np.exp(-x)) / a

    
# expected number of aftershocks

def k(m,k0,alpha,M0,d,gama,rho,c,tau,w):
    
    number_factor = k0 * np.exp(alpha * (m - M0))
    area_factor = np.pi * np.power(
        d * np.exp(gama * (m - M0)),
        -1 * rho
    ) / rho

    time_factor = np.exp(c/tau) * np.power(tau, -w) 
    time_fraction = upper_gamma_ext(-w, c/tau)
    
    
    return number_factor*area_factor*time_factor*time_fraction

#integrated omori kernel

def H(t,w,c,tau):
    return 1 - gammaincc(-w,(t+c)/tau)/gammaincc(-w,c/tau)


# omori decay kernel

def f(x,c,tau,w):
    return np.exp(-(x+c)/tau)/(gammaincc(-w,c/tau)*tau*np.power(((x+c)/tau),1+w)*gamma(-w))


def marked_ETAS_intensity(Tdat,Mdat,ground,area,k0,alpha,M0,d,gama,rho,c,tau,w):
    
    lam = np.zeros_like(Tdat)
    lam[0] = ground*area
    for i in range(len(lam)):
    
        cumulative = 0
        for j in range(1,i):
            cumulative += k(Mdat[j],k0,alpha,M0,d,gama,rho,c,tau,w)*f(Tdat[i]-Tdat[j],c,tau,w)

    
        lam[i] = ground*area + cumulative
    
    return lam 


def marked_likelihood(Tdat,Mdat,maxTime,ground,area,k0,alpha,M0,d,gama,rho,c,tau,w):
    temp = np.log(ground*area)
    for i in range(1,len(Tdat)):
        
        temp += np.log(ground*area + sum(k(Mdat[:(i )],k0,alpha,M0,d,gama,rho,c,tau,w) * f(Tdat[i] - Tdat[:(i)],c,tau,w))+1e-7)
        
    
    print(H(maxTime - Tdat,w,c,tau))
    temp = temp - ground*maxTime*area
    temp = temp - sum(k(Mdat,k0,alpha,M0,d,gama,rho,c,tau,w) * H(maxTime - Tdat,w,c,tau))
    
    return temp