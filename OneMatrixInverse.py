''' Sietske Lensen 2021
    Code accompanying Master thesis:
    Analytical approach of Sensitivity Analysis for Life Cycle Assessment '''

import numpy as np
import ComplexMultivariateNormal as CMN
import itertools
np.set_printoptions(precision=3)
np.random.seed()

''' This code calculates the covariance of the inverse of matrix A (pxp) whose coefficients are correlated and distributed lognormally. The covariance is calculated first through simulation and second though an analytical formula. The output matrix is called H (pxp)'''
# Initializing program input parameters
# p=3,seed=7,geo=4 convergeerde mooi
p = 3                               # Matrix dimensions
M = p*p                             # Total input coefficients
IND = np.arange(M)                  # All indices
INA = np.reshape(IND,[p,p])         # Indices of A
fINA = INA.reshape([np.size(INA)])  # Flat indices of A
P = 0.2                             # Ratio of zero means
N = 100000                          # Number of simulations
GEO = 3                             # Geometric series maximum

# Picking random parameters for the input coefficients
mu = np.random.normal(size=M)
mu *= np.random.choice([0,1],size=M,p=[P,1-P])
mu[np.diag(INA)] = 1
mumu = np.multiply.outer(mu,mu)

# Checks for the convergence of the randomly generated matrix. These numbers will show whether the geometric series should converge.
# print('Convergence \n', np.mean(np.matrix(np.identity(p)-mu[INA])**p))
print('Convergence results')
print((np.sum([(np.identity(p) - np.matrix(mu[INA]))**i for i in range(GEO)],axis=0)/np.matrix(mu[INA])**(-1)))

preSigma = (np.random.rand(M,M)-1/2)/5
scaler = np.ones_like(mumu)
scaler[np.diag(INA)], scaler[:,np.diag(INA)]  = 0,0
sigma = scaler * mumu * (np.exp(np.matmul(preSigma,preSigma.T)) - 1)

# Parameters of W - used in the analytical calulation
muW = -np.copy(mu)
muW[np.diag(INA)] = 0
sigmaW = sigma


# Parameter functions
''' Function that transforms parameters of normal multivariate Y to parameters of lognormal multivariate X, if X = exp(Y) '''
def NormToLN(meanY,sigmaY):
    meanX = np.exp(meanY + 1/2 * np.diag(np.matrix(sigmaY)))
    sigmaX = np.multiply(np.multiply.outer(meanX,meanX),np.exp(sigmaY)-1)
    return meanX, sigmaX

''' Function that transforms parameters of lognormal multivariate X to parameters of normal multivariate Y, if Y = ln(X) '''
def LNToNorm(meanX,sigmaX):
    temp = np.multiply.outer(meanX,meanX)
    sigmaY = np.log(np.divide(sigmaX,temp,out=np.zeros_like(sigmaX),where=temp!=0) + 1 + 0j)
    meanY = np.log(meanX + 0j) - 1/2 * np.real(np.diag(np.matrix(sigmaY)))
    return meanY, sigmaY



''' Part of the code where covariance is calculated through sampling '''
# The lognormal sample is constructed by taking the exponent of a (complex) normal sample, for which the parameter relationships are as in the above functions
# Using the package ComplexMultivariateNormal source: (\url)
SamplingMu, SamplingSigma = LNToNorm(mu,sigma) # Find normal parameters for initialized lognormal parameters
SamplingSigmaConj = np.triu(SamplingSigma)+np.conj(np.tril(SamplingSigma,-1))
samples = np.real(np.exp(CMN.ScipyComplexMultivariateNormal(SamplingMu,SamplingSigma,SamplingSigmaConj).rvs(N)))


#Model calculation from samples
sampA = samples[:,INA]
sampI = np.linalg.inv(sampA)
sampH = np.reshape(sampI,[N,p*p])
sampMuH = np.mean(sampH,axis=0)
sampSigHF = np.cov(sampH.T)


''' Function that calculates the mean and covariance of the product (Z) of a set of lognormally distributed and correlated coefficients (X) '''
def ProductParameters(MuX,SigmaX):
    # Calculation of parameters as in equation 16 of chapter 6
    temp = np.multiply.outer(MuX,MuX)
    Theta = np.divide(SigmaX,temp,out=np.zeros_like(SigmaX),where=temp!=0)+1
    np.fill_diagonal(Theta,1)    # Setting diagonal elements of Theta to 1, ensures that the subsequent product is corrected for not including the diagonal.
    GammaZ = np.product(Theta)
    MuZ = np.product(MuX)*np.sqrt(GammaZ)
    SigmaZ = np.product(np.diag(SigmaX)+MuX**2) * GammaZ**2 - MuZ**2
    return MuZ, SigmaZ

''' Function that selects the parameters of a set of coefficients'''
def MuSigmaFromLine(subset):
    subset = np.array(subset)
    if (subset.size == 0):
        returnable = np.array([]),np.array([])
    else: returnable = muW[subset], sigmaW[subset[:,None],subset]
    return returnable

''' Function that calculates the covariance of two parameter product sets'''
def CovLine(lineA,lineB,lineAB):
    MuAB, SigmaAB = MuSigmaFromLine(lineAB)
    MuA,  SigmaA  = MuSigmaFromLine(lineA)
    MuB,  SigmaB  = MuSigmaFromLine(lineB)
    return ProductParameters(MuAB,SigmaAB)[0]  - ProductParameters(MuA,SigmaA)[0]*ProductParameters(MuB,SigmaB)[0]

''' Function that determines which parameters take part in an element of the sum'''
def Chi(i,j,m,k,l,n):
    iterationIndices = itertools.product(range(p),repeat=m+n)
    MC = np.cumsum([0,m,n])
    temp = 0
    for elements in iterationIndices:
        lineA,lineB = [], []
        
        fullM = np.append(np.append(i,elements[MC[0]:MC[1]]),j).astype(int)
        fullN = np.append(np.append(k,elements[MC[1]:MC[2]]),l).astype(int)
        
        for x in range(len(fullM)-1):
            lineA.append(INA[fullM[x],fullM[x+1]])
        for y in range(len(fullN)-1):
            lineB.append(INA[fullN[y],fullN[y+1]])

        temp += CovLine(lineA,lineB,np.hstack((lineA,lineB)))
    return temp

counter = 0
countmax = p**4

''' Function that sums the covariances over the geometric powers'''
def Psi(i,j,k,l):
    temp = [[Chi(i,j,m,k,l,n) for m in range(GEO)] for n in range(GEO)]
    global counter
    counter +=1
    print('\r\t', np.round(counter/countmax*100,1), '%', end='\r')    #Progress
    return np.sum(temp)



''' Part of the code where covariance is calculated through analytics '''
# Initiate analytical calculation of covariance and compare with sampled covariance
print('Progress analytical calculation:')
SigH = np.array([[[[Psi(i,j,k,l) for l in range(p)] for k in range(p)] for j in range(p)] for i in range(p)])
SigHF = SigH.reshape([p*p,p*p])


# Comparison results
print('\n ** Results ** ')
print('Sampled covariance')
print(sampSigHF)
print('Analytical covariance')
print(SigHF)