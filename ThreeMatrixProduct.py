''' Sietske Lensen 2021
    Code accompanying Master thesis:
    Analytical approach of Sensitivity Analysis for Life Cycle Assessment '''

import numpy as np
import ComplexMultivariateNormal as CMN #retrieved from
np.set_printoptions(precision=3)
np.random.seed()


''' This code calculates the Covariance of a product of three matrices: A (nxp) and B (pxq) and C (qxr) whose coefficients are correlated and distributed lognormally. The covariance is calculated first through simulation and second though an analytical formula. The output matrix is called H (pxq)'''
# Initializing program input parameters
n,p,q,r = 3,3,4,3         # Matrix dimensions
MD = [[n,p],[p,q],[q,r]]  # Dimensions of each matrix
MS = [0,n*p,p*q,q*r]      # Coefficients per matrix
MC = np.cumsum(MS)        # Cumulative sum
M = np.sum(MS)            # Total cum of input coefficient
N = 100000                # Number of simulations
P = 0.2                   # Ratio of zero means


# Position of matrix coefficients in full list of coefficients. Later used for fancy indexing
IND = np.arange(M)                        # All indices
INA = np.reshape(IND[MC[0]:MC[1]],MD[0])  # Indices of A
INB = np.reshape(IND[MC[1]:MC[2]],MD[1])  # Indices of B
INC = np.reshape(IND[MC[2]:MC[3]],MD[2])  # Indices of C


# Picking random parameters for the input coefficients
mu = np.random.normal(size=M)*5
mu *= np.random.choice([0,1],size=M,p=[P,1-P])
mumu = np.multiply.outer(mu,mu)
preSigma = (np.random.rand(M,M)-1/2)/5
sigma = mumu * (np.exp(np.matmul(preSigma,preSigma.T)) - 1)


# Functions
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


''' Part of the code where covariance is calculated through sampling '''
# The lognormal sample is constructed by taking the exponent of a (complex) normal sample, for which the parameter relationships are as in the above functions
# Using the package ComplexMultivariateNormal source: (\url)
SamplingMu,SamplingSigma = LNToNorm(mu,sigma)   # Find normal parameters for initialized lognormal parameters
SamplingSigmaConj = np.triu(SamplingSigma)+np.conj(np.tril(SamplingSigma,-1))
samples = np.real(np.exp(CMN.ScipyComplexMultivariateNormal(SamplingMu,SamplingSigma,SamplingSigmaConj).rvs(N)))

# Model calculation from samples
sampA = samples[:,INA]      # A coefficients in matrices
sampB = samples[:,INB]      # B coefficients in matrices
sampC = samples[:,INC]      # C coefficients in matrices
sampH = np.matmul(sampA,np.matmul(sampB,sampC)) # Matrix product
sampSigHF = np.cov(np.reshape(sampH,[N,n*r]).T) # Model covariance
sampSigH = np.reshape(sampSigHF,[n,r,n,r])  # Model covariance tensor
print('Sampling done ')



''' Function selects the means and covariances of a set of coefficients'''
def LineToMuSigma(subset):
    return mu[subset], sigma[subset[:,None],subset]


''' Function to calculate covariance of a product within large sum'''
def ThreeMatCov(i,x1,x2,j,k,y1,y2,l):
    MuAB, SigmaAB = LineToMuSigma(np.array([INA[i,x1],INB[x1,x2],INC[x2,j],INA[k,y1],INB[y1,y2],INC[y2,l]]))
    MuA, SigmaA = LineToMuSigma(np.array([INA[i,x1],INB[x1,x2],INC[x2,j]]))
    MuB, SigmaB = LineToMuSigma(np.array([INA[k,y1],INB[y1,y2],INC[y2,l]]))
    return ProductParameters(MuAB,SigmaAB)[0]  - ProductParameters(MuA,SigmaA)[0]*ProductParameters(MuB,SigmaB)[0]

counter = 0
countmax = r*n*r*n

''' Function to calculate covariance of one output pair'''
def SingleCovariance(i,j,k,l):
    temp = [[[[ThreeMatCov(i,x1,x2,j,k,y1,y2,l) for y2 in range(q)] for y1 in range(p)] for x2 in range(q)] for x1 in range(p)]
    global counter
    counter += 1
    print('\r\t',np.round(counter/countmax*100,1),'%',end='\r')
    return np.sum(temp)

''' Function to organize the output covariances into matrix'''
def TotalCovariance():
    return [[[[SingleCovariance(i,j,k,l) for l in range(r)] for k in range(n)] for j in range(r)] for i in range(n)]



''' Part of the code where covariance is calculated through analytics '''
# Calculate the analytical covariance
print('Progress analytical calculation:')
SigH = TotalCovariance()            # Model covariance tensor
SigHF = np.reshape(SigH,[n*r,n*r])  # Model covariance



# Comparison results
print('\n ** Results ** ')
print('Sampled covariance')
print(sampSigHF)
print('Analytical covariance')
print(SigHF)