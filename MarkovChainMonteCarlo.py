from scipy.stats import norm
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import matplotlib.mlab as mlab

data=np.loadtxt("datos.dat")
n = data.shape[0]
x=data[0:n,0]
y=data[0:n,1]
def loglikelihood(x, y, sigmay, A, B,C,D):
    fuNCion=np.log10((10**A)/(((10**x/10**B)**C)*((1+10**x/10**B)**D)))
    return (np.sum(np.log(1./(np.sqrt(2.*np.pi) * sigmay))) +
            np.sum(-0.5 * (y - fuNCion)**2 / sigmay**2))
    
def logpriori(m, B):
    return 0.
    
def logposterior(x,y,sigmay, m,B):
    return (loglikelihood(x,y,sigmay, A,B,C,D) +
            logpriori(m, B))

def posterior(x, y, sigmay, m, B):
    return np.exp(logposterior(x, y, sigmay, A, B,C,D))

# first step
A = 1.0
B = 0.0
C = 0.0
D = 0.0

# step sizes
Astep = 1.5
Bstep = 0.5
Cstep = 1.0
Dstep = 0.1
        
#Number of steps
nsteps = 10000
    
chain = []
probs = []
NAcept = 0
    
#print 'Running MH for', nsteps, 'steps'
sigmay=1
# First point:
L_old    = loglikelihood(x, y, sigmay, A, B, C, D)
p_old    = logpriori(A, B)
prob_old = np.exp(L_old + p_old)

for i in range(nsteps):
    # step
    Anew_value = A + np.random.normal() * Astep
    Bnew_value = B + np.random.normal() * Bstep
    Cnew_value = C + np.random.normal() * Cstep
    Dnew_value = D + np.random.normal() * Dstep

    # loglikelihood

    L_new    = loglikelihood(x, y, sigmay, Anew_value, Bnew_value, Cnew_value, Dnew_value)
    p_new    = logpriori(Anew_value, Bnew_value)
    prob_new = np.exp(L_new + p_new)

    if (prob_new / prob_old > np.random.uniform()):
        # acept
        A = Anew_value
        B = Bnew_value
        C = Cnew_value
        D = Dnew_value
        L_old = L_new
        p_old = p_new
        prob_old = prob_new
        NAcept += 1
    else:
        pass

    chain.append((A,B,C,D))
    probs.append((L_old,p_old))
a = [A for A,B,C,D in chain]
b = [B for A,B,C,D in chain]
c = [C for A,B,C,D in chain]
d = [D for A,B,C,D in chain]

plt.figure(1)
NA,binA,patchesa=plt.hist(a,15,normed=True)
(Media_A,sigmaA)=norm.fit(a)
ya=norm.pdf(binA,Media_A,sigmaA)
plt.xlabel('A')
plt.plot(binA,ya,'r--')
print '---------------------------------------------------------'
print 'El valor de log(A)=log(p0) es ', Media_A, '+-', sigmaA
figure(1).savefig("A.png")
#plt.close()

plt.figure(2)
NB,binB,patchesb=plt.hist(b,15,normed=True)
(Media_B,sigmaB)=norm.fit(b)
yb=norm.pdf(binB,Media_B,sigmaB)
plt.xlabel('B')
plt.plot(binB,yb,'r--')
print 'El valor de log(B)=log(rc) es ', Media_B, '+-', sigmaB
figure(2).savefig("B.png")

plt.figure(3)
NC,binC,patchesc=plt.hist(c,bins=15,normed=True)
(Media_C,sigmaC)=norm.fit(c)
yc=norm.pdf(binC,Media_C,sigmaC)
plt.xlabel('C')
plt.plot(binC,yc,'r--')
print 'El valor de C=alpha es ', Media_C, '+-', sigmaC
figure(3).savefig("C.png")

plt.figure(4)
ND,binD,patchesd=plt.hist(d,15,normed=True)
(Media_D,sigmaD)=norm.fit(d)
yd=norm.pdf(binD,Media_D,sigmaD)
plt.xlabel('D')
plt.plot(binD,yd,'r--')
print 'El valor de D=beta es ', Media_D, '+-', sigmaD
print '---------------------------------------------------------'
figure(4).savefig("D.png")

plt.figure(5)
plt.scatter(x,y)
x1=np.linspace(-1.5,2.5,100)
y1=np.log10(10**Media_A/(((10**x1/10**(Media_B))**(Media_C))*((1+10**x1/10**(Media_B))**(Media_D))))
plt.xlabel('log(r)')
plt.ylabel('log(p)')
plt.scatter(x1,y1)
figure(5).savefig("perfil_densidad.png")
#plt.legend("datos")
plt.show()
