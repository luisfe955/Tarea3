import numpy as np
import os
import matplotlib.pyplot as plt
import glob
archivo = open('datos.dat', 'w')
files = glob.glob("output*.dat")
nombres = []
for f in files:  
    nombres.append(os.path.basename(f).split(".")[0].split("_")[1])
n_files = len(files)
def energy(data):
    E = data[:,6].sum() + data[:,7].sum()    
    return E

def radius(data):
    E_pot = data[:,6]
    min_pot = np.argmin(E_pot)
    #print "min_pot", min_pot
    x = data[:,0] - data[min_pot, 0]
    y = data[:,1] - data[min_pot, 1]
    z = data[:,2] - data[min_pot, 2]
    r = np.sqrt(x**2 + y**2 +z**2)
    r = np.sort(r)
    return r[1:]

i_snap = nombres[0] 
data_init = np.loadtxt("output_{}.dat".format(i_snap))
E_init = energy(data_init)
r_init = radius(data_init)
#print(E_init)


# time ./a.out 450 0.1
i_snap = nombres[1]
data_init = np.loadtxt("output_{}.dat".format(i_snap))
E_final = energy(data_init)
r_final = radius(data_init)
r_final = np.sort(r_final)
log_r_final = np.log10(r_final)
h, c = np.histogram(log_r_final,bins=30)
for i in range(len(h)):
    if h[i] == 0:
       np.delete(h, i)

#print h
log_r_center = 0.5 * (c[1:]+c[:-1])

for i in range(len(log_r_center)):
    #print(str(log_r_center[i])+' '+str(np.log10(h[i])-2.0*log_r_center[i]))
    archivo.write(str(log_r_center[i])+' '+str(np.log10(h[i])-2.0*log_r_center[i])+"\n")
archivo.close()
#    print(str(10**(log_r_center[i]))+' '+str(float(h[i])/float(2.0*log_r_center[i])))
#plt.figure()
#plt.plot(log_r_center, np.log10(h)-2.0*log_r_center)
#plt.plot(x,y)
#plt.show()
