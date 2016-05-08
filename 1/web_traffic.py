

import scipy as sp
import matplotlib.pyplot as plt
data = sp.genfromtxt("web_traffic.tsv", delimiter = "\t")
xa = data[:,0]
ya= data[:,1]
xa = xa[~sp.isnan(ya)]
ya = ya[~sp.isnan(ya)]
plt.scatter(xa,ya)
plt.xlabel("TIme")
plt.ylabel("hit/Hour")
plt.xticks([w*7*24 for w in range(10)],['week %i'%w for w in range(10)])
plt.autoscale(tight=True)

fp1,a,b,c,d=sp.polyfit(xa,ya,5,full=True) #done variation to find the perfectfitted curve
f1 = sp.poly1d(fp1)
fx = sp.linspace(0,xa[-1],1000)
plt.plot(fx,f1(fx),linewidth=4)
plt.show()
