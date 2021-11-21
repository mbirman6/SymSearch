## general
import sys
import timeit
## science
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve, minimize
from scipy.integrate import quad
from scipy.stats import norm
## mystuff
sys.path.insert(0,"utils")
from utils3 import *
from plots3 import *
from myfunctions import *

# seed=42
# seed=86
# np.random.seed(seed) # allways same random


N=100000

T1=[]
T2=[]
for m1,m2 in [[10,10],[50,50],[100,100]]:
    T1.append([])
    T2.append([])
    for i in range(N):
        a=np.random.poisson(m1)
        b=np.random.poisson(m2)
        T1[-1].append(a-b)
        T2[-1].append((a-b)/np.sqrt(a+b))
p1=HistsPlot(T1)
p1.xlim=[-80,80]
p1.nbins=80
p1.showplot()
p2=HistsPlot(T2)
p2.xlim=[-5,5]
p2.nbins=40
p2.showplot()

T1=[]
T2=[]
for m1,m2 in [[10,10],[50,50],[100,100]]:
    T1.append([])
    T2.append([])
    for i in range(N):
        a=np.random.poisson(m1)
        b=np.random.poisson(m2)
        T1[-1].append(a-b)
        T2[-1].append((a-b)/np.sqrt(a+b))
p1=HistsPlot(T1)
p1.xlim=[-80,80]
p1.nbins=80
p1.showplot()
p2=HistsPlot(T2)
p2.xlim=[-5,5]
p2.nbins=40
p2.showplot()
