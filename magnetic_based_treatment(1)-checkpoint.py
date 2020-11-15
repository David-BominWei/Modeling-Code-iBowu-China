'''
!/usr/bin/env python
 coding: utf-8
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.integrate import odeint
get_ipython().run_line_magic('matplotlib', 'inline')


# parameter setting
L = 20
H = 4
d = 3
frac = 6

vc = 4
alpha = 4*vc/H**2
gamma = 1
chi = 0.2

dt = 0.1
T = 100


# fluid field
def fluid_velocity(z):
    #return vc
    return vc - alpha * z**2 + 0.1


def NewtonianEquation(y, t):
    dydt = np.zeros(4)
    
    dydt[0] = y[2]
    dydt[1] = y[3]
    dydt[2] = gamma*(fluid_velocity(y[1]) - y[2])
    dydt[3] = 0
    
    if (y[0] >= -L/frac and y[0] <= L/frac):
        dydt[2] -= 2*chi*y[0]/(y[0]**2 + (d-y[1])**2)**2
        dydt[3] += 2*chi*(d-y[1])/(y[0]**2 + (d-y[1])**2)**2
    
    return dydt


def makeFig(x, y, v, id):
    gx = np.linspace(-L/2, L/2, 100)
    gy = np.linspace(-H/2, H/2, 100)
    X1, X2 = np.meshgrid(gx, gy)
    Y = fluid_velocity(X2)

    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)
    cm1 = plt.cm.get_cmap('viridis')
    cm2 = plt.cm.get_cmap('bwr')
    ax.scatter(X1, X2, c=Y, cmap=cm1)
    ax.scatter(x, y, c=v, cmap=cm2, vmin=0, vmax=5)
    rect = patches.Rectangle((-L/frac,0), 2*L/frac, H/2, alpha=0.3, fc='red', ec=None)
    ax.add_patch(rect)
    ax.set_xlim([-L/2, L/2])
    ax.set_ylim([-H/2, H/2])
    plt.savefig('fig'+str(id)+'.png')
    plt.show()


#y0 = np.zeros(4)
t = 0

i = 0
snapframe = 2

px0 = np.linspace(-L/2+0.1, L/2-0.1, 10)
#px0 = px0[:-1]
py0 = np.linspace(-H/2+0.1, H/2-0.1, 10)
px0, py0 = np.meshgrid(px0, py0)
px0 = px0.flatten()
py0 = py0.flatten()

y0list = []
count = []

for j in range(len(px0)):
    y0list.append([px0[j], py0[j], 0, 0])

while(t<T):
    xlist = []
    ylist = []
    vlist = []
       
    for j in range(len(px0)):
        y = odeint(NewtonianEquation, y0list[j], np.linspace(0, dt, 100))
        y0list[j] = y[-1]

        if y0list[j][0] > L/2:
            y0list[j][0] -= L
        elif y0list[j][0] < -L/2:
            y0list[j][0] += L

        if y0list[j][1] > H/2:
            y0list[j][1] = H/2
            y0list[j][3] *= -.1
        elif y0list[j][1] < -H/2:
            y0list[j][1] = -H/2
            y0list[j][3] *= -.1
            
        xlist.append(y0list[j][0])
        ylist.append(y0list[j][1])
        vlist.append(np.sqrt((y0list[j][2:]**2).sum()))
    
    if (i%snapframe == 0):
        index = int(i/snapframe)
        makeFig(xlist, ylist, vlist, index)
     
    
    count.append(((np.array(xlist) > -L/frac)*(np.array(xlist) < L/frac)*(np.array(ylist) > 0).astype(int)).sum())
    
    t += dt
    i += 1



plt.plot(count)


plt.plot(count)

count1 = count
count1 = np.array(count1)

count2 = count
count2 = np.array(count2)


def smooth(x,window_len=100,window='hanning'):

    if window_len<3:
        return x

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


smo1 = smooth(count1)
smo2 = smooth(count2)

plt.plot(smo1, linewidth=3, label='magnetic+')
plt.plot(smo2, linewidth=3, label='magnetic-')
plt.legend(fontsize=10)



