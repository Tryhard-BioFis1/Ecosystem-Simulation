import pysindy as ps
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

f = open("DataPySINDy.txt", "r")
s = f.read()
x_s, y_s = '', ''
idx = 1
while s[idx] != ']':
    x_s+=s[idx]; idx+=1
idx+=3
while s[idx] != ']':
    y_s+=s[idx]; idx+=1
x_raw = [int(d) for d in x_s.split(', ')]
y_raw = [int(d) for d in y_s.split(', ')]
x_train = savgol_filter(x_raw, 101, 2)
y_train = savgol_filter(y_raw, 101, 2)

dt = 1e-6

feature_names = ["x","y"]
X = np.stack((x_train, y_train ), axis=-1)
model = ps.SINDy(optimizer = ps.STLSQ(threshold=0.2),
                 feature_library=ps.PolynomialLibrary(degree=4),
                 feature_names=feature_names )
model.fit(X, t=dt, quiet=True )
model.print()

t_sim = np.arange(0,dt*len(x_train), dt)
xy_sim = model.simulate(x0 = (x_train[0], y_train[0]), t=t_sim).T

# Dx = lambda x,y: 2243248.681 + 10024.034 *x + -4906.295 *y + -1.427 *x*x + -40.099 *x *y + -30.471 *y*y
# Dy = lambda x,y: -962412.918 + 1419.449 *x + -6242.624 *y + -0.239 *x*x + 7.999 *x *y + 1.857*y*y
# xi = x_train[0]
# yi = y_train[0]
# app_x_train = []
# app_y_train = []
# for i in range(len(x_train)):
#     app_x_train.append(xi)
#     app_y_train.append(yi)
#     xi = xi + dt*Dx(xi, yi)
#     yi = yi + dt*Dy(xi, yi)


plt.plot(x_raw, y_raw, )
plt.plot(x_train, y_train)
plt.plot(xy_sim[0], xy_sim[1])
# plt.plot(app_x_train, app_y_train)
plt.show()
