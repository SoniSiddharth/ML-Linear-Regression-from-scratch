import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import imageio
import os
from preprocessing.polynomial_features import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from IPython.display import HTML, Image


# ----------------------------------- Line graph ------------------------------------
print("------------------------------ Line graph --------------------------------------")

# dataset generation
N = 20
X = np.random.randn(N, 1)
y = pd.Series(np.matrix(X+np.array([[5]]*N)).transpose().tolist()[0])
X = pd.DataFrame(X)

LR = LinearRegression(fit_intercept=True)
LR.fit_vectorised(X, y, batch_size=20,n_iter=300)
y_hat = LR.predict(X)

# take the images of the plots
temp_files = []
for i in range(0,300,10):
    plt.close()
    plt.axis([-2,5,-2,10]) 
    plt.scatter(X.to_numpy(), y.tolist(), color = "red")
    plt.plot(X.to_numpy(), LR.predict_plot(LR.data_used,LR.theta_history[i]), color = "brown")
    plt.title("Linear Regression plot")
    plt.xlabel("feature values")
    plt.ylabel("Predicted Y values")
    temp_files.append("./temp_images/{}.png".format(i))
    plt.savefig("./temp_images/{}.png".format(i))

# taking each image and creating a gif
with imageio.get_writer('line_plot.gif', mode='I') as writer:
    for name in temp_files:
        image = imageio.imread(name)
        writer.append_data(image)

for name in set(temp_files):
    os.remove(name)

# ----------------------------------------------- contour --------------------------------------
print("----------------------------------- Contour plot ---------------------------------------")

plt.style.use('seaborn-white')

def costFunctionReg(X,y,theta,lamda = 10):    
    m = len(y) 
    J = 0
    h = X @ theta
    J_reg = (lamda / (2*m)) * np.sum(np.square(theta))
    J = float((1./(2*m)) * (h - y).T @ (h - y)) + J_reg
    if np.isnan(J):
        return(np.inf)    
    return(J)

x = np.array([i*np.pi/180 for i in range(60,300,4)])
y = 4*x + 7 + np.random.normal(0,3,len(x))

X_data = x
X_data = pd.DataFrame(X_data)
y = pd.Series(y)

LR = LinearRegression(fit_intercept=True)
LR.plot_contour(X_data, y, [], [])
thetas = LR.thetas
X = LR.data_used

y_noise = y.to_numpy()
# print(y_noise.shape)

# setting up meshgrid
l = 25
T1, T2 = np.meshgrid(np.linspace(-3,8,100),np.linspace(2,10,100))
zs = np.array([costFunctionReg(X, y_noise.reshape(-1,1),np.array([t1,t2]).reshape(-1,1),l) 
                    for t1, t2 in zip(np.ravel(T1), np.ravel(T2))])
Z = zs.reshape(T1.shape)

coeff_1 = [i[0] for i in LR.thetas]
coeff_2 = [i[1] for i in LR.thetas]

theta_0 = []
theta_1 = []

for i in range (len(coeff_1)):
    theta_0.append(coeff_1[i][0])
for i in range (len(coeff_2)):
    theta_1.append(coeff_2[i][0])

matplotlib.rcParams['animation.embed_limit'] = 2**128

#Plot the contour
fig1, ax1 = plt.subplots(figsize = (7,7))
ax1.contour(T1, T2, Z, 100, cmap = 'jet')

# create animation
line, = ax1.plot([], [], 'r', label = 'Gradient descent', lw = 1.5)
point, = ax1.plot([], [], '*', color = 'red', markersize = 4)
value_display = ax1.text(0.02, 0.02, '', transform=ax1.transAxes)

theta_0 = [i[0] for i in LR.thetas]
theta_1 = [i[1] for i in LR.thetas]

def init_1():
    line.set_data([], [])
    point.set_data([], [])
    value_display.set_text('')
    return line, point, value_display

def animate_1(i):
    line.set_data(theta_0[:i], theta_1[:i])
    point.set_data(theta_0[i], theta_1[i])
    return line, point, value_display

ax1.legend(loc = 1)
anim1 = animation.FuncAnimation(fig1, animate_1, init_func=init_1,
                               frames=len(theta_0), interval=100, 
                               repeat_delay=60, blit=True)
HTML(anim1.to_jshtml())
anim1.save('contour_plot.gif', writer='Pillow', fps=10)


# ------------------------------------------ surface plot ----------------------------------------------
print("-------------------------------------- Surface plot --------------------------------------------")

coeff_1 = [i[0] for i in LR.thetas]
coeff_2 = [i[1] for i in LR.thetas]

theta_0 = []
theta_1 = []

for i in range (len(coeff_1)):
    theta_0.append(coeff_1[i][0])
for i in range (len(coeff_2)):
    theta_1.append(coeff_2[i][0])

fig2 = plt.figure(figsize = (7,7))
ax2 = Axes3D(fig2)

#Surface plot
ax2.plot_surface(T1, T2, Z, rstride = 5, cstride = 5, cmap = 'jet', alpha=0.5)

ax2.set_xlabel('theta 1')
ax2.set_ylabel('theta 2')
ax2.set_zlabel('error')
ax2.view_init(45, -45)

# Create animation
line2, = ax2.plot([], [], [], 'r-', label = 'Gradient descent', lw = 1.5)
point2, = ax2.plot([], [], [], '*', color = 'red')
display_value2 = ax2.text(2., 2., 27.5, '', transform=ax1.transAxes)

def init_2():
    line2.set_data_3d([], [], [])
    point2.set_data_3d([], [], [])
    return line2, point2

def animate_2(i):
    line2.set_data_3d(theta_0[:i], theta_1[:i], theta_0[:i])
    point2.set_data_3d(theta_0[i], theta_1[i], theta_0[i])
    return line2, point2, display_value2

ax2.legend(loc = 1)
anim2 = animation.FuncAnimation(fig2, animate_2, init_func=init_2,
                               frames=len(theta_0), interval=120, 
                               repeat_delay=60, blit=True)

anim2.save('surface_plot.gif', writer='Pillow', fps=10)
plt.show()

# References - https://xavierbourretsicotte.github.io/animation_ridge.html