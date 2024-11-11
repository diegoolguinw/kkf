import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.gaussian_process.kernels import Matern
from DynamicalSystems import DynamicalSystem, create_additive_system
from kEDMD import KoopmanOperator
from applyKKKF import apply_koopman_kalman_filter

# Define system parameters
beta, gamma = 0.12, 0.04

# Define system dynamics
def f(x):
    return x + np.array([-beta*x[0]*x[1], beta*x[0]*x[1] - gamma*x[1], gamma*x[1]])

def g(x):
    return np.array([x[1]])

# Setup system dimensions and kernel
N = 300
nx, ny = 3, 1
k = Matern(length_scale=N**(-1/nx), nu=0.5)

# Setup distributions
X_dist = stats.dirichlet(alpha=1*np.ones(nx))
dyn_dist = stats.multivariate_normal(mean=np.zeros(3), cov=1e-5*np.eye(3))
obs_dist = stats.multivariate_normal(mean=np.zeros(1), cov=1e-3*np.eye(1))

# Create dynamical system
dyn = DynamicalSystem(nx, ny, f, g, X_dist, dyn_dist, obs_dist)

iters = 100

x0 = np.array([0.9,0.1,0.0])
x = np.zeros((iters, nx))
y = np.zeros((iters, ny))

x[0] = x0
y[0] = g(x[0]) + obs_dist.rvs()

for i in range(1, iters):
    x[i] = f(x[i-1]) + dyn.dist_dyn.rvs()
    y[i] = g(x[i]) + obs_dist.rvs()

# Initialize Koopman operator and Kalman filter
x0_prior = np.array([0.8, 0.15, 0.05])
d0 = stats.multivariate_normal(mean=x0_prior, cov=0.1*np.eye(3))

Koop = KoopmanOperator(k, dyn)
sol = apply_koopman_kalman_filter(Koop, y, d0, N, noise_samples=100)

conf = np.zeros((iters, nx))
for i in range(iters):
   conf[i, :] = np.sqrt(np.diag(sol.Px_plus[i,:,:]))

err1 = sol.x_plus - 1.96*conf
err2 = sol.x_plus + 1.96*conf

labels = ["S (Real)", "I (Real)", "R (Real)"]
colors = ["blue", "red", "green"]

plt.plot(sol.x_plus, label=["S (KKF)", "I (KKF)", "R (KKF)"])

for i in range(nx):
    plt.fill_between(np.arange(iters), err1[:,i], err2[:,i], alpha=0.6)
    plt.scatter(np.arange(iters), x[:,i], label=labels[i], color=colors[i], s=1.4)

plt.xlabel("Días")
plt.ylabel("Población")
plt.title("Estimación de KKF")
plt.legend()
plt.show()
