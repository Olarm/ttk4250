# %% imports
import scipy
import scipy.io
import scipy.stats

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try: # see if tqdm is available, otherwise define it as a dummy
    try: # Ipython seem to require different tqdm.. try..except seem to be the easiest way to check
        __IPYTHON__
        #from tqdm.notebook import tqdm
        import tqdm
    except:
        #from tqdm import tqdm
        import tqdm
except Exception as e:
    print(e)
    print(
        "install tqdm (conda install tqdm, or pip install tqdm) to get nice progress bars. "
    )

    def tqdm(iterable, *args, **kwargs):
        return iterable

from eskf import (
    ESKF,
    POS_IDX,
    VEL_IDX,
    ATT_IDX,
    ACC_BIAS_IDX,
    GYRO_BIAS_IDX,
    ERR_ATT_IDX,
    ERR_ACC_BIAS_IDX,
    ERR_GYRO_BIAS_IDX,
)

from quaternion import quaternion_to_euler
from cat_slice import CatSlice

# %% plot config check and style setup


# to see your plot config
print(f"matplotlib backend: {matplotlib.get_backend()}")
print(f"matplotlib config file: {matplotlib.matplotlib_fname()}")
print(f"matplotlib config dir: {matplotlib.get_configdir()}")
plt.close("all")

# try to set separate window ploting
if "inline" in matplotlib.get_backend():
    print("Plotting is set to inline at the moment:", end=" ")

    if "ipykernel" in matplotlib.get_backend():
        print("backend is ipykernel (IPython?)")
        print("Trying to set backend to separate window:", end=" ")
        import IPython

        IPython.get_ipython().run_line_magic("matplotlib", "")
    else:
        print("unknown inline backend")

print("continuing with this plotting backend", end="\n\n\n")


# set styles
try:
    # installed with "pip install SciencePLots" (https://github.com/garrettj403/SciencePlots.git)
    # gives quite nice plots
    plt_styles = ["science", "grid", "bright", "no-latex"]
    plt.style.use(plt_styles)
    print(f"pyplot using style set {plt_styles}")
except Exception as e:
    print(e)
    print("setting grid and only grid and legend manually")
    plt.rcParams.update(
        {
            # setgrid
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.color": "k",
            "grid.alpha": 0.5,
            "grid.linewidth": 0.5,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": True,
            "legend.numpoints": 1,
        }
    )

# %% load data and plot
filename_to_load = "task_real.mat"
loaded_data = scipy.io.loadmat(filename_to_load)

do_corrections = True # TODO: set to false for the last task
if do_corrections:
    S_a = loaded_data['S_a']
    S_g = loaded_data['S_g']
else:
    # Only accounts for basic mounting directions
    S_a = S_g = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

lever_arm = loaded_data["leverarm"].ravel()
timeGNSS = loaded_data["timeGNSS"].ravel()
timeIMU = loaded_data["timeIMU"].ravel()
z_acceleration = loaded_data["zAcc"].T
z_GNSS = loaded_data["zGNSS"].T
z_gyroscope = loaded_data["zGyro"].T
accuracy_GNSS = loaded_data['GNSSaccuracy'].ravel()

dt = np.mean(np.diff(timeIMU))
steps = len(z_acceleration) 
gnss_steps = len(z_GNSS)

# %% Measurement noise
# Continous noise

cont_gyro_noise_std = 1*4.36e-5  # (rad/s)/sqrt(Hz)
cont_acc_noise_std = 0.8*1.167e-3  # (m/s**2)/sqrt(Hz)

# Discrete sample noise at simulation rate used
rate_std = cont_gyro_noise_std * np.sqrt(1 / dt)
acc_std = cont_acc_noise_std * np.sqrt(1 / dt)

# Bias values
rate_bias_driving_noise_std = 5e-5
cont_rate_bias_driving_noise_std =  rate_bias_driving_noise_std / np.sqrt(1 / dt)

acc_bias_driving_noise_std = 4e-3
cont_acc_bias_driving_noise_std = acc_bias_driving_noise_std / np.sqrt(1 / dt)

# Position and velocity measurement
p_std = np.array([0.2, 0.2, 0.4]) # Measurement noise
p_std = np.diag(p_std)**2

p_acc = 1e-12

p_gyro = 1e-12

# %% Estimator
eskf = ESKF(
    acc_std,
    rate_std,
    cont_acc_bias_driving_noise_std,
    cont_rate_bias_driving_noise_std,
    p_acc,
    p_gyro,
    S_a = S_a, # set the accelerometer correction matrix
    S_g = S_g, # set the gyro correction matrix,
    debug=False # False to avoid expensive debug checks
)


# %% Allocate
x_est = np.zeros((steps, 16))
P_est = np.zeros((steps, 15, 15))

x_pred = np.zeros((steps, 16))
P_pred = np.zeros((steps, 15, 15))

NIS = np.zeros(gnss_steps)
NIS_x = np.zeros(gnss_steps)
NIS_y = np.zeros(gnss_steps)
NIS_z = np.zeros(gnss_steps)
NIS_xy = np.zeros(gnss_steps)

# %% Initialise
x_pred[0, POS_IDX] = np.array([0, 0, 0]) # starting 5 metres above ground
x_pred[0, VEL_IDX] = np.array([0, 0, 0]) # starting at 20 m/s due north
x_pred[0, ATT_IDX] = np.array([
    np.cos(45 * np.pi / 180),
    0, 0,
    np.sin(45 * np.pi / 180)
])  # nose to east, right to south and belly down.

P_pred[0][POS_IDX**2] = 10**2 * np.eye(3)
P_pred[0][VEL_IDX**2] = 3**2 * np.eye(3)
P_pred[0][ERR_ATT_IDX**2] = (np.pi/30)**2 * np.eye(3) # error rotation vector (not quat)
P_pred[0][ERR_ACC_BIAS_IDX**2] = 0.05**2 * np.eye(3)
P_pred[0][ERR_GYRO_BIAS_IDX**2] = (1e-3)**2 * np.eye(3)

# %% Run estimation

N = steps
GNSSk = 0

for k in tqdm.trange(N):
    if timeIMU[k] >= timeGNSS[GNSSk]:
        R_GNSS = p_std * accuracy_GNSS[GNSSk] # TODO: Current GNSS covariance
        
        (
            NIS[GNSSk], 
            NIS_x[GNSSk],
            NIS_y[GNSSk],
            NIS_z[GNSSk],
            NIS_xy[GNSSk],
        ) = eskf.NIS_GNSS_position(x_pred[k], P_pred[k], z_GNSS[GNSSk], R_GNSS, lever_arm)
        
        x_est[k], P_est[k] = eskf.update_GNSS_position(x_pred[k], P_pred[k], z_GNSS[GNSSk], R_GNSS, lever_arm) # TODO
        if eskf.debug:
            assert np.all(np.isfinite(P_est[k])), f"Not finite P_pred at index {k}"

        GNSSk += 1
    else:
        # no updates, so estimate = prediction
        x_est[k] = x_pred[k]# TODO
        P_est[k] = P_pred[k]# TODO

    if k < N - 1:
        x_pred[k + 1], P_pred[k + 1] = eskf.predict(x_est[k], P_est[k], z_acceleration[k+1], z_gyroscope[k+1], dt)# TODO

    if eskf.debug:
        assert np.all(np.isfinite(P_pred[k])), f"Not finite P_pred at index {k + 1}"


# %% Plots

fig1 = plt.figure(1)
ax = plt.axes(projection='3d')

ax.plot3D(x_est[0:N, 1], x_est[0:N, 0], -x_est[0:N, 2])
ax.plot3D(z_GNSS[0:GNSSk, 1], z_GNSS[0:GNSSk, 0], -z_GNSS[0:GNSSk, 2])
ax.set_xlabel('East [m]')
ax.set_xlabel('North [m]')
ax.set_xlabel('Altitude [m]')

plt.grid()

# state estimation
t = np.linspace(0, dt*(N-1), N)
eul = np.apply_along_axis(quaternion_to_euler, 1, x_est[:N, ATT_IDX])

fig2, axs2 = plt.subplots(5, 1, figsize=(10,8))
fig2.tight_layout()

axs2[0].plot(t, x_est[0:N, POS_IDX])
axs2[0].set(ylabel='NED position [m]')
axs2[0].legend(['North', 'East', 'Down'])
plt.grid()

axs2[1].plot(t, x_est[0:N, VEL_IDX])
axs2[1].set(ylabel='Velocities [m/s]')
axs2[1].legend(['North', 'East', 'Down'])
plt.grid()

axs2[2].plot(t, eul[0:N] * 180 / np.pi)
axs2[2].set(ylabel='Euler angles [deg]')
axs2[2].legend(['\phi', '\theta', '\psi'])
plt.grid()

axs2[3].plot(t, x_est[0:N, ACC_BIAS_IDX])
axs2[3].set(ylabel='Accl bias [m/s^2]')
axs2[3].legend(['x', 'y', 'z'])
plt.grid()

axs2[4].plot(t, x_est[0:N, GYRO_BIAS_IDX] * 180 / np.pi * 3600)
axs2[4].set(ylabel='Gyro bias [deg/h]')
axs2[4].legend(['x', 'y', 'z'])
plt.grid()

fig2.suptitle('States estimates')
plt.savefig("report/figures/real_2.eps", format="eps")


# %% Consistency
confprob = 0.95
CI1 = np.array(scipy.stats.chi2.interval(confprob, 1)).reshape((2, 1))
CI1N = np.array(scipy.stats.chi2.interval(confprob, 1 * N)) / N
CI2 = np.array(scipy.stats.chi2.interval(confprob, 2)).reshape((2, 1))
CI2N = np.array(scipy.stats.chi2.interval(confprob, 2 * N)) / N
CI3 = np.array(scipy.stats.chi2.interval(confprob, 3)).reshape((2, 1))
CI3N = np.array(scipy.stats.chi2.interval(confprob, 3 * N)) / N

ANIS_x = NIS_x[:GNSSk].mean()
ANIS_y = NIS_y[:GNSSk].mean()
ANIS_z = NIS_z[:GNSSk].mean()
ANIS_xy = NIS_xy[:GNSSk].mean()

insideCI = np.mean((CI3[0] <= NIS[:GNSSk]) * (NIS[:GNSSk] <= CI3[1]))
insideCIx = np.mean((CI1[0] <= NIS_x[:GNSSk]) * (NIS_x[:GNSSk] <= CI1[1]))
insideCIy = np.mean((CI1[0] <= NIS_y[:GNSSk]) * (NIS_y[:GNSSk] <= CI1[1]))
insideCIz = np.mean((CI1[0] <= NIS_z[:GNSSk]) * (NIS_z[:GNSSk] <= CI1[1]))
insideCIxy = np.mean((CI2[0] <= NIS_xy[:GNSSk]) * (NIS_xy[:GNSSk] <= CI2[1]))

fig3, axs3 = plt.subplots(2, 1, clear=True, figsize=(10,3))
fig3.tight_layout()

axs3[0].plot(NIS[:GNSSk])
axs3[0].plot(np.array([0, N-1]) * dt, (CI3@np.ones((1, 2))).T)
axs3[0].set(
    title=f"NIS ({100 *  insideCI:.1f} inside {100 * confprob} confidence interval)"
)

axs3[1].plot(NIS_x[:GNSSk])
axs3[1].plot(NIS_y[:GNSSk])
axs3[1].plot(NIS_z[:GNSSk])
axs3[1].plot(np.array([0, N-1]) * dt, (CI1@np.ones((1, 2))).T)
axs3[1].set(
    title=f"NIS inside {100 * confprob} confidence interval:  x = {100 *  insideCIx:.1f},  y = {100 *  insideCIy:.1f},  z = {100 *  insideCIz:.1f}"
)


plt.savefig("report/figures/real_3.eps", format="eps")

# %% box plots
fig4 = plt.figure()

gauss_compare = np.sum(np.random.randn(3, GNSSk)**2, axis=0)
plt.boxplot([NIS[0:GNSSk], gauss_compare], notch=True)
plt.legend('NIS', 'gauss')
plt.grid()

# %%
