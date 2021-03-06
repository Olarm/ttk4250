# %% Imports
from scipy.io import loadmat
from scipy.stats import chi2

try:
    from tqdm import tqdm
except ImportError as e:
    print(e)
    print("install tqdm for progress bar")

    # def tqdm as dummy
    def tqdm(*args, **kwargs):
        return args[0]


import numpy as np
from EKFSLAM import EKFSLAM
import matplotlib
import matplotlib.pyplot as plt
import pylab as plb
from matplotlib import animation
from plotting import ellipse
from vp_utils import detectTrees, odometry, Car
from utils import rotmat2d, find_nearest
from sklearn.decomposition import PCA

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
    #plt_styles = ["science", "grid", "bright", "no-latex"]
    plt_styles = ["science", "grid", "bright"]
    plt.style.use(plt_styles)
    print(f"pyplot using style set {plt_styles}")
except Exception as e:
    print(e)
    print("setting grid and only grid and legend manually")
    plt.rcParams.update(
        {
            #"font.size":6,
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


# %% Load data
VICTORIA_PARK_PATH = "./victoria_park/"
realSLAM_ws = {
    **loadmat(VICTORIA_PARK_PATH + "aa3_dr"),
    **loadmat(VICTORIA_PARK_PATH + "aa3_lsr2"),
    **loadmat(VICTORIA_PARK_PATH + "aa3_gpsx"),
}

timeOdo = (realSLAM_ws["time"] / 1000).ravel()
timeLsr = (realSLAM_ws["TLsr"] / 1000).ravel()
timeGps = (realSLAM_ws["timeGps"] / 1000).ravel()

steering = realSLAM_ws["steering"].ravel()
speed = realSLAM_ws["speed"].ravel()
LASER = (
    realSLAM_ws["LASER"] / 100
)  # Divide by 100 to be compatible with Python implementation of detectTrees
La_m = realSLAM_ws["La_m"].ravel()
Lo_m = realSLAM_ws["Lo_m"].ravel()

K = timeOdo.size
mK = timeLsr.size
Kgps = timeGps.size

# %% Parameters

L = 2.83  # axel distance
H = 0.76  # center to wheel encoder
a = 0.95  # laser distance in front of first axel
b = 0.5  # laser distance to the left of center

car = Car(L, H, a, b)

sigmas = [0.07, 0.07, 0.3/180*np.pi]
CorrCoeff = np.array([[1, 0, 0], [0, 1, 0.9], [0, 0.9, 1]])
Q = np.diag(sigmas) @ CorrCoeff @ np.diag(sigmas)

R = np.diag([0.005, 0.003/180*np.pi])

R_gps = np.diag([0.5, 0.5])

JCBBalphas = np.array([5e-13, 5e-10])

sensorOffset = np.array([car.a + car.L, car.b])
doAsso = True

slam = EKFSLAM(Q, R, R_gps=R_gps, do_asso=doAsso, alphas=JCBBalphas, sensor_offset=sensorOffset)

# For consistency testing
alpha = 0.05
confidence_prob = 1 - alpha

xupd = np.zeros((mK, 3))
a = [None] * mK
NIS = np.zeros(mK)
NISnorm = np.zeros(mK)
CI = np.zeros((mK, 2))
CInorm = np.zeros((mK, 2))

# Initialize state
eta = np.array([Lo_m[0], La_m[1], 36 * np.pi / 180]) # you might want to tweak these for a good reference
P = np.zeros((3, 3))

mk_first = 1  # first seems to be a bit off in timing
mk = mk_first
t = timeOdo[0]

# %%  run
N = 1000 #K
GPSi1, GPSk2, GPSi2 = 0,0,0

doPlot = False

lh_pose = None

do_GPS_NIS = True
if do_GPS_NIS:
    GPS_NIS = np.zeros((timeGps.shape))
    GPS_NEES = np.zeros((timeGps.shape))

if doPlot:
    fig, ax = plt.subplots(num=1, clear=True)

    lh_pose = ax.plot(eta[0], eta[1], "k", lw=3)[0]
    sh_lmk = ax.scatter(np.nan, np.nan, c="r", marker="x")
    sh_Z = ax.scatter(np.nan, np.nan, c="b", marker=".")

do_raw_prediction = True
if do_raw_prediction:  # TODO: further processing such as plotting
    odos = np.zeros((K, 3))
    odox = np.zeros((K, 3))
    odox[0] = eta

    for k in range(min(N, K - 1)):
        odos[k + 1] = odometry(speed[k + 1], steering[k + 1], 0.025, car)
        odox[k + 1], _ = slam.predict(odox[k], P, odos[k + 1])

P = np.zeros((3, 3))

for k in tqdm(range(N)):
    if mk < mK - 1 and timeLsr[mk] <= timeOdo[k + 1]:
        # Force P to symmetric: there are issues with long runs (>10000 steps)
        # seem like the prediction might be introducing some minor asymetries,
        # so best to force P symetric before update (where chol etc. is used).
        # TODO: remove this for short debug runs in order to see if there are small errors
        P = (P + P.T) / 2
        dt = timeLsr[mk] - t
        if dt < 0:  # avoid assertions as they can be optimized avay?
            raise ValueError("negative time increment")

        t = timeLsr[mk]  # ? reset time to this laser time for next post predict
        odo = odometry(speed[k + 1], steering[k + 1], dt, car)
        eta, P = slam.predict(eta, P, odo) # TODO predict

        z = detectTrees(LASER[mk])
        eta, P, NIS[mk], a[mk] = slam.update(eta, P, z)# TODO update

        num_asso = np.count_nonzero(a[mk] > -1)

        if num_asso > 0:
            NISnorm[mk] = NIS[mk] / (2 * num_asso)
            CInorm[mk] = np.array(chi2.interval(confidence_prob, 2 * num_asso)) / (
                2 * num_asso
            )
        else:
            NISnorm[mk] = 1
            CInorm[mk].fill(1)

        xupd[mk] = eta[:3]

        if doPlot:
            sh_lmk.set_offsets(eta[3:].reshape(-1, 2))
            if len(z) > 0:
                zinmap = (
                    rotmat2d(eta[2])
                    @ (
                        z[:, 0] * np.array([np.cos(z[:, 1]), np.sin(z[:, 1])])
                        + slam.sensor_offset[:, None]
                    )
                    + eta[0:2, None]
                )
                sh_Z.set_offsets(zinmap.T)
            lh_pose.set_data(*xupd[mk_first:mk, :2].T)

            ax.set(
                xlim=[-200, 200],
                ylim=[-200, 200],
                title=f"step {k}, laser scan {mk}, landmarks {len(eta[3:])//2},\nmeasurements {z.shape[0]}, num new = {np.sum(a[mk] == -1)}",
            )
            plt.draw()
            plt.pause(0.00001)

        mk += 1

    if k < K - 1:
        dt = timeOdo[k + 1] - t
        t = timeOdo[k + 1]
        odo = odometry(speed[k + 1], steering[k + 1], dt, car)
        
        if np.allclose(timeGps[GPSk2], timeOdo[k], atol=1e-1):
            error = eta[:2] - [Lo_m[GPSk2], La_m[GPSk2]]
            GPS_NEES[GPSi2] = error @ P[:2,:2] @ error
            GPSi2 += 1
            GPSk2 += 1
        elif (timeGps[GPSk2] + 0.1) < timeOdo[k]:
            GPSk2 += 1
            
        eta, P = slam.predict(eta, P, odo)
        
        GPS_idx = (np.abs(timeGps - timeOdo[k])).argmin()
        if np.allclose(timeGps[GPS_idx], timeOdo[k], atol=1e-2):
            z_GPS = [Lo_m[GPS_idx], La_m[GPS_idx]]
            GPS_NIS[GPSi1] = slam.GNSS_NIS(eta[:2], P[:2,:2], z_GPS)
            GPSi1 += 1

        

# %% Consistency

l = timeGps[timeGps < timeLsr[mk-1]].shape[0]
idxs = np.array([], dtype="int")
short_idxs = np.array([], dtype="int")
exclude = np.array([], dtype="int")
i = 0
for j, value in enumerate(timeGps[timeGps < timeLsr[mk-1]]):
    index = find_nearest(timeLsr[:mk-1], value)
    idxs = np.append(idxs, index)
    if index not in short_idxs:
        short_idxs = np.append(short_idxs, index)
        i += 1
    else:
        exclude = np.append(exclude, j)
        
short_idxs = np.delete(short_idxs, 0)
gps_mask = np.ones(La_m.shape[0], dtype=bool)
gps_mask[exclude] = False
gps_mask[l:] = False
gps_mask[0] = False
        
idxs = idxs.astype("int")
t_errors = timeGps[:l] - timeLsr[idxs]
error_squared = np.sqrt(t_errors @ t_errors)
    
gps = np.array([Lo_m[:l], La_m[:l]])
xests = xupd[idxs,:2].T
gps_means = np.mean(gps, axis=1)
xests_means= np.mean(xests, axis=1)
gps_c = gps.T - gps_means
xests_c = xests.T - xests_means
D = (gps_c).T @ (xests_c)
U,S,V = np.linalg.svd(D, full_matrices=True)
Rot = V@U.T
rotation = np.arccos(Rot[0,0]) * 180 / np.pi
rotation = round(rotation, 3)
trans =  xests_means - Rot @ gps_means
[Lo_mn, La_mn] = (gps.T @ Rot.T ).T

# GPS velocities
dt = timeGps[1:] - timeGps[:-1]
da = La_m[1:] - La_m[:-1]
do = Lo_m[1:] - Lo_m[:-1]
velocities = np.sqrt(da**2 + do**2)/dt

gps_masked = np.array([Lo_m[gps_mask], La_m[gps_mask]])
timeGps_masked = timeGps[gps_mask]
dt_gps_masked = timeGps_masked[1:] - timeGps_masked[:-1]
dgps_masked = gps_masked[:,1:] - gps_masked[:,:-1]
gps_masked_vel = np.sqrt(dgps_masked[0,:]**2 + dgps_masked[1,:]**2)/dt_gps_masked

# Estimated velocities
dt_e = timeLsr[short_idxs[1:]] - timeLsr[short_idxs[:-1]]
xy_e = xupd[short_idxs[1:],:2] - xupd[short_idxs[:-1], :2]
vel_est = np.sqrt(xy_e[:,0]**2 + xy_e[:,1]**2)/dt_e

mask = np.ones(velocities.size, dtype=bool)
remove = np.argwhere(velocities > 6).ravel()
mask[remove] = False
velocities_cleaned = velocities[mask]
mask = np.insert(mask, 0, True)
#Lo_mc = Lo_m[temp_mask]
#La_mc = La_m[temp_mask]
#timeGps_c = timeGps[mask]

# NIS
insideCI = (CInorm[:mk, 0] <= NISnorm[:mk]) * (NISnorm[:mk] <= CInorm[:mk, 1])

fig3, ax3 = plt.subplots(nrows=2, ncols=2, num=3, clear=True, figsize=(10,4))
ax3[0,0].plot(CInorm[:mk, 0], "--")
ax3[0,0].plot(CInorm[:mk, 1], "--")
ax3[0,0].plot(NISnorm[:mk], lw=0.5)
ax3[0,0].set_title(f"NIS, {insideCI.mean()*100:.2f}% inside CI, ANIS = {NISnorm[:mk].mean()}")

CI_NIS = np.array(chi2.interval(1-alpha, 2))

insideCI = (CI_NIS[0] <= GPS_NIS) * (GPS_NIS <= CI_NIS[1])
ax3[1,0].plot(np.full(GPSi1, CI_NIS[0]), '--')
ax3[1,0].plot(np.full(GPSi1, CI_NIS[1]), '--')
ax3[1,0].plot(GPS_NIS[:GPSi1], lw=0.5)
ax3[1,0].set_title(f"GNSS NIS {insideCI.mean()*100:.2f}% inside CI, ANIS = {np.round(GPS_NIS.mean(),2)}")

vel_rmse = np.sqrt(np.linalg.norm(gps_masked_vel) - np.linalg.norm(vel_est))
LO_RMSE = np.sqrt(np.sum((gps_masked[0,:] - xupd[short_idxs,0])**2))
LA_RMSE = np.sqrt(np.sum((gps_masked[1,:] - xupd[short_idxs,1])**2))
Time_RMSE = np.sqrt(np.sum((timeGps_masked - timeLsr[short_idxs])**2))

ax3[0,1].plot(gps_masked[0,:], label="GNSS lo")
ax3[0,1].plot(gps_masked[1,:], label="GNSS la")
ax3[0,1].plot(xupd[short_idxs,0], label="xupd lo")
ax3[0,1].plot(xupd[short_idxs,1], label="xupd la")
ax3[0,1].legend(prop={'size': 7})
ax3[0,1].set_title(f"Longitude RMSE {np.round(LO_RMSE,2)}, Latitude RMSE {np.round(LA_RMSE, 2)}, Time RMSE = {np.round(Time_RMSE, 2)}")

ax3[1,1].plot(gps_masked_vel, label="GNSS velocity")
ax3[1,1].plot(vel_est, label="xupd velocity")
ax3[1,1].legend(prop={'size': 7})
ax3[1,1].set_title(f"GNSS and estimated velocities, RMSE {np.round(vel_rmse, 2)}")

plt.tight_layout()

# %% slam

if do_raw_prediction:
    fig5, ax5 = plt.subplots(num=5, clear=True)
    ax5.scatter(
        Lo_m,#[timeGps < timeOdo[N - 1]],
        La_m,#[timeGps < timeOdo[N - 1]],
        c="r",
        marker=".",
        label="GPS",
    )
    ax5.scatter(
        Lo_mn,#[timeGps < timeOdo[N - 1]],
        La_mn,#[timeGps < timeOdo[N - 1]],
        c="g",
        marker=".",
        label="GPS",
    )
    ax5.plot(*odox[:N, :2].T, label="odom")
    ax5.grid()
    ax5.set_title("GPS vs odometry integration")
    ax5.legend()

# %%
fig6, ax6 = plt.subplots(num=6, clear=True)
ax6.scatter(*eta[3:].reshape(-1, 2).T, color="r", marker="x", s=10)
ax6.plot(*xupd[mk_first:mk, :2].T)
ax6.scatter(
    Lo_m[timeGps < timeOdo[N - 1]],
    La_m[timeGps < timeOdo[N - 1]],
    c="g",
    s=5,
    marker=".",
    label="GNSS",
)
ax6.scatter(
    Lo_mn,
    La_mn,
    s=5,
    c="k",
    marker=".",
    #label=r"GNSS $\theta$ = "+f"{rotation}$^\circ$, $T$ = {np.round(trans,2)}",
    label="GNSS transformed"
)
ax6.legend(prop={'size': 8})
ax6.set(
    title = f"landmarks {len(eta[3:])//2}"
    #title=f"Steps {k}, laser scans {mk-1}, landmarks {len(eta[3:])//2},\nmeasurements {z.shape[0]}, num new = {np.sum(a[mk] == -1)}"
)
plt.show()

# %%
