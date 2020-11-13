# %% Imports
from typing import List, Optional

from scipy.io import loadmat
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.gridspec as gridspec
from scipy.stats import chi2
import utils

try:
    from tqdm import tqdm
except ImportError as e:
    print(e)
    print("install tqdm to have progress bar")

    # def tqdm as dummy as it is not available
    def tqdm(*args, **kwargs):
        return args[0]

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



from EKFSLAM import EKFSLAM
from plotting import ellipse

# %% Load data
simSLAM_ws = loadmat("simulatedSLAM")

## NB: this is a MATLAB cell, so needs to "double index" to get out the measurements of a time step k:
#
# ex:
#
# z_k = z[k][0] # z_k is a (2, m_k) matrix with columns equal to the measurements of time step k
#
##
z = [zk.T for zk in simSLAM_ws["z"].ravel()]

landmarks = simSLAM_ws["landmarks"].T
odometry = simSLAM_ws["odometry"].T
poseGT = simSLAM_ws["poseGT"].T

K = len(z)
M = len(landmarks)

# %% Initilize
Q = np.diag([0.005, 0.005, 0.004/180*np.pi])    # TODO
R = np.diag([0.0033, 0.02/180*np.pi])          # TODO

doAsso = True

JCBBalphas = np.array([5e-10, 5e-10])     # TODO
# first is for joint compatibility, second is individual
# these can have a large effect on runtime either through the number of landmarks created
# or by the size of the association search space.

slam = EKFSLAM(Q, R, do_asso=doAsso, alphas=JCBBalphas)

# allocate
eta_pred: List[Optional[np.ndarray]] = [None] * K
P_pred: List[Optional[np.ndarray]] = [None] * K
eta_hat: List[Optional[np.ndarray]] = [None] * K
P_hat: List[Optional[np.ndarray]] = [None] * K
a: List[Optional[np.ndarray]] = [None] * K
NIS = np.zeros(K)
NISnorm = np.zeros(K)
CI = np.zeros((K, 2))
CInorm = np.zeros((K, 2))
NEESes = np.zeros((K, 3))

# For consistency testing
alpha = 0.05

# init
eta_pred[0] = poseGT[0]  # we start at the correct position for reference
P_pred[0] = np.zeros((3, 3))  # we also say that we are 100% sure about that

# %% Set up plotting
# plotting

doAssoPlot = False
playMovie = False
if doAssoPlot:
    figAsso, axAsso = plt.subplots(num=1, clear=True)

# %% Run simulation
N = K

print("starting sim (" + str(N) + " iterations)")

for k, z_k in tqdm(enumerate(z[:N])):

    eta_hat[k], P_hat[k], NIS[k], a[k] = slam.update(eta_pred[k], P_pred[k], z[k]) # TODO update

    if k < K - 1:
        eta_pred[k + 1], P_pred[k + 1] = slam.predict(eta_hat[k], P_hat[k].copy(), odometry[k])# TODO predict

    assert (
        eta_hat[k].shape[0] == P_hat[k].shape[0]
    ), "dimensions of mean and covariance do not match"

    num_asso = np.count_nonzero(a[k] > -1)

    CI[k] = chi2.interval(1-alpha, 2 * num_asso)

    if num_asso > 0:
        NISnorm[k] = NIS[k] / (2 * num_asso)
        CInorm[k] = CI[k] / (2 * num_asso)
    else:
        NISnorm[k] = 1
        CInorm[k].fill(1)

    NEESes[k] = slam.NEESes(eta_hat[k][:3], P_hat[k][:3,:3], poseGT[k]) # TODO, use provided function slam.NEESes

    if doAssoPlot and k > 0:
        axAsso.clear()
        axAsso.grid()
        zpred = slam.h(eta_pred[k]).reshape(-1, 2)
        axAsso.scatter(z_k[:, 0], z_k[:, 1], label="z")
        axAsso.scatter(zpred[:, 0], zpred[:, 1], label="zpred")
        xcoords = np.block([[z_k[a[k] > -1, 0]], [zpred[a[k][a[k] > -1], 0]]]).T
        ycoords = np.block([[z_k[a[k] > -1, 1]], [zpred[a[k][a[k] > -1], 1]]]).T
        for x, y in zip(xcoords, ycoords):
            axAsso.plot(x, y, lw=3, c="r")
        axAsso.legend()
        axAsso.set_title(f"k = {k}, {np.count_nonzero(a[k] > -1)} associations")
        plt.draw()
        plt.pause(0.001)


print("sim complete")

pose_est = np.array([x[:3] for x in eta_hat[:N]])
lmk_est = [eta_hat_k[3:].reshape(-1, 2) for eta_hat_k in eta_hat]
lmk_est_final = lmk_est[N - 1]

np.set_printoptions(precision=4, linewidth=100)

# %% Plotting of results
import pylab as plb
mins = np.amin(landmarks, axis=0)
maxs = np.amax(landmarks, axis=0)

ranges = maxs - mins
offsets = ranges * 0.2

mins -= offsets
maxs += offsets

fig1 = plt.figure(figsize=(10,4))
G = gridspec.GridSpec(2, 2)

# landmarks

ax1 = plb.subplot(G[:, 0])
ax1.scatter(*landmarks.T, c="r", marker="^")
ax1.scatter(*lmk_est_final.T, c="b", marker=".")
# Draw covariance ellipsis of measurements
for l, lmk_l in enumerate(lmk_est_final):
    idxs = slice(3 + 2 * l, 3 + 2 * l + 2)
    rI = P_hat[N - 1][idxs, idxs]
    el = ellipse(lmk_l, rI, 5, 200)
    ax1.plot(*el.T, "b", linewidth=0.5)

ax1.plot(*poseGT.T[:2], c="r", label="gt", linewidth=0.5)
ax1.plot(*pose_est.T[:2], c="g", label="est", linewidth=0.5)
ax1.plot(*ellipse(pose_est[-1, :2], P_hat[N - 1][:2, :2], 5, 200).T, c="g")
ax1.set(title="A: results", xlim=(mins[0], maxs[0]), ylim=(mins[1], maxs[1]))
ax1.axis("equal")
ax1.grid()

# %% RMSE

tags = ['all', 'pos', 'heading']
pos_err = np.linalg.norm(pose_est[:N,:2] - poseGT[:N,:2], axis=1)
heading_err = np.abs(utils.wrapToPi(pose_est[:N,2] - poseGT[:N,2]))
errs = np.vstack((pos_err, heading_err))

ylabels = ['m', 'deg']
scalings = np.array([1, 180/np.pi])

ax2 = plb.subplot(G[0,1])
ax2.plot(errs[0]*scalings[0])
ax2.set_title(f"B: {tags[1]}: RMSE {np.sqrt((errs[0]**2).mean())*scalings[0]} {ylabels[0]}")
ax2.set_ylabel(f"[{ylabels[0]}]")
ax2.grid()

ax3 = plb.subplot(G[1, 1])
ax3.plot(errs[1]*scalings[1])
ax3.set_title(f"C: {tags[2]}: RMSE {np.sqrt((errs[1]**2).mean())*scalings[1]} {ylabels[1]}")
ax3.set_ylabel(f"[{ylabels[1]}]")
ax3.grid()

fig1.tight_layout()

#plb.show()
#plt.savefig("report/figures/sim_results.eps", format="eps")

# %% Consistency

fig2, ax2 = plt.subplots(nrows=2, ncols=2, num=2, figsize=(10,3), clear=True)

# NIS

insideCI = (CInorm[:N,0] <= NISnorm[:N]) * (NISnorm[:N] <= CInorm[:N,1])
ax2[0,0].plot(CInorm[:N,0], '--')
ax2[0,0].plot(CInorm[:N,1], '--')
ax2[0,0].plot(NISnorm[:N], lw=0.5)
ax2[0,0].set_title(f'A: NIS {round(insideCI.mean()*100, 4)}% inside CI, ANIS = {NISnorm[:N].mean()}')

# NEES

dfs = [3, 2, 1]

CI_NEES = chi2.interval(1-alpha, dfs[0])
ax2[0,1].plot(np.full(N, CI_NEES[0]), '--')
ax2[0,1].plot(np.full(N, CI_NEES[1]), '--')
ax2[0,1].plot(NEESes[:N,0], lw=0.5)
insideCI = (CI_NEES[0] <= NEESes[:N,0]) * (NEESes[:N,0] <= CI_NEES[1])
ax2[0,1].set_title(f'B: NEES {tags[0]}: {insideCI.mean()*100}% inside CI')
CI_ANEES = np.array(chi2.interval(1-alpha, dfs[0]*N)) / N
print(f"CI ANEES {tags[0]}: {CI_ANEES}")
print(f"ANEES {tags[0]}: {NEESes[:N,0].mean()}")

CI_NEES = chi2.interval(1-alpha, dfs[1])
ax2[1,0].plot(np.full(N, CI_NEES[0]), '--')
ax2[1,0].plot(np.full(N, CI_NEES[1]), '--')
ax2[1,0].plot(NEESes[:N,1], lw=0.5)
insideCI = (CI_NEES[0] <= NEESes[:N,1]) * (NEESes[:N,1] <= CI_NEES[1])
ax2[1,0].set_title(f'C: NEES {tags[1]}: {insideCI.mean()*100}% inside CI')
CI_ANEES = np.array(chi2.interval(1-alpha, dfs[1]*N)) / N
print(f"CI ANEES {tags[1]}: {CI_ANEES}")
print(f"ANEES {tags[1]}: {NEESes[:N,1].mean()}")

CI_NEES = chi2.interval(1-alpha, dfs[2])
ax2[1,1].plot(np.full(N, CI_NEES[0]), '--')
ax2[1,1].plot(np.full(N, CI_NEES[1]), '--')
ax2[1,1].plot(NEESes[:N,2], lw=0.5)
insideCI = (CI_NEES[0] <= NEESes[:N,2]) * (NEESes[:N,2] <= CI_NEES[1])
ax2[1,1].set_title(f'D: NEES {tags[2]}: {insideCI.mean()*100}% inside CI')
CI_ANEES = np.array(chi2.interval(1-alpha, dfs[2]*N)) / N
print(f"CI ANEES {tags[2]}: {CI_ANEES}")
print(f"ANEES {tags[2]}: {NEESes[:N,2].mean()}")

fig2.tight_layout()
plt.savefig("report/figures/sim_NIS_NEES.eps", format="eps")

# %% Movie time

if playMovie:
    try:
        print("recording movie...")

        from celluloid import Camera

        pauseTime = 0.05
        fig_movie, ax_movie = plt.subplots(num=6, clear=True)

        camera = Camera(fig_movie)

        ax_movie.grid()
        ax_movie.set(xlim=(mins[0], maxs[0]), ylim=(mins[1], maxs[1]))
        camera.snap()

        for k in tqdm(range(N)):
            ax_movie.scatter(*landmarks.T, c="r", marker="^")
            ax_movie.plot(*poseGT[:k, :2].T, "r-")
            ax_movie.plot(*pose_est[:k, :2].T, "g-")
            ax_movie.scatter(*lmk_est[k].T, c="b", marker=".")

            if k > 0:
                el = ellipse(pose_est[k, :2], P_hat[k][:2, :2], 5, 200)
                ax_movie.plot(*el.T, "g")

            numLmk = lmk_est[k].shape[0]
            for l, lmk_l in enumerate(lmk_est[k]):
                idxs = slice(3 + 2 * l, 3 + 2 * l + 2)
                rI = P_hat[k][idxs, idxs]
                el = ellipse(lmk_l, rI, 5, 200)
                ax_movie.plot(*el.T, "b")

            camera.snap()
        animation = camera.animate(interval=100, blit=True, repeat=False)
        print("playing movie")

    except ImportError:
        print(
            "Install celluloid module, \n\n$ pip install celluloid\n\nto get fancy animation of EKFSLAM."
        )

plt.show()
# %%
