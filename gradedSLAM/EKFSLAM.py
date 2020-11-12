from typing import Tuple
import numpy as np
from scipy.linalg import block_diag
import scipy.linalg as la
from scipy import stats
from utils import rotmat2d, block_diag_einsum
from JCBB import JCBB
import utils

# import line_profiler
# import atexit

# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)


class EKFSLAM:
    def __init__(
        self,
        Q,
        R,
        do_asso=False,
        alphas=np.array([0.001, 0.0001]),
        sensor_offset=np.zeros(2),
        R_gps = None
    ):

        self.Q = Q
        self.R = R
        self.do_asso = do_asso
        self.alphas = alphas
        self.sensor_offset = sensor_offset
        self.R_gps = R_gps

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Add the odometry u to the robot state x.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the odometry

        Returns
        -------
        np.ndarray, shape = (3,)
            the predicted state
        """
        # TODO, eq (11.7). Should wrap heading angle between (-pi, pi), see utils.wrapToPi
        psi = x[2]
        xpred = np.array([x[0] + u[0] * np.cos(psi) - u[1] * np.sin(psi),
                          x[1] + u[0] * np.sin(psi) + u[1] * np.cos(psi),
                          utils.wrapToPi(psi + u[2])])


        assert xpred.shape == (3,), "EKFSLAM.f: wrong shape for xpred"
        return xpred

    def Fx(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Calculate the Jacobian of f with respect to x.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the odometry

        Returns
        -------
        np.ndarray
            The Jacobian of f wrt. x.
        """
        # TODO, eq (11.13)
        psi = x[2]
        Fx = np.array([[1, 0, -u[0] * np.sin(psi) - u[1] * np.cos(psi)],
                       [0, 1, u[0] * np.cos(psi) - u[1] * np.sin(psi)],
                       [0, 0, 1]])


        assert Fx.shape == (3, 3), "EKFSLAM.Fx: wrong shape"
        return Fx

    def Fu(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Calculate the Jacobian of f with respect to u.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the odometry

        Returns
        -------
        np.ndarray
            The Jacobian of f wrt. u.
        """
        # TODO, eq (11.14)
        psi = x[2]
        Fu = np.array([[np.cos(psi), -np.sin(psi), 0],
                       [np.sin(psi), np.cos(psi), 0],
                       [0, 0, 1]])

        assert Fu.shape == (3, 3), "EKFSLAM.Fu: wrong shape"
        return Fu

    def predict(
        self, eta: np.ndarray, P: np.ndarray, z_odo: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the robot state using the zOdo as odometry the corresponding state&map covariance.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2*#landmarks,)
            the robot state and map concatenated
        P : np.ndarray, shape=(3 + 2*#landmarks,)*2
            the covariance of eta
        z_odo : np.ndarray, shape=(3,)
            the measured odometry

        Returns
        -------
        Tuple[np.ndarray, np.ndarray], shapes= (3 + 2*#landmarks,), (3 + 2*#landmarks,)*2
            predicted mean and covariance of eta.
        """
        # check inout matrix
        assert np.allclose(P, P.T), "EKFSLAM.predict: not symmetric P input"
        assert np.all(
            np.linalg.eigvals(P) >= 0
        ), "EKFSLAM.predict: non-positive eigen values in P input"
        assert (
            eta.shape * 2 == P.shape
        ), "EKFSLAM.predict: input eta and P shape do not match"
        etapred = np.empty_like(eta)

        x = eta[:3]
        etapred[:3] = self.f(x, z_odo) # TODO robot state prediction
        etapred[3:] = eta[3:] # TODO landmarks: no effect

        Fx = self.Fx(x, z_odo) # TODO
        Fu = self.Fu(x, z_odo) # TODO

        # evaluate covariance prediction in place to save computation
        # only robot state changes, so only rows and colums of robot state needs changing
        # cov matrix layout:
        # [[P_xx, P_xm],
        # [P_mx, P_mm]]
        P[:3, :3] = Fx @ P[:3, :3] @ Fx.T + Fu @ self.Q @ Fu.T # TODO robot cov prediction
        P[:3, 3:] = Fx @ P[:3, 3:]  # TODO robot-map covariance prediction
        P[3:, :3] = P[:3, 3:].T     # TODO map-robot covariance: transpose of the above


        assert np.allclose(P, P.T), "EKFSLAM.predict: not symmetric P"
        assert np.all(
            np.linalg.eigvals(P) > 0
        ), "EKFSLAM.predict: non-positive eigen values"
        assert (
            etapred.shape * 2 == P.shape
        ), "EKFSLAM.predict: calculated shapes does not match"
        return etapred, P

    def h(self, eta: np.ndarray) -> np.ndarray:
        """Predict all the landmark positions in sensor frame.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2 * #landmarks,)
            The robot state and landmarks stacked.

        Returns
        -------
        np.ndarray, shape=(2 * #landmarks,)
            The landmarks in the sensor frame.
        """
        # extract states and map
        x = eta[0:3]
        ## reshape map (2, #landmarks), m[:, j] is the jth landmark
        m = eta[3:].reshape((-1, 2)).T

        Rot = rotmat2d(-x[2])

        # None as index ads an axis with size 1 at that position.
        # Numpy broadcasts size 1 dimensions to any size when needed
        
        # TODO, relative position of landmark to sensor on robot in world frame
        delta_m = (m.T - x[:2])

        # TODO, predicted measurements in cartesian coordinates, beware sensor offset for VP
        zpredcart =  delta_m - Rot.T @ self.sensor_offset 
        z_body = Rot @ zpredcart.T

        zpred_r = la.norm(zpredcart, axis=1)  # TODO, ranges
        zpred_theta = np.arctan2(z_body[1], z_body[0])# TODO, bearings
        zpred = np.stack((zpred_r, zpred_theta))    # TODO, the two arrays above stacked on top of each other vertically like 
                                                    # [ranges; 
                                                    #  bearings]
                                                    # into shape (2, #lmrk)

        zpred = zpred.T.ravel() # stack measurements along one dimension, [range1 bearing1 range2 bearing2 ...]

        assert (
            zpred.ndim == 1 and zpred.shape[0] == eta.shape[0] - 3
        ), "SLAM.h: Wrong shape on zpred"
        return zpred

    def H(self, eta: np.ndarray) -> np.ndarray:
        """Calculate the jacobian of h.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2 * #landmarks,)
            The robot state and landmarks stacked.

        Returns
        -------
        np.ndarray, shape=(2 * #landmarks, 3 + 2 * #landmarks)
            the jacobian of h wrt. eta.
        """
        # extract states and map
        x = eta[0:3]
        ## reshape map (2, #landmarks), m[j] is the jth landmark
        m = eta[3:].reshape((-1, 2)).T

        numM = m.shape[1]

        Rot = rotmat2d(x[2])
        Rpihalf = rotmat2d(np.pi / 2)

        delta_m = m - x[:2].reshape((2,1)) # TODO, relative position of landmark to robot in world frame. m - rho that appears in (11.15) and (11.16)

        zc =  (delta_m.T - Rot @ self.sensor_offset).T # TODO, (2, #measurements), each measured position in cartesian coordinates like
        # [x coordinates;
        #  y coordinates]


        # NOTE: Calculate zpred here for some speed increase probably
        zpred = self.h(eta)                                          # TODO (2, #measurements), predicted measurements, like
        zpred = np.reshape(zpred, (2, zpred.shape[0] //2), "F") # [ranges;
                                                                #  bearings]
        #zr = zpred[0, :]    # TODO, ranges

        # Allocate H and set submatrices as memory views into H
        # You may or may not want to do this like this
        H = np.zeros((2 * numM, 3 + 2 * numM)) # TODO, see eq (11.15), (11.16), (11.17)

        z = np.zeros((numM,1))
        ones = np.ones((numM,1))
        Hx_tops = -np.concatenate(((1/la.norm(zc, axis=0) * delta_m).T, z), axis=1) # Correct
        Hx_bottoms = np.concatenate(((1/la.norm(zc, axis=0)**2 * delta_m).T @ Rpihalf, -ones), axis=1) # Correct
        Hx = np.concatenate((Hx_tops, Hx_bottoms), axis=1)
        Hx = Hx.reshape(-1,3)

        Hm = -Hx[:,:2]
        Hm = Hm.reshape(numM,-1,2)
        
        H[:,:3] = Hx
        H[:,3:] = la.block_diag(*Hm)

        
        # TODO: You can set some assertions here to make sure that some of the structure in H is correct
        
        return H

    def add_landmarks(
        self, eta: np.ndarray, P: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate new landmarks, their covariances and add them to the state.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2*#landmarks,)
            the robot state and map concatenated
        P : np.ndarray, shape=(3 + 2*#landmarks,)*2
            the covariance of eta
        z : np.ndarray, shape(2 * #newlandmarks,)
            A set of measurements to create landmarks for

        Returns
        -------
        Tuple[np.ndarray, np.ndarray], shapes=(3 + 2*(#landmarks + #newlandmarks,), (3 + 2*(#landmarks + #newlandmarks,)*2
            eta with new landmarks appended, and its covariance
        """
        n = P.shape[0]
        assert z.ndim == 1, "SLAM.add_landmarks: z must be a 1d array"

        numLmk = z.shape[0] // 2

        lmnew = np.empty_like(z)

        Gx = np.empty((numLmk * 2, 3))
        Rall = np.zeros((numLmk * 2, numLmk * 2))

        sensor_offset_world = rotmat2d(eta[2]) @ self.sensor_offset # For transforming landmark position into world frame
        sensor_offset_world_der = rotmat2d(eta[2] + np.pi / 2) @ self.sensor_offset # Used in Gx

        rot = rotmat2d(z[1::2] + eta[2])
        #new_rot = rot.swapaxes(0,2).swapaxes(1,2).ravel().reshape((18,2))
        
        vecZ = np.array([-np.sin(z[1::2] + eta[2]), np.cos(z[1::2] + eta[2])]).T
        Gx[:,:2] = np.tile(np.eye(2), (numLmk, 1))
        Gx[:,2] = (z[0::2].reshape(-1,1) * vecZ + sensor_offset_world_der).ravel()

        lmnew = ((z[0::2] * rot[:,0]).T + eta[0:2] + sensor_offset_world).ravel()

        right = np.insert(z[0::2], np.arange(0,numLmk,1), 0)
        left = np.tile([1,0],numLmk)
        np.concatenate((left, right)).reshape(-1,2, order="F")

        for j in range(numLmk):
            ind = 2 * j
            inds = slice(ind, ind + 2)
            zj = z[inds]

            Gz = rot[:,:,j] @ np.diag([1, zj[0]]) # TODO

            Rall[inds, inds] = Gz @ self.R @ Gz.T # TODO, Gz * R * Gz^T, transform measurement covariance from polar to cartesian coordinates

        assert len(lmnew) % 2 == 0, "SLAM.add_landmark: lmnew not even length"
        etaadded = np.hstack((eta, lmnew))# TODO, append new landmarks to state vector
        
        n_R = Rall.shape[0]
        Padded = np.zeros((n+n_R,n+n_R))
        
        Padded[:n,:n] = P
        Padded[n:,n:] = Gx@P[:3,:3]@Gx.T + Rall
        Padded[n:,:n] = Gx @ P[:3,:]
        Padded[:n,n:] = Padded[n:,:n].T
        
        assert (
            etaadded.shape * 2 == Padded.shape
        ), "EKFSLAM.add_landmarks: calculated eta and P has wrong shape"
        assert np.allclose(
            Padded, Padded.T
        ), "EKFSLAM.add_landmarks: Padded not symmetric"
        assert np.all(
            np.linalg.eigvals(Padded) >= 0
        ), "EKFSLAM.add_landmarks: Padded not PSD"

        return etaadded, Padded

    def associate(
        self, z: np.ndarray, zpred: np.ndarray, H: np.ndarray, S: np.ndarray,
    ):  # -> Tuple[*((np.ndarray,) * 5)]:
        """Associate landmarks and measurements, and extract correct matrices for these.

        Parameters
        ----------
        z : np.ndarray,
            The measurements all in one vector
        zpred : np.ndarray
            Predicted measurements in one vector
        H : np.ndarray
            The measurement Jacobian matrix related to zpred
        S : np.ndarray
            The innovation covariance related to zpred

        Returns
        -------
        Tuple[*((np.ndarray,) * 5)]
            The extracted measurements, the corresponding zpred, H, S and the associations.

        Note
        ----
        See the associations are calculated  using JCBB. See this function for documentation
        of the returned association and the association procedure.
        """
        if self.do_asso:
            # Associate
            a = JCBB(z, zpred, S, self.alphas[0], self.alphas[1])

            # Extract associated measurements
            zinds = np.empty_like(z, dtype=bool)
            zinds[::2] = a > -1  # -1 means no association
            zinds[1::2] = zinds[::2]
            zass = z[zinds]

            # extract and rearange predicted measurements and cov
            zbarinds = np.empty_like(zass, dtype=int)
            zbarinds[::2] = 2 * a[a > -1]
            zbarinds[1::2] = 2 * a[a > -1] + 1

            zpredass = zpred[zbarinds]
            Sass = S[zbarinds][:, zbarinds]
            Hass = H[zbarinds]

            assert zpredass.shape == zass.shape
            assert Sass.shape == zpredass.shape * 2
            assert Hass.shape[0] == zpredass.shape[0]

            return zass, zpredass, Hass, Sass, a
        else:
            # should one do something her
            pass

    def update(
        self, eta: np.ndarray, P: np.ndarray, z: np.ndarray, gps = None
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Update eta and P with z, associating landmarks and adding new ones.

        Parameters
        ----------
        eta : np.ndarray
            [description]
        P : np.ndarray
            [description]
        z : np.ndarray, shape=(#detections, 2)
            [description]

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float, np.ndarray]
            [description]
        """
        numLmk = (eta.size - 3) // 2
        assert (len(eta) - 3) % 2 == 0, "EKFSLAM.update: landmark lenght not even"

        if numLmk > 0:
            # Prediction and innovation covariance
            zpred = self.h(eta) # TODO
            H = self.H(eta)     # TODO

            # Here you can use simply np.kron (a bit slow) to form the big (very big in VP after a while) R,
            # or be smart with indexing and broadcasting (3d indexing into 2d mat) realizing you are adding the same R on all diagonals
            S = H @ P @ H.T + block_diag_einsum(self.R, numLmk)
            
            assert (
                S.shape == zpred.shape * 2
            ), "EKFSLAM.update: wrong shape on either S or zpred"
            z = z.ravel()  # 2D -> flat

            # Perform data association
            za, zpred, Ha, Sa, a = self.associate(z, zpred, H, S)

            # No association could be made, so skip update
            if za.shape[0] == 0:
                etaupd = eta # TODO
                Pupd = P # TODO
                NIS = 1 # TODO: beware this one when analysing consistency.

            else:
                # Create the associated innovation
                v = za.ravel() - zpred  # za: 2D -> flat
                v[1::2] = utils.wrapToPi(v[1::2])

                # Kalman mean update
                #S_cho_factors = la.cho_factor(Sa) # Optional, used in places for S^-1, see scipy.linalg.cho_factor and scipy.linalg.cho_solve
                W = P @ Ha.T @ la.inv(Sa) # TODO, Kalman gain, can use S_cho_factors
                #W = la.cho_solve(S_cho_factors, Ha @ P).T
                etaupd = eta + W @ v # TODO, Kalman update

                # Kalman cov update: use Joseph form for stability
                jo = -W @ Ha
                jo[np.diag_indices(jo.shape[0])] += 1  # same as adding Identity mat
                Pupd = jo @ P# @ jo.T + W @ np.kron(np.eye(za.size//2), self.R)@W.T
                # TODO, Kalman update. This is the main workload on VP after speedups

                # calculate NIS, can use S_cho_factors
                NIS = (v.T @ la.inv(Sa) @ v)# - CI[0]) / (CI[1] - CI[0]) # TODO
                #NIS = v.T @ la.cho_solve(S_cho_factors, v)


                # When tested, remove for speed
                assert np.allclose(Pupd, Pupd.T), "EKFSLAM.update: Pupd not symmetric"
                assert np.all(
                    np.linalg.eigvals(Pupd) > 0
                ), "EKFSLAM.update: Pupd not positive definite"

        else:  # All measurements are new landmarks,
            a = np.full(z.shape[0], -1)
            z = z.flatten()
            NIS = 0 # TODO: beware this one, you can change the value to for instance 1
            etaupd = eta
            Pupd = P

        # Create new landmarks if any is available
        if self.do_asso:
            is_new_lmk = a == -1
            if np.any(is_new_lmk):
                z_new_inds = np.empty_like(z, dtype=bool)
                z_new_inds[::2] = is_new_lmk
                z_new_inds[1::2] = is_new_lmk
                z_new = z[z_new_inds]
                etaupd, Pupd = self.add_landmarks(etaupd, Pupd, z_new)# TODO, add new landmarks.

        assert np.allclose(Pupd, Pupd.T), "EKFSLAM.update: Pupd must be symmetric"
        assert np.all(np.linalg.eigvals(Pupd) >= 0), "EKFSLAM.update: Pupd must be PSD"

        return etaupd, Pupd, NIS, a

    @classmethod
    def NEESes(cls, x: np.ndarray, P: np.ndarray, x_gt: np.ndarray,) -> np.ndarray:
        """Calculates the total NEES and the NEES for the substates
        Args:
            x (np.ndarray): The estimate
            P (np.ndarray): The state covariance
            x_gt (np.ndarray): The ground truth
        Raises:
            AssertionError: If any input is of the wrong shape, and if debug mode is on, certain numeric properties
        Returns:
            np.ndarray: NEES for [all, position, heading], shape (3,)
        """

        assert x.shape == (3,), f"EKFSLAM.NEES: x shape incorrect {x.shape}"
        assert P.shape == (3, 3), f"EKFSLAM.NEES: P shape incorrect {P.shape}"
        assert x_gt.shape == (3,), f"EKFSLAM.NEES: x_gt shape incorrect {x_gt.shape}"

        d_x = x - x_gt
        d_x[2] = utils.wrapToPi(d_x[2])
        assert (
            -np.pi <= d_x[2] <= np.pi
        ), "EKFSLAM.NEES: error heading must be between (-pi, pi)"

        d_p = d_x[0:2]
        P_p = P[0:2, 0:2]
        assert d_p.shape == (2,), "EKFSLAM.NEES: d_p must be 2 long"
        d_heading = d_x[2]  # Note: scalar
        assert np.ndim(d_heading) == 0, "EKFSLAM.NEES: d_heading must be scalar"
        P_heading = P[2, 2]  # Note: scalar
        assert np.ndim(P_heading) == 0, "EKFSLAM.NEES: P_heading must be scalar"

        # NB: Needs to handle both vectors and scalars! Additionally, must handle division by zero

        try:
            NEES_all = d_x @ (np.linalg.solve(P, d_x))
            NEES_pos = d_p @ (np.linalg.solve(P_p, d_p))
            NEES_heading = d_heading ** 2 / P_heading
        except (ZeroDivisionError, np.linalg.LinAlgError):
            NEES_all = 1.0
            NEES_pos = 1.0
            NEES_heading = 1.0 # TODO: beware

        NEESes = np.array([NEES_all, NEES_pos, NEES_heading])
        NEESes[np.isnan(NEESes)] = 1.0  # We may divide by zero, # TODO: beware

        assert np.all(NEESes >= 0), "ESKF.NEES: one or more negative NEESes"
        return NEESes
    
    def GNSS_NIS(self, x: np.ndarray, P: np.ndarray, z_gnss):
        H = np.eye(2)
        S = H @ P @ H.T + self.R_gps
        v = z_gnss - x
        NIS_GPS = v.T @ la.inv(S) @ v
        return NIS_GPS
        