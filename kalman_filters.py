from kalman_rpe import *
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import unscented_transform
from numpy import dot
from copy import deepcopy


class MyUKF(UnscentedKalmanFilter):
    def __init__(self, num_params, num_circs):
        self.dt=1

        # # calculate number of circuits
        # num_circs = 0
        # for d in edesign.keys():
        #     num_circs += len(edesign[d])
        # self.num_circs = num_circs
        # self.R = (0.25/num_shots)*np.eye(num_circs)

        self.points = MerweScaledSigmaPoints(num_params, alpha=1e-3, beta=2, kappa=0)
        super().__init__(num_params, num_circs, dt=1, hx=self.hx, fx=self.fx, points=self.points)

    def fx(self, x, dt, u=None):
        if u is None:
            u = np.zeros(len(x))
        return x + u

    def hx(self, x, d, data_format="cartesian"):
        pvec = make_probability_vector(x, d)
        return format_observation(pvec, data_format)

    def check_consistency_at_d(self, d):
        """Check that the experiment design at depth d is consistent for the sigma points."""
        pass

    def update(self, observation, d, num_shots, UT=None, hx=None, data_format="cartesian"):
        """
        Update the UKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter.

        update one generation at a time

        Parameters
        ----------

        observation: emperical distribution

        d: int, depth of the circuit

        num_shots: int, number of shots per circuit

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.

        hx : callable h(x, **hx_args), optional
            Measurement function. If not provided, the default
            function passed in during construction will be used.

        **hx_args : keyword argument
            arguments to be passed into h(x) after x -> h(x, **hx_args)
        """


        if hx is None:
            hx = self.hx

        if UT is None:
            UT = unscented_transform

        # elif isscalar(R):
        #     R = eye(self._dim_z) * R
        
        z = format_observation(observation, data_format)
        #alpha_vec = (counts + np.ones(len(counts)))/(num_shots + len(counts)) Uncomment for Dirichlet covar
        #R = np.diag(alpha_vec*(1-alpha_vec)/(num_shots + len(counts) +1))
        R = (0.25/num_shots)*np.eye(len(observation))

        # pass prior sigmas through h(x) to get measurement sigmas
        # the shape of sigmas_h will vary if the shape of z varies, so
        # recreate each time
        sigmas_h = []
        for s in self.sigmas_f:
            sigmas_h.append(hx(s, d, data_format=data_format))

        self.sigmas_h = np.atleast_2d(sigmas_h)

        # mean and covariance of prediction passed through unscented transform
        zp, self.S = UT(self.sigmas_h, self.Wm, self.Wc, R, self.z_mean, self.residual_z)
        self.SI = self.inv(self.S)

        # compute cross variance of the state and the measurements
        Pxz = self.cross_variance(self.x, zp, self.sigmas_f, self.sigmas_h)
        
        # DEBUG
        #print(self.sigmas_f, self.sigmas_h)
        #print(np.linalg.norm(Pxz), np.linalg.norm(self.S), np.linalg.norm(self.SI))

        self.K = dot(Pxz, self.SI)        # Kalman gain
        self.y = self.residual_z(z, zp)   # residual

        x_new = self.x + dot(self.K, self.y)
        # phase unwrapping on x[2]
        # x_new[2] = np.unwrap([x_new[2]])
        # if np.linalg.norm(x_new - self.x) < tol:
        #     break
        # else:
        #     self.x = x_new
        #     self.compute_process_sigmas(self.dt, self.fx)
        #     #if i == max_iter - 1:
        #     #    print(f"WARNING: Max iterations reached without convergence with tolerance {tol}.")
        self.x = x_new
        self.P = self.P - self.K@self.S@self.K.T

        # save measurement and posterior state
        self.x_post = self.x.copy()
        self.z = deepcopy(z)
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    def update_from_dataset(self, dataset, depths, num_shots, data_format="cartesian", save_history=False):
        if save_history:
            self.xhistory = []
            self.Phistory = []
        for d in depths:
            x_cos_circ, x_sin_circ, z_cos_circ, z_sin_circ, interleaved_cos_circ, interleaved_sin_circ = make_ordered_circuit_list(d)
            if type(dataset) == dict:
                x_cos_obs = dataset[x_cos_circ]['1']/sum(dataset[x_cos_circ].values())
                x_sin_obs = dataset[x_sin_circ]['1']/sum(dataset[x_sin_circ].values())
                z_cos_obs = dataset[z_cos_circ]['1']/sum(dataset[z_cos_circ].values())
                z_sin_obs = dataset[z_sin_circ]['1']/sum(dataset[z_sin_circ].values())
                interleaved_cos_obs = dataset[interleaved_cos_circ]['1']/sum(dataset[interleaved_cos_circ].values())
                interleaved_sin_obs = dataset[interleaved_sin_circ]['1']/sum(dataset[interleaved_sin_circ].values())
            else:
                # assume a pygsti dataset
                x_cos_obs = dataset[x_cos_circ]['1']/sum(dataset[x_cos_circ].counts.values())
                x_sin_obs = dataset[x_sin_circ]['1']/sum(dataset[x_sin_circ].counts.values())
                z_cos_obs = dataset[z_cos_circ]['1']/sum(dataset[z_cos_circ].counts.values())
                z_sin_obs = dataset[z_sin_circ]['1']/sum(dataset[z_sin_circ].counts.values())
                interleaved_cos_obs = dataset[interleaved_cos_circ]['1']/sum(dataset[interleaved_cos_circ].counts.values())
                interleaved_sin_obs = dataset[interleaved_sin_circ]['1']/sum(dataset[interleaved_sin_circ].counts.values())       
            observation = np.array([x_cos_obs, x_sin_obs, z_cos_obs, z_sin_obs, interleaved_cos_obs, interleaved_sin_obs])
            self.predict()
            self.update(observation, d, num_shots, data_format=data_format)
            if save_history:
                self.xhistory.append(self.x.copy())
                self.Phistory.append(self.P.copy())
        if save_history:
            return self.xhistory, self.Phistory
        
        
        

def make_jacobian(xstate, d, gate_depolarization=0., spam_depolarization=0., epsilon=1e-6):
    jac = np.zeros((6, 3))
    for i in range(len(xstate)):
        xstate_plus = xstate.copy()
        xstate_plus[i] += epsilon
        p_plus = make_probability_vector(xstate_plus, d)
        xstate_minus = xstate.copy()
        xstate_minus[i] -= epsilon
        p_minus = make_probability_vector(xstate_minus, d)
        p_diff = p_plus - p_minus
        if (p_diff == 0).all():
            p_grad = np.zeros(6)
        else:
            p_grad = p_diff/(2*epsilon)
        jac[:, i] = p_grad
    return jac

class KalmanFilter:
    """Extended Kalman filter of RPE data"""
    def __init__(self, x0, P0, save_history=False):
        self.x = x0.copy()
        self.P = P0.copy()
        self.save_history = save_history
        if save_history:
            self.xhistory = [self.x.copy()]
            self.Phistory = [self.P.copy()]

    def update(self, observation, d, num_shots, data_format="cartesian", estimated_gate_depol=0., estimated_spam_depol=0.):
        pvec = make_probability_vector(self.x, d)
        R = (0.25/num_shots)*np.eye(len(observation))
        H = make_jacobian(self.x, d, estimated_gate_depol, estimated_spam_depol)
        K = self.P@H.T@np.linalg.inv(H@self.P@H.T + R)
        self.x = self.x + K@(observation - pvec)
        self.P = self.P - K@H@self.P
        if self.save_history:
            self.xhistory.append(self.x.copy())
            self.Phistory.append(self.P.copy())

    def iterated_update(self, observation, d, num_shots, data_format="cartesian", estimated_gate_depol=0., estimated_spam_depol=0., tol=1e-8, max_iter=100):
        R = (0.25/num_shots)*np.eye(len(observation))
        x = self.x.copy()
        for i in range(0, max_iter):
            pvec = make_probability_vector(x, d)
            H = make_jacobian(x, d, estimated_gate_depol, estimated_spam_depol)
            K = self.P@H.T@np.linalg.inv(H@self.P@H.T + R)
            x_post = x + K@(observation - pvec)
            if np.linalg.norm(K@(observation - pvec)) < tol:
                self.x = x_post
                self.P = self.P - K@H@self.P
                if self.save_history:
                    self.xhistory.append(self.x.copy())
                    self.Phistory.append(self.P.copy())
                break
            elif i == max_iter - 1:
                print(f"WARNING: Max iterations reached without convergence with tolerance {tol}.")
                self.x = x_post
                self.P = self.P - K@H@self.P
                if self.save_history:
                    self.xhistory.append(self.x.copy())
                    self.Phistory.append(self.P.copy())
                break
            else:
                x = x_post

            
        

        