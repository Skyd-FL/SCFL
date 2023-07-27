import numpy as np
from utils.setting_setup import *
import scipy


class env_agent_utils():
    def __init__(self):
        pass

    def _round(self, input, unit):
        return np.round(input / unit) * unit

    def _wrapState(self):
        self.ChannelGain = self._ChannelGain_Calculate(self.sigma_data)
        state = np.concatenate((np.array(self.ChannelGain).reshape(1, -1), np.array(self.U_location).reshape(1, -1),
                                np.array(self.User_trajectory).reshape(1, -1)), axis=1)
        return state

    def _decomposeState(self, state):
        H = state[0: self.N_User]
        U_location = state[self.N_User: 2 * self.N_User + 2]
        User_trajectory = state[self.N_User + 2: 2 * self.N_User + 4]
        return [
            np.array(H), np.array(U_location), np.array(User_trajectory)
        ]

    def _wrapAction(self):
        action = np.concatenate((np.array([[self.beta]]).reshape(1, self.N_User),
                                 np.array([[self.f_u]]).reshape(1, self.N_User),
                                 np.array([[self.p_u]]).reshape(1, self.N_User),
                                 np.array([[self.butt]]).reshape(1, 1),
                                 np.array([[self.tau]]).reshape(1, 1),
                                 ), axis=1)
        return action

    def _decomposeAction(self, action):
        # beta [N_User] : resource allocation set
        # f_u  : maximum local computation capacity of user u
        # p_u  : maximum power of user u
        # butt : local accuracy of users
        # tau  : sampling delay
        mini_eps = 10e-28
        beta = action[0][0: self.N_User].astype(float)
        beta = scipy.special.softmax(beta, axis=None)

        f_u = action[0][self.N_User: 2 * self.N_User].astype(float) * self.f_u_max
        p_u = (action[0][2 * self.N_User: 3 * self.N_User].astype(float)) * (self.p_u_max-mini_eps) + mini_eps
        butt = (action[0][3 * self.N_User: 3 * self.N_User+1].astype(float))
        # tau = (action[0][3 * self.N_User+1: 3 * self.N_User+2].astype(float)) * 100
        tau = (action[0][3 * self.N_User+1: 3 * self.N_User+2].astype(float)) * (self.skip_max-1) + 1
        return [
            np.array(beta).reshape((1, self.N_User)).squeeze(),
            np.array(f_u).reshape((1, self.N_User)).squeeze(),
            np.array(p_u).reshape((1, self.N_User)).squeeze(),
            np.array([[butt]]).reshape(1, 1),
            np.round(np.array([[tau]]).reshape(1, 1)),
        ]
