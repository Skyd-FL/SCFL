import numpy as np
from utils.setting_setup import *
import scipy


class env_agent_utils():
    def __init__(self):
        pass

    def _round(self,input, unit):
        return np.round(input / unit) * unit

    def _wrapState(self):
        self.ChannelGain = self._ChannelGain_Calculated(self.sigma_data)
        state = np.array(self.ChannelGain)
        return state

    def _decomposeState(self, state):
        H = state[0: self.N_User]
        return [
            np.array(H)
        ]

    def _wrapAction(self):
        action = np.concatenate((np.array([[self.beta]]).reshape(1, self.N_User),
                                 np.array([[self.f_u]]).reshape(1, self.N_User),
                                 np.array([[self.p_u]]).reshape(1, self.N_User)), axis=1)
        return action

    def _decomposeAction(self, action):
        # beta :
        # f_u  : maximum local computation capacity of user u
        # p_u  : maximum power of user u
        # butt : local accuracy of users
        # tau  : sampling delay
        beta = action[0][0: self.N_User].astype(float)
        beta = scipy.special.softmax(beta, axis=None)

        f_u = action[0][self.N_User: 2 * self.N_User].astype(float)
        p_u = (action[0][2 * self.N_User: 3 * self.N_User].astype(float))*self.p_u_max
        butt = 0
        tau  = 0

        return [
            np.array(beta).reshape((1,self.N_User)),
            np.array(f_u).reshape((1,self.N_User)),
            np.array(p_u).reshape((1,self.N_User))
        ]

        # return [
        #         np.array(beta).reshape(1, self.N_User).squeeze(),
        #         np.array(o).reshape(1, self.N_User).squeeze(),
        #         np.array(P_n).reshape(1, self.N_User).squeeze()
        #        ]
