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
                                 np.array([[self.p_u]]).reshape(1, self.N_User)), axis=1)
        return action

    def _decomposeAction(self, action):
        # make output for resource allocation beta (range: [0,P_u_max])
        # make output for power (range: [0,1])
        # make output for compression ratio: (range: [0,1])
        beta = action[0][0: self.N_User].astype(float)
        beta = scipy.special.softmax(beta, axis=None)

        f_u = action[0][self.N_User: 2 * self.N_User].astype(float)
        p_u = (action[0][2 * self.N_User: 3 * self.N_User].astype(float))*self.p_u_max

        # print(f"beta: {beta}")
        # print(f"f_u: {o}")
        # print(f"p_u: {P_n}")

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
