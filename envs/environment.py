import math

import numpy as np
from utils.setting_setup import *
import scipy
from dataset_est.get_entropy import *
from envs.env_utils import *
from envs.env_agent_utils import *
from utils.func_utils import *


class SCFL_env(env_utils, env_agent_utils):
    def __init__(self, args):
        # Network setting
        self.noise = args.noise
        self.lamda = args.lamda
        self.N_User = args.user_num
        self.G_CU_list = np.ones((self.N_User, 1))  # User directivity
        self.G_BS_t = 1  # BS directivity
        self.Num_BS = 1  # Number of Base Stations
        self.max_step = args.max_step
        self.Z_u = 10000  # Data size
        self.sigma_data = 0.01

        # Power setting
        self.p_u_max = args.poweru_max
        self.eta = 0.7  # de tinh R_u
        self.sigma = 3.9811 * (np.e ** (-21 + 7))  # -174 dBm/Hz -> W/Hz
        # Bandwidth
        self.B = args.bandwidth

        # Base station initialization
        self.BS_x = 0
        self.BS_y = 0
        self.BS_R_Range = 1
        self.BS_R_min = 0.1

        # initialization
        self.low_freq = args.low_freq
        self.high_freq = args.high_freq
        self.C_u = np.random.uniform(low=self.low_freq, high=self.high_freq, size=self.N_User)
        self.D_u = 500

        self.pen_coeff = args.pen_coeff  # coefficient of penalty defined by lamba in paper
        self.data_size = args.data_size
        self.kappa = 10 ^ -28
        self.f_u_max = args.f_u_max
        self.B = args.Bandwidth

        self.xi = 0.1
        self.coeff = 1
        self.Time_max = args.tmax  # max time per round
        self.lastSample_time = 0.1

        # AI Model/Dataset Coefficient
        self.gamma = args.L * (1 - uniform_generator(mean=0, std=0.1))
        self.Lipschitz = args.L
        self.delta = (2 / args.L) * (1 - uniform_generator(mean=0.2, std=0.2))
        self.eta_accuracy = args.eta_accuracy  # target local accuracy
        self.epsilon0_accuracy = args.epsilon0_accuracy  # target global accuracy

        """ Generalization Gap Calculation """
        self.dataset = "mnist"  # Choose the dataset to get the entropy value , option ["mnist","cifar10"]
        if self.dataset == "mnist":
            self.entropyH = entropy_holder.get_value("mnist_data")
            print("Value entropy of MNIST dataset: ", self.entropyH)
        elif self.dataset == "cifar10":
            self.entropyH = entropy_holder.get_value("cifar10_dataset")
            print("Value entropy of CIFAR10 dataset", self.entropyH)
        else:
            print("Invalid key")
        self.coeff_c0 = self.entropyH
        self.coeff_c1 = 1
        mutual_I = self.coeff_c0 * np.exp(-self.coeff_c1 * self.lastSample_time)
        self.Psi = 2 ** self.entropyH * np.sqrt(2 * (self.entropyH - mutual_I))
        print(f"this is Psi that you need :{self.Psi}")

        """ =============== """
        """     Actions     """
        """ =============== """
        self.beta = np.random.randint(0, self.N_User, size=[self.N_User, 1])
        self.f_u = np.reshape((np.random.rand(1, self.N_User) * self.f_u_max), (self.N_User, 1))
        self.p_u = np.reshape((np.random.rand(1, self.N_User) * self.p_u_max), (self.N_User, 1))

        """ ========================================= """
        """ ===== Function-based Initialization ===== """
        """ ========================================= """
        self.BS_location = np.expand_dims(self._location_BS_Generator(), axis=0)
        self.U_location = self._location_CU_Generator()
        self.User_trajectory = self._trajectory_U_Generator()
        self.distance_CU_BS = self._distance_Calculated(self.U_location, self.BS_location)

        self.ChannelGain = self._ChannelGain_Calculated(self.sigma_data)
        self.commonDataRate = self._calculateDataRate(self.ChannelGain)
        self.E = 0  # initialize rewards)

        """ ============================ """
        """     Environment Settings     """
        """ ============================ """
        self.rewardMatrix = np.array([])
        self.observation_space = self._wrapState().squeeze()
        self.action_space = self._wrapAction()

    def step(self, action, step):
        penalty = 0
        self.beta, self.f_u, self.p_u = self._decomposeAction(action)  #
        # Environment change
        self.User_trajectory = np.expand_dims(self._trajectory_U_Generator(), axis=0)
        self.U_location = self.User_trajectory + self.U_location
        state_next = self._wrapState()  # State wrap

        self.ChannelGain = self._ChannelGain_Calculated(self.sigma_data)  # Re-calculate channel gain
        self.E = self._Energy()  # Energy
        self.num_Iglob = self._calculateGlobalIteration()  # Global Iterations
        self.num_Iu = self._calculateLocalIteration()  # Local Iterations
        self.t_trans = self._calTimeTrans()  # Transmission Time
        self.Au = self.num_Iu * self.C_u * self.D_u  # Iterations x Cycles x Samples

        # Penalty 1:
        penalty += max(np.sum(((self.Au / self.f_u + self.t_trans) - self.Time_max)), 0)
        self.penalty = penalty
        reward = - self.E + self.pen_coeff * penalty

        # Stop at Maximum Glob round
        if (step == self.max_step) & (step == self.num_Iglob):
            done = True
        else:
            done = False
        info = None
        return state_next, reward, done, info

    def reset(self):
        #  System initialization
        self.BS_location = np.expand_dims(self._location_BS_Generator(), axis=0)
        self.U_location = self._location_CU_Generator()
        self.User_trajectory = self._trajectory_U_Generator()
        # Distance calculation
        self.distance_CU_BS = self._distance_Calculated(self.BS_location, self.U_location)

        # re-calculate channel gain
        self.ChannelGain = self._ChannelGain_Calculated(self.sigma_data)
        state_next = self._wrapState()
        return state_next


if __name__ == '__main__':
    args = get_arguments()
    env = SCFL_env(args)
