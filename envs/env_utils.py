import numpy as np
from utils.setting_setup import *
import scipy
import math


class env_utils():
    def __init__(self):
        pass

    def _location_BS_Generator(self):
        BS_location = [self.BS_x, self.BS_y]
        # print(BS_location)
        return np.array(BS_location)

    # Iot initialization
    def _location_CU_Generator(self):
        userList = []
        # hUser_temp = 1.65
        for i in range(self.N_User):
            r = self.BS_R_Range * np.sqrt(np.random.rand()) + self.BS_R_min
            theta = np.random.uniform(-np.pi, np.pi)
            xUser_temp = self.BS_x + r * np.cos(theta)
            yUser_temp = self.BS_y + r * np.sin(theta)
            userList.append([xUser_temp, yUser_temp])
            U_location = np.array(userList)
            # print(U_location)
        return U_location

    def _trajectory_U_Generator(self):
        userList = []
        for i in range(self.N_User):
            theta = 0
            theta = theta + np.pi / 360
            r = np.sin(theta)
            xUser_temp = r * np.cos(2 * theta)
            yUser_temp = r * np.sin(2 * theta)
            userList.append([xUser_temp, yUser_temp])
            User_trajectory = np.array(userList)
        return User_trajectory

    def _distance_Calculated(self, A, B):
        return np.array([np.sqrt(np.sum((A - B) ** 2, axis=1))]).transpose()

    # def _ChannelGain_Calculated(self, sigma_data):
    #     print(f"DataRate: {self.G_BS_t}|"
    #           f"P_u: {self.G_CU_list}|"
    #           f"Channel: {self.lamda}|"
    #           f"AWGN: {awgn_coeff}|"
    #           f"Nume: {numerator}|"
    #           f"Deno: {denominator}")
    #     numerator = self.G_BS_t * self.G_CU_list * (self.lamda ** 2)
    #     denominator = (4 * np.pi * self.distance_CU_BS) ** 2
    #     awgn_coeff = np.random.normal(1, sigma_data)
    #     ChannelGain = (numerator * awgn_coeff) / denominator
    #     # print(ChannelGain)
    #     return np.array(ChannelGain)

    def _ChannelGain_Calculate(self, sigma_data):
        speed_of_light = 3 * 10 ** 8  # Speed of light in meters per second
        pi = math.pi
        awgn_coeff = np.random.normal(1, sigma_data)
        ChannelGain = (speed_of_light * awgn_coeff / (4 * pi * self.freq_carrier * self.distance_CU_BS)) ** 0.5

        return np.array(ChannelGain)

    def _calculateDataRate(self, channelGain_BS_CU):
        """
        The SNR is measured by:
        :param self:
        :param channelGain_BS_CU:
        :return:
        SNR = Numerator / Denominator
        Numerator = H_k * P_k
        Denominator = N_0 * B_k
        Datarate = B_k np.log2(1+Numerator/Denominator)
        """
        Numerator = channelGain_BS_CU * self.p_u  # self.P must be a list among all users [1, ... , U]
        Denominator = self.B * self.beta * self.sigma  # self.B must be a list among all users [1, ... , U]

        DataRate = self.B * self.beta * np.log2(1 + (Numerator / Denominator))
        # print(f"DataRate: {DataRate}|"
        #       f"P_u: {self.p_u}|"
        #       f"Channel: {channelGain_BS_CU}|"
        #       f"Bandwidth: {self.B}|"
        #       f"Allocation: {self.beta}|"
        #       f"Noise: {self.sigma}")
        return DataRate

    def _calculateGlobalIteration(self):
        """
        :params    : Lipschitz ~ L-smooth
        :params    : Xi ~ Constants
        :params    : N_User ~ Number of users
        :params    : Eta_accuracy ~
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        :variables : epsilon0_accuracy ~ local accuracy of the local users
        :variables :
        ==================================================
        :return: required number of rounds for convergence
        """
        Numerator = np.log(1 / self.epsilon0_accuracy) * 2 * self.N_User * self.Lipschitz * self.xi
        Denominator = self.xi * (self.Lipschitz + 2) * self.Psi + \
                      self.xi * self.Lipschitz / self.N_User - self.eta_accuracy * self.gamma
        return Numerator / Denominator

    def _calculateLocalIteration(self):
        # print(f"L-smooth: {self.Lipschitz}|"
        #       f"Strongly Convex: {self.gamma}|"
        #       f"Step size: {self.delta}|"
        #       f"Target Accuracy: {self.eta_accuracy}")
        return 2 / ((2 - self.Lipschitz * self.delta) * self.delta * self.gamma) * np.log2(1 / self.eta_accuracy)

    def _calTimeTrans(self):
        self.DataRate = self._calculateDataRate(self.ChannelGain.reshape(1, -1))
        print(self.data_size)
        return np.divide(self.data_size, self.DataRate)

    def _Energy(self):
        """
        Intermediate Energy
        :return:
        """
        self.DataRate = self._calculateDataRate(self.ChannelGain.reshape(1, -1))
        # Calculate computation energy
        self.num_Iu = 2 / ((2 - self.Lipschitz * self.delta) * self.delta * self.gamma) * np.log2(1 / self.eta_accuracy)
        self.EC_u = self.num_Iu * self.kappa * self.C_u * self.D_u * self.f_u ** 2
        # Calculate transmission energy
        self.t_trans = self._calTimeTrans()
        self.ET_u = np.multiply(self.p_u, self.t_trans)

        return np.sum(self.EC_u) + np.sum(self.ET_u)
