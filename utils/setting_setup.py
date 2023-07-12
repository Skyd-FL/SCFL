import argparse
from pathlib import Path


def get_arguments():
    parser = argparse.ArgumentParser()
    """ ======================================================== """
    """ ====================== Run config ===================== """
    """ ======================================================== """
    parser.add_argument("--seed", type=int, default=730,
                        help="one manual random seed")
    parser.add_argument("--n-seed", type=int, default=1,
                        help="number of runs")

    # --------------------- Path
    parser.add_argument("--data-dir", type=Path, default="D:/Datasets/",
                        help="Path to the mnist dataset")
    parser.add_argument("--exp-dir", type=Path, default="D:/Github/1-RepresentationLearning/IVAE/experiments",
                        help="Path to the experiment folder, where all logs/checkpoints will be stored")

    """ ======================================================== """
    """ ====================== Flag & name ===================== """
    """ ======================================================== """
    parser.add_argument("--mode", type=str, default="train",
                        help="experiment mode")
    parser.add_argument("--log-delay", type=float, default=2.0,
                        help="Time between two consecutive logs (in seconds)")
    parser.add_argument("--eval", type=bool, default=True,
                        help="Evaluation Trigger")
    parser.add_argument("--log-flag", type=bool, default=False,
                        help="Logging Trigger")
    parser.add_argument("--f-cluster", type=bool, default=True,
                        help="Trigger the clustering to get salient feature of specific categories")
    parser.add_argument("--plot-interval", type=int, default=500000000,
                        help="Number of step needed to plot new accuracy plot")
    parser.add_argument("--save-flag", type=bool, default=True,
                        help="Save Trigger")
    parser.add_argument("--algo", type=str, default="SAC",
                        help="type of training agent")

    """ ======================================================== """
    """ ================== Environment config ================== """
    """ ======================================================== """
    parser.add_argument("--ai-network", type=str, default="cnn",
                        help="AI network type")
    parser.add_argument("--drl-algo", choices=['ddpg-ei', 'ddpg'],
                        default='ddpg-ei',
                        help="choice of DRL algorithm")
    parser.add_argument("--noise", type=float, default=0.01,
                        help="network noise")
    parser.add_argument("--user-num", type=int, default=10,
                        help="number of users")
    parser.add_argument("--lamda", type=float, default=0.001,
                        help="signal wave length")
    parser.add_argument("--power", type=float, default=1,
                        help="max power of BS threshold")
    parser.add_argument("--poweru_max", type=float, default=10,
                        help="max power of user threshold")
    parser.add_argument("--power0", type=float, default=1,
                        help="power of BS")
    parser.add_argument("--powern", type=float, default=1,
                        help="power of users")
    parser.add_argument("--bandwidth", type=float, default=20e6,
                        help="signal bandwidth")
    parser.add_argument("--L", type=float, default=50,
                        help="Lipschitz smooth variables")

    """ ======================================================== """
    """ ===================== Agent config ===================== """
    """ ======================================================== """
    parser.add_argument("--memory-size", type=int, default=10000,
                        help="size of the replay memory")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="data batch size")
    parser.add_argument("--ou-theta", type=float, default=1.0,
                        help="ou noise theta")
    parser.add_argument("--ou-sigma", type=float, default=0.1,
                        help="ou noise sigma")
    parser.add_argument("--initial-steps", type=int, default=1e4,
                        help="initial random steps")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor")
    parser.add_argument("--tau", type=float, default=5e-3,
                        help="initial random steps")
    parser.add_argument("--max-episode", type=int, default=50,
                        help="max episode")
    parser.add_argument("--max-step", type=int, default=500,
                        help="max number of step per episode")
    parser.add_argument("--semantic-mode", type=str, default="learn",
                        help="learn | infer")
    parser.add_argument("--pen-coeff", type=float, default=0,
                        help="coefficient for penalty")

    """
    ____________________________________________________________________________________________________________________
    Args for SAC
    ____________________________________________________________________________________________________________________

    """
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--target_update_interval', type=int, default=1,
                        metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    # parser.add_argument('--tau1', type=float, default=0.005,
    #                     metavar='G',
    #                     help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.2,
                        metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False,
                        metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--hidden_size', type=int, default=128,
                        metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--start_steps', type=int, default=1, metavar='N',
                        help='Steps sampling random actions (default: 100)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    # parser.add_argument('--num_steps', type=int, default=1, metavar='N',
    #                     help='maximum number of steps (default: 1000000)')
    #####################################################################################
    parser.add_argument('--low_freq', type=int, default=10000, metavar='C_u_min',
                        help='min number of CPU cycles needed to compute one sample data (default: 10000)')
    parser.add_argument('--high_freq', type=int, default=30000, metavar='C_u_max',
                        help='max number of CPU cycles needed to compute one sample data (default: 30000)')
    parser.add_argument('--data-size', type=float, default=28.1*1024, metavar='size',
                        help='Data size of user to transmit [bit] (default: 30000)')
    parser.add_argument('--Bandwidth', type=float, default=20e6, metavar='system_bandwidth',
                        help='Bandwidth of system [Hz] (default: 20e6)')
    parser.add_argument('--f_u_max', type=float, default=2e9, metavar='computation_capacity_max',
                        help='Computation capacity of user [Hz] (default: 20e6)')
    parser.add_argument('--eta_accuracy', type=float, default=0.01, metavar='local_accuracy',
                        help='Local accuracy of system (default: 0.01)')
    parser.add_argument('--epsilon0_accuracy', type=float, default=0.01, metavar='global_accuracy',
                        help='Global accuracy of system (default: 0.01)')
    parser.add_argument('--tmax', type=int, default=10, metavar='T_max',
                        help='Upper bound of time completion (default: 1000)')
    return parser.parse_args()
