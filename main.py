import argparse
import datetime
import numpy as np
import itertools
import torch
# from torch.utils.tensorboard import SummaryWriter
from agents.sac.agent_sac import *
from agents.ddpg.agent import *

from agents.sac.modules.replay_memory import *
from envs.environment import *
from envs.env_utils import *
from envs.env_agent_utils import *
from utils.setting_setup import *
from utils.result_utils import *
from beautifultable import BeautifulTable


args = get_arguments()

table = BeautifulTable(maxwidth=140, detect_numerics=False)
table.rows.append(["AI Network", args.ai_network, "Algorithm", args.algo, "Plot Interval", args.plot_interval])
table.rows.append(["Comp. Cap.", args.f_u_max, "Data Size", args.data_size, "Lipschitz", args.L])
table.rows.append(["F_c(MHz)", args.freq_carrier, "Bandwidth", args.bandwidth, "Num. User", args.user_num])
table.rows.append(["AI Network", args.ai_network, "Algorithm", args.algo, "Plot Interval", args.plot_interval])

print(table)

env = SCFL_env(args)

if args.algo == "DDPG":
    agent = DDPGAgent(
        args,
        env
    )
elif args.algo == "SAC":
    agent = SAC(
        args,
        env
    )
else:
    agent = SAC(
        args,
        env
    )
# agent.train(args)
