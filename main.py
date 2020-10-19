import argparse

from hrac.train import run_hrac


parser = argparse.ArgumentParser()

parser.add_argument("--algo", default="hrac", type=str)
parser.add_argument("--seed", default=2, type=int)
parser.add_argument("--eval_freq", default=5e3, type=float)
parser.add_argument("--max_timesteps", default=5e6, type=float)
parser.add_argument("--save_models", action="store_true")
parser.add_argument("--env_name", default="AntMaze", type=str)
parser.add_argument("--load", default=False, type=bool)
parser.add_argument("--log_dir", default="./logs", type=str)
parser.add_argument("--no_correction", action="store_true")
parser.add_argument("--inner_dones", action="store_true")
parser.add_argument("--absolute_goal", action="store_true")
parser.add_argument("--binary_int_reward", action="store_true")
parser.add_argument("--load_adj_net", default=False, action="store_true")

parser.add_argument("--gid", type=int, default=0)
parser.add_argument("--traj_buffer_size", type=int, default=50000)
parser.add_argument("--lr_r", type=float, default=2e-4)
parser.add_argument("--r_margin_pos", type=float, default=1.0)
parser.add_argument("--r_margin_neg", type=float, default=1.2)
parser.add_argument("--r_training_epochs", type=int, default=25)
parser.add_argument("--r_batch_size", type=int, default=64)
parser.add_argument("--r_hidden_dim", type=int, default=128)
parser.add_argument("--r_embedding_dim", type=int, default=32)
parser.add_argument("--goal_loss_coeff", type=float, default=20.)

parser.add_argument("--manager_propose_freq", default=10, type=int)
parser.add_argument("--train_manager_freq", default=10, type=int)
parser.add_argument("--discount", default=0.99, type=float)
parser.add_argument("--ctrl_discount", default=0.95, type=float)

# Manager Parameters
parser.add_argument("--man_tau", default=0.005, type=float)
parser.add_argument("--man_batch_size", default=128, type=int)
parser.add_argument("--man_buffer_size", default=2e5, type=int)
parser.add_argument("--man_rew_scale", default=0.1, type=float)
parser.add_argument("--man_act_lr", default=1e-4, type=float)
parser.add_argument("--man_crit_lr", default=1e-3, type=float)
parser.add_argument("--candidate_goals", default=10, type=int)

# Controller Parameters
parser.add_argument("--ctrl_tau", default=0.005, type=float)
parser.add_argument("--ctrl_batch_size", default=128, type=int)
parser.add_argument("--ctrl_buffer_size", default=2e5, type=int)
parser.add_argument("--ctrl_rew_scale", default=1.0, type=float)
parser.add_argument("--ctrl_act_lr", default=1e-4, type=float)
parser.add_argument("--ctrl_crit_lr", default=1e-3, type=float)

# Noise Parameters
parser.add_argument("--noise_type", default="normal", type=str)
parser.add_argument("--ctrl_noise_sigma", default=1., type=float)
parser.add_argument("--man_noise_sigma", default=1., type=float)

# Run the algorithm
args = parser.parse_args()

if args.env_name in ["AntGather", "AntMazeSparse"]:
    args.man_rew_scale = 1.0
    if args.env_name == "AntGather":
        args.inner_dones = True

run_hrac(args)
