import os
import argparse

from solver import Solver


def main(args):
    solver = Solver(args)
    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='hrac')
    parser.add_argument('--env_name', type=str, default='KeyChest', help='Maze or KeyChest')
    parser.add_argument('--gid', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--eval_freq', type=int, default=50)
    parser.add_argument('--eval_episodes', type=int, default=50)
    parser.add_argument('--load_model', action='store_true', default=False)
    parser.add_argument('--save_models', action='store_true', default=False)

    # Policy parameters
    parser.add_argument('--man_act_lr', type=float, default=1e-4)
    parser.add_argument('--man_crit_lr', type=float, default=1e-3)
    parser.add_argument('--ctrl_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay_critic', type=float, default=0.)
    parser.add_argument('--hidden_dim', type=int, default=300)

    parser.add_argument('--manager_propose_freq', '-k', type=int, default=10)
    parser.add_argument('--man_discount', type=float, default=0.99)
    parser.add_argument('--ctrl_discount', type=float, default=0.99)
    parser.add_argument('--man_rew_scale', type=float, default=1.0)
    parser.add_argument('--ctrl_rew_scale', type=float, default=1.0)
    parser.add_argument('--goal_loss_coeff', type=float, default=20.0)
    parser.add_argument('--n_noisy_goals', type=int, default=20)

    parser.add_argument('--man_buffer_size', '-c', type=int, default=20000)
    parser.add_argument('--man_batch_size', type=int, default=64)
    parser.add_argument('--man_soft_sync_rate', '-r', type=float, default=0.001)
    parser.add_argument('--man_noise_sigma', '-s', type=float, default=5.0)
    parser.add_argument('--man_policy_update_freq', '-f', type=int, default=2)

    parser.add_argument('--ctrl_entropy', type=float, default=0.01)

    # Adjacency network parameters
    parser.add_argument('--lr_r', type=float, default=2e-4)
    parser.add_argument('--r_margin_pos', type=float, default=1.0)
    parser.add_argument('--r_margin_neg', type=float, default=1.2)
    parser.add_argument('--r_init_steps', type=int, default=50000)
    parser.add_argument('--r_init_epochs', type=int, default=50)
    parser.add_argument('--r_training_freq', type=int, default=50000)
    parser.add_argument('--r_training_epochs', type=int, default=25)
    parser.add_argument('--r_batch_size', type=int, default=64)

    parser.add_argument('--r_hidden_dim', type=int, default=128)
    parser.add_argument('--r_embedding_dim', type=int, default=32)

    args = parser.parse_args()

    print('=' * 30)
    for key, val in vars(args).items():
        print('{}: {}'.format(key, val))

    main(args)
