import argparse

from hrac.eval import eval_hrac


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", default="hrac", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gid", type=int, default=0)
    parser.add_argument("--env_name", default="AntMaze", type=str)
    parser.add_argument("--eval_episodes", default=100, type=int)
    parser.add_argument("--load", default=True, type=bool)
    parser.add_argument("--model_dir", default="./pretrained_models", type=str)
    parser.add_argument("--manager_propose_freq", default=10, type=int)
    parser.add_argument("--absolute_goal", action="store_true")
    parser.add_argument("--binary_int_reward", action="store_true")

    args = parser.parse_args()

    eval_hrac(args)
