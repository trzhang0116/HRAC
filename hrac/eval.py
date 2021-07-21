import torch

import numpy as np
import os

import hrac.hrac as hrac
from envs import EnvWithGoal, GatherEnv
from envs.create_maze_env import create_maze_env
from envs.create_gather_env import create_gather_env


def evaluate_policy(env, env_name, manager_policy, controller_policy,
                    calculate_controller_reward, ctrl_rew_scale,
                    manager_propose_frequency=10, eval_episodes=100):
    print("Starting evaluation...")
    env.evaluate = True

    with torch.no_grad():
        avg_reward = 0.
        avg_controller_rew = 0.
        global_steps = 0
        goals_achieved = 0
        for eval_ep in range(eval_episodes):
            obs = env.reset()

            goal = obs["desired_goal"]
            state = obs["observation"]

            done = False
            step_count = 0
            env_goals_achieved = 0
            while not done:
                if step_count % manager_propose_frequency == 0:
                    subgoal = manager_policy.sample_goal(state, goal)

                step_count += 1
                global_steps += 1
                action = controller_policy.select_action(state, subgoal, evaluation=True)
                new_obs, reward, done, _ = env.step(action)
                if env_name != "AntGather" and env.success_fn(reward):
                    env_goals_achieved += 1
                    goals_achieved += 1
                    done = True

                goal = new_obs["desired_goal"]
                new_state = new_obs["observation"]

                subgoal = controller_policy.subgoal_transition(state, subgoal, new_state)

                avg_reward += reward
                avg_controller_rew += calculate_controller_reward(state, subgoal, new_state, ctrl_rew_scale)

                state = new_state

        avg_reward /= eval_episodes
        avg_controller_rew /= global_steps
        avg_step_count = global_steps / eval_episodes
        avg_env_finish = goals_achieved / eval_episodes

        print("---------------------------------------")
        print("Evaluation over {} episodes:\nAvg Ctrl Reward: {:.3f}".format(eval_episodes, avg_controller_rew))
        if env_name == "AntGather":
            print("Avg reward: {:.1f}".format(avg_reward))
        else:
            print("Goals achieved: {:.1f}%".format(100*avg_env_finish))
        print("Avg Steps to finish: {:.1f}".format(avg_step_count))
        print("---------------------------------------")

        env.evaluate = False
        return avg_reward, avg_controller_rew, avg_step_count, avg_env_finish


def get_reward_function(dims, absolute_goal=False, binary_reward=False):
    if absolute_goal and binary_reward:
        def controller_reward(z, subgoal, next_z, scale):
            z = z[:dims]
            next_z = next_z[:dims]
            reward = float(np.linalg.norm(subgoal - next_z, axis=-1) <= 1) * scale
            return reward
    elif absolute_goal:
        def controller_reward(z, subgoal, next_z, scale):
            z = z[:dims]
            next_z = next_z[:dims]
            reward = -np.linalg.norm(subgoal - next_z, axis=-1) * scale
            return reward
    elif binary_reward:
        def controller_reward(z, subgoal, next_z, scale):
            z = z[:dims]
            next_z = next_z[:dims]
            reward = float(np.linalg.norm(z + subgoal - next_z, axis=-1) <= 1) * scale
            return reward
    else:
        def controller_reward(z, subgoal, next_z, scale):
            z = z[:dims]
            next_z = next_z[:dims]
            reward = -np.linalg.norm(z + subgoal - next_z, axis=-1) * scale
            return reward

    return controller_reward


def eval_hrac(args):
    if args.env_name == "AntGather":
        env = GatherEnv(create_gather_env(args.env_name, args.seed), args.env_name)
        env.seed(args.seed)   
    elif args.env_name in ["AntMaze", "AntMazeSparse", "AntPush", "AntFall"]:
        env = EnvWithGoal(create_maze_env(args.env_name, args.seed), args.env_name)
        env.seed(args.seed)
    else:
        raise NotImplementedError

    low = np.array((-10, -10, -0.5, -1, -1, -1, -1,
                    -0.5, -0.3, -0.5, -0.3, -0.5, -0.3, -0.5, -0.3))
    max_action = float(env.action_space.high[0])
    high = -low
    man_scale = (high - low) / 2
    if args.env_name == "AntFall":
        controller_goal_dim = 3
    else:
        controller_goal_dim = 2
    if args.absolute_goal:
        man_scale[0] = 30
        man_scale[1] = 30
        no_xy = False
    else:
        no_xy = True
    action_dim = env.action_space.shape[0]
    discrete = False

    obs = env.reset()

    goal = obs["desired_goal"]
    state = obs["observation"]

    torch.cuda.set_device(args.gid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_name = "{}_{}_{}".format(args.env_name, args.algo, args.seed)
    output_data = {"frames": [], "reward": [], "dist": []}    

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    state_dim = state.shape[0]
    if args.env_name in ["AntMaze", "AntPush", "AntFall"]:
        goal_dim = goal.shape[0]
    else:
        goal_dim = 0

    controller_policy = hrac.Controller(
        state_dim=state_dim,
        goal_dim=controller_goal_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_lr=0,
        critic_lr=0,
        no_xy=no_xy,
        absolute_goal=args.absolute_goal,
        policy_noise=0,
        noise_clip=0
    )

    manager_policy = hrac.Manager(
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=controller_goal_dim,
        actor_lr=0,
        critic_lr=0,
        candidate_goals=10,
        correction=True,
        scale=man_scale,
        goal_loss_coeff=10.,
        absolute_goal=args.absolute_goal
    )

    if args.load:
        try:
            manager_policy.load(args.model_dir, args.env_name, args.algo)
            controller_policy.load(args.model_dir, args.env_name, args.algo)
            print("Loaded successfully.")
            just_loaded = True
        except Exception as e:
            just_loaded = False
            print(e, "Loading failed.")
    else:
        just_loaded = False

    calculate_controller_reward = get_reward_function(
        controller_goal_dim, absolute_goal=args.absolute_goal, binary_reward=args.binary_int_reward)

    evaluate_policy(env, args.env_name, manager_policy, controller_policy, calculate_controller_reward,
                    1.0, args.manager_propose_freq, args.eval_episodes)
