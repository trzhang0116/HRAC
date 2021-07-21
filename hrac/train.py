import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np
import pandas as pd
from math import ceil

import hrac.utils as utils
import hrac.hrac as hrac
from hrac.models import ANet
from envs import EnvWithGoal, GatherEnv
from envs.create_maze_env import create_maze_env
from envs.create_gather_env import create_gather_env


"""
HIRO part adapted from
https://github.com/bhairavmehta95/data-efficient-hrl/blob/master/hiro/train_hiro.py
"""


def evaluate_policy(env, env_name, manager_policy, controller_policy,
                    calculate_controller_reward, ctrl_rew_scale,
                    manager_propose_frequency=10, eval_idx=0, eval_episodes=5):
    print("Starting evaluation number {}...".format(eval_idx))
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
            reward = float(np.linalg.norm(subgoal - next_z, axis=-1) <= 1.414) * scale
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
            reward = float(np.linalg.norm(z + subgoal - next_z, axis=-1) <= 1.414) * scale
            return reward
    else:
        def controller_reward(z, subgoal, next_z, scale):
            z = z[:dims]
            next_z = next_z[:dims]
            reward = -np.linalg.norm(z + subgoal - next_z, axis=-1) * scale
            return reward

    return controller_reward


def update_amat_and_train_anet(n_states, adj_mat, state_list, state_dict, a_net, traj_buffer,
        optimizer_r, controller_goal_dim, device, args):
    for traj in traj_buffer.get_trajectory():
        for i in range(len(traj)):
            for j in range(1, min(args.manager_propose_freq, len(traj) - i)):
                s1 = tuple(np.round(traj[i][:controller_goal_dim]).astype(np.int32))
                s2 = tuple(np.round(traj[i+j][:controller_goal_dim]).astype(np.int32))
                if s1 not in state_list:
                    state_list.append(s1)
                    state_dict[s1] = n_states
                    n_states += 1
                if s2 not in state_list:
                    state_list.append(s2)
                    state_dict[s2] = n_states
                    n_states += 1
                adj_mat[state_dict[s1], state_dict[s2]] = 1
                adj_mat[state_dict[s2], state_dict[s1]] = 1
    print("Explored states: {}".format(n_states))
    flags = np.ones((30, 30))
    for s in state_list:
        flags[int(s[0]), int(s[1])] = 0
    print(flags)
    if not args.load_adj_net:
        print("Training adjacency network...")
        utils.train_adj_net(a_net, state_list, adj_mat[:n_states, :n_states],
                            optimizer_r, args.r_margin_pos, args.r_margin_neg,
                            n_epochs=args.r_training_epochs, batch_size=args.r_batch_size,
                            device=device, verbose=False)

        if args.save_models:
            r_filename = os.path.join("./models", "{}_{}_a_network.pth".format(args.env_name, args.algo))
            torch.save(a_net.state_dict(), r_filename)
            print("----- Adjacency network {} saved. -----".format(episode_num))

    traj_buffer.reset()

    return n_states


def run_hrac(args):
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_models and not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(os.path.join(args.log_dir, args.algo)):
        os.makedirs(os.path.join(args.log_dir, args.algo))
    output_dir = os.path.join(args.log_dir, args.algo)
    print("Logging in {}".format(output_dir))

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
    policy_noise = 0.2
    noise_clip = 0.5
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

    obs = env.reset()

    goal = obs["desired_goal"]
    state = obs["observation"]

    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.algo))
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
        actor_lr=args.ctrl_act_lr,
        critic_lr=args.ctrl_crit_lr,
        no_xy=no_xy,
        absolute_goal=args.absolute_goal,
        policy_noise=policy_noise,
        noise_clip=noise_clip
    )

    manager_policy = hrac.Manager(
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=controller_goal_dim,
        actor_lr=args.man_act_lr,
        critic_lr=args.man_crit_lr,
        candidate_goals=args.candidate_goals,
        correction=not args.no_correction,
        scale=man_scale,
        goal_loss_coeff=args.goal_loss_coeff,
        absolute_goal=args.absolute_goal
    )

    calculate_controller_reward = get_reward_function(
        controller_goal_dim, absolute_goal=args.absolute_goal, binary_reward=args.binary_int_reward)

    if args.noise_type == "ou":
        man_noise = utils.OUNoise(state_dim, sigma=args.man_noise_sigma)
        ctrl_noise = utils.OUNoise(action_dim, sigma=args.ctrl_noise_sigma)

    elif args.noise_type == "normal":
        man_noise = utils.NormalNoise(sigma=args.man_noise_sigma)
        ctrl_noise = utils.NormalNoise(sigma=args.ctrl_noise_sigma)

    manager_buffer = utils.ReplayBuffer(maxsize=args.man_buffer_size)
    controller_buffer = utils.ReplayBuffer(maxsize=args.ctrl_buffer_size)

    # Initialize adjacency matrix and adjacency network
    n_states = 0
    state_list = []
    state_dict = {}
    adj_mat = np.diag(np.ones(1000, dtype=np.uint8))
    traj_buffer = utils.TrajectoryBuffer(capacity=args.traj_buffer_size)
    a_net = ANet(controller_goal_dim, args.r_hidden_dim, args.r_embedding_dim)
    if args.load_adj_net:
        print("Loading adjacency network...")
        a_net.load_state_dict(torch.load("./models/a_network.pth"))
    a_net.to(device)
    optimizer_r = optim.Adam(a_net.parameters(), lr=args.lr_r)

    if args.load:
        try:
            manager_policy.load("./models")
            controller_policy.load("./models")
            print("Loaded successfully.")
            just_loaded = True
        except Exception as e:
            just_loaded = False
            print(e, "Loading failed.")
    else:
        just_loaded = False

    # Logging Parameters
    total_timesteps = 0
    timesteps_since_eval = 0
    timesteps_since_manager = 0
    episode_timesteps = 0
    timesteps_since_subgoal = 0
    episode_num = 0
    done = True
    evaluations = []

    # Train
    while total_timesteps < args.max_timesteps:
        if done:
            if total_timesteps != 0 and not just_loaded:
                if episode_num % 10 == 0:
                    print("Episode {}".format(episode_num))
                # Train controller
                ctrl_act_loss, ctrl_crit_loss = controller_policy.train(controller_buffer, episode_timesteps,
                    batch_size=args.ctrl_batch_size, discount=args.ctrl_discount, tau=args.ctrl_soft_sync_rate)
                if episode_num % 10 == 0:
                    print("Controller actor loss: {:.3f}".format(ctrl_act_loss))
                    print("Controller critic loss: {:.3f}".format(ctrl_crit_loss))
                writer.add_scalar("data/controller_actor_loss", ctrl_act_loss, total_timesteps)
                writer.add_scalar("data/controller_critic_loss", ctrl_crit_loss, total_timesteps)

                writer.add_scalar("data/controller_ep_rew", episode_reward, total_timesteps)
                writer.add_scalar("data/manager_ep_rew", manager_transition[4], total_timesteps)

                # Train manager
                if timesteps_since_manager >= args.train_manager_freq:
                    timesteps_since_manager = 0
                    r_margin = (args.r_margin_pos + args.r_margin_neg) / 2

                    man_act_loss, man_crit_loss, man_goal_loss = manager_policy.train(controller_policy,
                        manager_buffer, ceil(episode_timesteps/args.train_manager_freq),
                        batch_size=args.man_batch_size, discount=args.man_discount, tau=args.man_soft_sync_rate,
                        a_net=a_net, r_margin=r_margin)
                    
                    writer.add_scalar("data/manager_actor_loss", man_act_loss, total_timesteps)
                    writer.add_scalar("data/manager_critic_loss", man_crit_loss, total_timesteps)
                    writer.add_scalar("data/manager_goal_loss", man_goal_loss, total_timesteps)

                    if episode_num % 10 == 0:
                        print("Manager actor loss: {:.3f}".format(man_act_loss))
                        print("Manager critic loss: {:.3f}".format(man_crit_loss))
                        print("Manager goal loss: {:.3f}".format(man_goal_loss))

                # Evaluate
                if timesteps_since_eval >= args.eval_freq:
                    timesteps_since_eval = 0
                    avg_ep_rew, avg_controller_rew, avg_steps, avg_env_finish =\
                        evaluate_policy(env, args.env_name, manager_policy, controller_policy,
                            calculate_controller_reward, args.ctrl_rew_scale, args.manager_propose_freq,
                            len(evaluations))

                    writer.add_scalar("eval/avg_ep_rew", avg_ep_rew, total_timesteps)
                    writer.add_scalar("eval/avg_controller_rew", avg_controller_rew, total_timesteps)

                    evaluations.append([avg_ep_rew, avg_controller_rew, avg_steps])
                    output_data["frames"].append(total_timesteps)
                    if args.env_name == "AntGather":
                        output_data["reward"].append(avg_env_finish)
                        writer.add_scalar("eval/avg_steps_to_finish", avg_steps, total_timesteps)
                        writer.add_scalar("eval/perc_env_goal_achieved", avg_env_finish, total_timesteps)
                    else:
                        output_data["reward"].append(avg_ep_rew)
                    output_data["dist"].append(-avg_controller_rew)

                    if args.save_models:
                        controller_policy.save("./models", args.env_name, args.algo)
                        manager_policy.save("./models", args.env_name, args.algo)

                if traj_buffer.full():
                     n_states = update_amat_and_train_anet(n_states, adj_mat, state_list, state_dict, a_net, traj_buffer,
                        optimizer_r, controller_goal_dim, device, args)

                if len(manager_transition[-2]) != 1:                    
                    manager_transition[1] = state
                    manager_transition[5] = float(True)
                    manager_buffer.add(manager_transition)

            obs = env.reset()
            goal = obs["desired_goal"]
            state = obs["observation"]
            traj_buffer.create_new_trajectory()
            traj_buffer.append(state)
            done = False
            episode_reward = 0
            episode_timesteps = 0
            just_loaded = False
            episode_num += 1

            subgoal = manager_policy.sample_goal(state, goal)
            if not args.absolute_goal:
                subgoal = man_noise.perturb_action(subgoal,
                    min_action=-man_scale[:controller_goal_dim], max_action=man_scale[:controller_goal_dim])
            else:
                subgoal = man_noise.perturb_action(subgoal,
                    min_action=np.zeros(controller_goal_dim), max_action=2*man_scale[:controller_goal_dim])

            timesteps_since_subgoal = 0
            manager_transition = [state, None, goal, subgoal, 0, False, [state], []]

        action = controller_policy.select_action(state, subgoal)
        action = ctrl_noise.perturb_action(action, -max_action, max_action)
        action_copy = action.copy()

        next_tup, manager_reward, done, _ = env.step(action_copy)

        manager_transition[4] += manager_reward * args.man_rew_scale
        manager_transition[-1].append(action)

        next_goal = next_tup["desired_goal"]
        next_state = next_tup["observation"]

        manager_transition[-2].append(next_state)
        traj_buffer.append(next_state)

        controller_reward = calculate_controller_reward(state, subgoal, next_state, args.ctrl_rew_scale)
        subgoal = controller_policy.subgoal_transition(state, subgoal, next_state)

        controller_goal = subgoal
        episode_reward += controller_reward

        if args.inner_dones:
            ctrl_done = done or timesteps_since_subgoal % args.manager_propose_freq == 0
        else:
            ctrl_done = done

        controller_buffer.add(
            (state, next_state, controller_goal, action, controller_reward, float(ctrl_done), [], []))

        state = next_state
        goal = next_goal

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
        timesteps_since_manager += 1
        timesteps_since_subgoal += 1

        if timesteps_since_subgoal % args.manager_propose_freq == 0:
            manager_transition[1] = state
            manager_transition[5] = float(done)

            manager_buffer.add(manager_transition)
            subgoal = manager_policy.sample_goal(state, goal)

            if not args.absolute_goal:
                subgoal = man_noise.perturb_action(subgoal,
                    min_action=-man_scale[:controller_goal_dim], max_action=man_scale[:controller_goal_dim])
            else:
                subgoal = man_noise.perturb_action(subgoal,
                    min_action=np.zeros(controller_goal_dim), max_action=2*man_scale[:controller_goal_dim])

            timesteps_since_subgoal = 0
            manager_transition = [state, None, goal, subgoal, 0, False, [state], []]

    # Final evaluation
    avg_ep_rew, avg_controller_rew, avg_steps, avg_env_finish = evaluate_policy(
        env, args.env_name, manager_policy, controller_policy, calculate_controller_reward,
        args.ctrl_rew_scale, args.manager_propose_freq, len(evaluations))
    evaluations.append([avg_ep_rew, avg_controller_rew, avg_steps])
    output_data["frames"].append(total_timesteps)
    if args.env_name == 'AntGather':
        output_data["reward"].append(avg_ep_rew)
    else:
        output_data["reward"].append(avg_env_finish)
    output_data["dist"].append(-avg_controller_rew)

    if args.save_models:
        controller_policy.save("./models", args.env_name, args.algo)
        manager_policy.save("./models", args.env_name, args.algo)

    writer.close()

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(os.path.join("./results", file_name+".csv"), float_format="%.4f", index=False)
    print("Training finished.")
