import os
import argparse
import time
import gym
import torch
import numpy as np
from itertools import count
import logging

from sac.replay_memory import ReplayMemory
from sac.sac import SAC
from sac.sac_model import SAC_MODEL
from model import EnsembleDynamicsModel
from predict_env import PredictEnv
from sample_env import EnvSampler

from flow import flow
from flow.coupledflow import CoupledFlow
from flow.rewarder import Rewarder



now = int(round(time.time() * 1000))
now02 = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(now / 1000))

def readParser():
    parser = argparse.ArgumentParser(description='MBPO')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--env_name', default="Hopper-v2",
                        help='Mujoco Gym environment (default: Hopper-v2)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')

    parser.add_argument('--use_decay', type=bool, default=True, metavar='G',
                        help='discount factor for reward (default: 0.99)')

    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--target_entropy', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')

    parser.add_argument('--num_networks', type=int, default=7, metavar='E',
                        help='ensemble size (default: 7)')
    parser.add_argument('--num_elites', type=int, default=5, metavar='E',
                        help='elite size (default: 5)')
    parser.add_argument('--pred_hidden_size', type=int, default=200, metavar='E',help='hidden size for predictive model')
    parser.add_argument('--reward_size', type=int, default=1, metavar='E',
                        help='environment reward size')

    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')

    parser.add_argument('--model_retain_epochs', type=int, default=1, metavar='A',
                        help='retain epochs')
    parser.add_argument('--model_train_freq', type=int, default=250, metavar='A',help='frequency of training')
    parser.add_argument('--rollout_batch_size', type=int, default=100000, metavar='A',help='rollout number M')
    parser.add_argument('--epoch_length', type=int, default=1000, metavar='A',
                        help='steps per epoch')
    parser.add_argument('--num_epoch', type=int, default=1000, metavar='A',
                        help='total number of epochs')
    parser.add_argument('--min_pool_size', type=int, default=1000, metavar='A',
                        help='minimum pool size')
    parser.add_argument('--real_ratio', type=float, default=0.05, metavar='A',
                        help='ratio of env samples / model samples')
    parser.add_argument('--train_every_n_steps', type=int, default=1, metavar='A',
                        help='frequency of training policy')
    parser.add_argument('--num_train_repeat', type=int, default=20, metavar='A',
                        help='times to training policy per step')
    parser.add_argument('--max_train_repeat_per_step', type=int, default=5, metavar='A',
                        help='max training times per step')
    parser.add_argument('--policy_train_batch_size', type=int, default=256, metavar='A',
                        help='batch size for training policy')
    parser.add_argument('--init_exploration_steps', type=int, default=5000, metavar='A',
                        help='exploration steps initially')
    parser.add_argument('--max_path_length', type=int, default=1000, metavar='A',
                        help='max length of path')


    parser.add_argument('--model_type', default='pytorch', metavar='A',
                        help='predict model -- pytorch or tensorflow')

    parser.add_argument('--cuda', default=True, action="store_true",
                        help='run on CUDA (default: True)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automaically adjust α (default: False)')


    # new add
    parser.add_argument('--MPCHorizon', default=6)
    parser.add_argument('--n_trajs', default=6)
    parser.add_argument('--model_gamma', default=0.99)
    parser.add_argument('--ClipDMoNoise', default=0.5)
    #parser.add_argument('--deter_model', default=False)
    parser.add_argument('--deter_model', default=True)
    parser.add_argument('--StoPoMPC', default=False)
    #parser.add_argument('--StoPoMPC', default=True)
    parser.add_argument('--flow_option', default=1)
    parser.add_argument('--flow_depth', default=2)

    parser.add_argument('--Eva_PretraModel', default=False)
    parser.add_argument('--checkpoint_path', type=str, default='log/checkpoint',
                        help='Path to a checkpoint file to resume training from.')
    parser.add_argument('--use_algo', type=str, default='discriminator',help='flowrl, mbpo')



    return parser.parse_args()


def load_model(checkpoint_path, agent):
    checkpoint = torch.load(checkpoint_path)
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    total_step = checkpoint['total_step']
    sum_reward = checkpoint['sum_reward']
    args = checkpoint['args']
    return agent, total_step, sum_reward, args

def evaluate(args, env_sampler, agent, checkpoint_path):
    # Load the pre-trained model
    agent, _, _, _ = load_model(checkpoint_path, agent)
    print(f"Loaded model from {checkpoint_path}")

    sum_reward = 0
    done = False
    test_step = 0
    env_sampler.current_state = None

    while not done and test_step != args.max_path_length:
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, eval_t=True)
        sum_reward += reward
        test_step += 1

    print(f"Evaluation reward: {sum_reward}")
def train(args, env_sampler, predict_env, agent, env_pool, model_pool, couple_flow, rewarder, model_agent, rollout_schedule):
    total_step = 0
    rollout_length =1
    rollout_depth =args.flow_depth

    # Load model if a checkpoint is provided
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        agent, total_step, _, _ = load_model(args.checkpoint_path, agent)
        print(f"Resuming training from step {total_step}")

    exploration_before_start(args, env_sampler, env_pool, agent)

    for epoch_step in range(args.num_epoch):
        start_step = total_step
        train_policy_steps = 0
        for i in count():
            cur_step = total_step - start_step

            if cur_step >= args.epoch_length and len(env_pool) > args.min_pool_size:  # epoch_length is 1000
                break

            if args.use_algo == 'flowrl':
                if i == 0:
                    train_predict_model(args, env_pool, predict_env, total_step)
                #function1
                train_predict_model_by_couple_flow(args, env_pool, predict_env, agent, rollout_depth, rewarder, model_agent, cur_step, total_step)
                if cur_step > 0 and cur_step % args.model_train_freq == 0 and args.real_ratio < 1.0:
                    #  if cur_step > 0 and cur_step % 500 == 0 and args.real_ratio < 1.0:

                    new_rollout_length = set_rollout_length(epoch_step, rollout_schedule)
                    if rollout_length != new_rollout_length:
                        rollout_length = new_rollout_length
                        model_pool = resize_model_pool(args, rollout_length, model_pool)

                    rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length)
            elif args.use_algo == 'mbpo':
                if cur_step > 0 and cur_step % args.model_train_freq == 0 and args.real_ratio < 1.0:
                    train_predict_model(args, env_pool, predict_env, total_step)

                    new_rollout_length = set_rollout_length(epoch_step, rollout_schedule)
                    if rollout_length != new_rollout_length:
                        rollout_length = new_rollout_length
                        model_pool = resize_model_pool(args, rollout_length, model_pool)

                    rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length)


            cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
            env_pool.push(cur_state, action, reward, next_state, done)

            if len(env_pool) > args.min_pool_size:
                train_policy_steps += train_policy_repeats(args, total_step, train_policy_steps, cur_step, env_pool, model_pool, agent)

            total_step += 1

            if total_step % args.epoch_length == 0:
                '''
                avg_reward_len = min(len(env_sampler.path_rewards), 5)
                avg_reward = sum(env_sampler.path_rewards[-avg_reward_len:]) / avg_reward_len
                logging.info("Step Reward: " + str(total_step) + " " + str(env_sampler.path_rewards[-1]) + " " + str(avg_reward))
                print(total_step, env_sampler.path_rewards[-1], avg_reward)
                '''
                env_sampler.current_state = None
                sum_reward = 0
                done = False
                test_step = 0

                while (not done) and (test_step != args.max_path_length):
                    cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, eval_t=True)
                    sum_reward += reward
                    test_step += 1
                # logger.record_tabular("total_step", total_step)
                # logger.record_tabular("sum_reward", sum_reward)
                # logger.dump_tabular()

                folder_path = f"./results/{args.env_name}"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                if args.use_algo == 'discriminator':
                    file_name = f"{folder_path}/{args.env_name}_discriminator_{now02}.txt"
                elif args.use_algo == 'flowrl':
                    file_name = f"{folder_path}/{args.env_name}_flowRL_{now02}.txt"
                elif args.use_algo == 'mbpo':
                    file_name = f"{folder_path}/{args.env_name}_mbpo_{now02}.txt"

                with open(file_name, "a") as file:
                    file.write(f"{total_step}\t{sum_reward}\n")

                logging.info("Step Reward: " + str(total_step) + " " + str(sum_reward))
                print(total_step, sum_reward)
                #释放内存
                torch.cuda.empty_cache()


def exploration_before_start(args, env_sampler, env_pool, agent):
    for i in range(args.init_exploration_steps):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
        env_pool.push(cur_state, action, reward, next_state, done)


def set_rollout_length(epoch_step, rollout_schedule):
    min_epoch, max_epoch, min_length, max_length = rollout_schedule
    if epoch_step <= min_epoch:
        rollout_length = min_length
    else:
        t = (epoch_step - min_epoch) / (max_epoch - min_epoch)
        t = min(t, 1)
        rollout_length = t * (max_length - min_length) + min_length

    # rollout_length = int(y)
    # rollout_length = (min(max(args.rollout_min_length + (epoch_step - args.rollout_min_epoch)
    #                          / (args.rollout_max_epoch - args.rollout_min_epoch) * (args.rollout_max_length - args.rollout_min_length),
    #                          args.rollout_min_length), args.rollout_max_length))
    return int(rollout_length)


def train_predict_model(args, env_pool, predict_env, total_step):
    # Get all samples from environment
    state, action, reward, next_state, done = env_pool.sample(len(env_pool))
    delta_state = next_state - state
    inputs = np.concatenate((state, action), axis=-1)
    labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)

    predict_env.model.train(args, inputs, labels, total_step, batch_size=256, holdout_ratio=0.2)


def train_couple_flow(args, env_pool, predict_env, agent, rollout_depth, rewarder, total_step):
    batch_size = 1e5
    aux_model_pool = ReplayMemory(batch_size)
    # aux_model_pool.clear()
    # get simulated sample
    rollout_aux_model(args, predict_env, agent, aux_model_pool, env_pool, rollout_depth, 1000, rewarder,total_step)


def train_predict_model_by_couple_flow(args, env_pool, predict_env, agent, rollout_depth, rewarder, model_agent, cur_step, total_step):

    train_couple_flow(args, env_pool, predict_env, agent, rollout_depth, rewarder, total_step)

    loss_list = []
    # simple change model by distributions
    if cur_step % 250 == 0:
        for i in range(200):
            flag = train_model_by_rl(args, env_pool, model_agent, agent, rewarder, loss_list, i)
            if not flag:
                break


def train_model_by_rl(args, env_pool, model_agent, agent, rewarder, loss_list, index):
    batch_size = 256
    state, action, reward, next_state, done = env_pool.sample(batch_size, rewarder=rewarder)
    reward, done = np.squeeze(reward), np.squeeze(done)
    done = (~done).astype(int)
    action_next = agent.select_action(next_state)
    # action_next = agent.select_action(next_state, eval=True)
    # construct S A S'
    S = np.concatenate((state, action), axis=-1)
    A = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), next_state), axis=-1)
    S_next = np.concatenate((next_state, action_next), axis=-1)

    model_agent.update_parameters((S, A, reward, S_next, done), 1)

    if index < 5:
        loss_list.append(reward.mean(0))
    else:
        now_index = index % 5
        loss_list[now_index] = reward.mean(0)
        min_value = min(loss_list)
        min_index = loss_list.index(min_value)
        if min_index == now_index:
            return False

    return True

def resize_model_pool(args, rollout_length, model_pool):
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(rollout_length * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch

    sample_all = model_pool.return_all()
    new_model_pool = ReplayMemory(new_pool_size)
    new_model_pool.push_batch(sample_all)

    return new_model_pool


def rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length):
    state, action, reward, next_state, done = env_pool.sample_all_batch(args.rollout_batch_size)
    for i in range(rollout_length):
        # TODO: Get a batch of actions
        action = agent.select_action(state)
        next_states, rewards, terminals, info = predict_env.step(state, action)
        # TODO: Push a batch of samples
        model_pool.push_batch([(state[j], action[j], rewards[j], next_states[j], terminals[j]) for j in range(state.shape[0])])
        nonterm_mask = ~terminals.squeeze(-1)
        if nonterm_mask.sum() == 0:
            break
        state = next_states[nonterm_mask]

def rollout_aux_model(args, predict_env, agent, model_pool, env_pool, rollout_depth, batch_size, rewarder, total_step):
    state, action, reward, next_state, done = env_pool.sample_all_batch(int(batch_size))
    #reward = np.reshape(reward, (-1, 1))
    #input = torch.Tensor(np.concatenate((state, action, reward), axis=-1)).to(args.device)
    #r = rewarder.get_reward(input, not_rl=True)
    # rollout_depth = 1
    # flow rollout
    for i in range(rollout_depth):
        # TODO: Get a batch of actions
        action = agent.select_action(state)
        next_states, rewards, terminals, info = predict_env.step(state, action, deterministic=False)
        # TODO: Push a batch of samples
        model_pool.push_batch([(state[j], action[j], rewards[j], next_states[j], terminals[j]) for j in range(state.shape[0])])
        # imput flow and struct new reward
        #if total_step % 1000 == 0 and i == 0:
            # 计算当前回合的 rewards 均值
        #    flow_rewards_mean = torch.mean(r).item()
            # 打开文件以追加写入数据
        #    flow_file_name = f"./log/{args.env_name}/{args.env_name}_flow_r_{now02}.txt"
        #    with open(flow_file_name, 'a') as flow_log_file:
        #        flow_log_file.write(f"Epoch {int(total_step / 1000)} Avg Reward: {flow_rewards_mean}\n")
            
        nonterm_mask = ~terminals.squeeze(-1)
        if nonterm_mask.sum() == 0:
            break
        state = next_states[nonterm_mask]

    if total_step % 1000 == 0:
        rewarder.update(env_pool=env_pool, model_pool=model_pool)

    return model_pool


def train_policy_repeats(args, total_step, train_step, cur_step, env_pool, model_pool, agent):
    if total_step % args.train_every_n_steps > 0:
        return 0

    if train_step > args.max_train_repeat_per_step * total_step:
        return 0

    for i in range(args.num_train_repeat):
        env_batch_size = int(args.policy_train_batch_size * args.real_ratio)
        model_batch_size = args.policy_train_batch_size - env_batch_size

        env_state, env_action, env_reward, env_next_state, env_done = env_pool.sample(int(env_batch_size))

        if model_batch_size > 0 and len(model_pool) > 0:
            model_state, model_action, model_reward, model_next_state, model_done = model_pool.sample_all_batch(int(model_batch_size))
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = np.concatenate((env_state, model_state), axis=0), \
                                                                                    np.concatenate((env_action, model_action),
                                                                                                   axis=0), np.concatenate(
                (np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0), \
                                                                                    np.concatenate((env_next_state, model_next_state),
                                                                                                   axis=0), np.concatenate(
                (np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
        else:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = env_state, env_action, env_reward, env_next_state, env_done

        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
        batch_done = (~batch_done).astype(int)
        qf1_loss, qf2_loss, policy_loss, alpha_loss, alpha_tlogs = agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_done), args.policy_train_batch_size, i)
        if total_step % 1000 == 0:
            folder_path = f"./results/{args.env_name}/loss"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            file_name = f"{folder_path}/{args.env_name}_r+_{now02}.txt"
            with open(file_name, 'a') as f:
                f.write(f"Epoch: {total_step / 1000}, qf1_loss: {qf1_loss}, qf2_loss: {qf2_loss}, "
                        f"policy_loss: {policy_loss}, alpha_loss: {alpha_loss}, alpha_tlogs: {alpha_tlogs}\n")
    return args.num_train_repeat


from gym.spaces import Box
from env import register_mbpo_environments


class SingleEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SingleEnvWrapper, self).__init__(env)
        obs_dim = env.observation_space.shape[0]
        obs_dim += 2
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        torso_height, torso_ang = self.env.sim.data.qpos[1:3]  # Need this in the obs for determining when to stop
        obs = np.append(obs, [torso_height, torso_ang])

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        torso_height, torso_ang = self.env.sim.data.qpos[1:3]
        obs = np.append(obs, [torso_height, torso_ang])
        return obs


def main(args=None):
    if args is None:
        args = readParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    torch.set_num_threads(10)
    args.seed = torch.randint(0, 10000, (1,)).item()
    # Initial environment
    register_mbpo_environments()
    env = gym.make(args.env_name)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    print("use cuda:", args.device)

    state_size = np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)
    print('s,a:', state_size, action_size)

    if args.env_name == 'InvertedPendulum-v2':
        args.model_train_freq = 125
        args.epoch_length = 250
        args.num_train_repeat = 30
        args.init_exploration_steps = 300
        # set rollout length
        rollout_schedule = [1, 15, 1, 1]
    elif args.env_name == 'SwimmerTruncatedEnv-v2':
        args.init_exploration_steps = 2000
        rollout_schedule = [1, 20, 1, 1]

    elif args.env_name == 'HalfCheetah-v2':
        args.num_train_repeat = 40
        rollout_schedule = [1, 30, 1, 1]

    elif args.env_name == 'Humanoid-v2':
        args.n_trajs = 3
        args.pred_hidden_size = 400
        state_size = int(45)
        rollout_schedule = [20, 300, 1, 25]

    elif args.env_name == 'HumanoidTruncatedObsEnv-v2':
        args.n_trajs = 3
        args.pred_hidden_size = 400
        state_size = int(45)
        rollout_schedule = [20, 300, 1, 25]

    elif args.env_name == 'Ant-v2':
        args.n_trajs = 3
        rollout_schedule = [20, 100, 1, 1]
        args.flow_depth = 4

    elif args.env_name == 'AntTruncatedObsEnv-v2':
        args.n_trajs = 3
        state_size = int(27)
        rollout_schedule = [20, 100, 1, 1]
    elif args.env_name == 'Walker2d-v2':
        rollout_schedule = [1, 80, 1, 1]
    else:
        rollout_schedule = [20, 100, 1, 15]

    # Intial agent
    agent = SAC(state_size, env.action_space, args)

    # Initial ensemble model
    if args.model_type == 'pytorch':
        env_model = EnsembleDynamicsModel(args, args.num_networks, args.num_elites, state_size, action_size, args.reward_size, args.pred_hidden_size,
                                          use_decay=args.use_decay)

    # Predict environments
    predict_env = PredictEnv(args, env_model, args.env_name, args.model_type)


    model_agent = SAC_MODEL(state_size+action_size, 1+state_size, args, predict_env)

    # Initial pool for env
    env_pool = ReplayMemory(args.replay_size)
    # Initial pool for model
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(1 * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch
    model_pool = ReplayMemory(new_pool_size)

    # Sampler of environment
    env_sampler = EnvSampler(env, max_path_length=args.max_path_length)

    #  model based flow
    flow_type = 'RealNVP'
    flow_args = {"input_size": None, "n_blocks": 1, "hidden_size": 256, "n_hidden": 2, "cond_label_size": None, "batch_norm": True}
    #flow_type = 'MAF'
    #flow_args = {"input_size": None, "n_blocks": 1, "hidden_size": 256, "n_hidden": 2, "cond_label_size": None,"activation": "relu", "input_order": "sequential", "batch_norm": True}

    if args.flow_option == 0:
        flow_args['input_size'] = state_size + 1
    elif args.flow_option == 1:
        flow_args['input_size'] = state_size + action_size + 1

    if args.env_name == 'Ant-v2':
        if args.flow_option == 1:
            temp = 27 + action_size
        elif args.flow_option == 0:
            temp = 27

        flow_args['input_size'] = temp
    elif args.env_name == 'AntTruncatedObsEnv-v2':
        if args.flow_option == 1:
            temp = 27 + action_size + 1
        elif args.flow_option == 0:
            temp = 27

        flow_args['input_size'] = temp
    elif args.env_name == 'Humanoid-v2':
        if args.flow_option == 1:
            temp = 45 + 17 + 1
        elif args.flow_option == 0:
            temp = 45

        flow_args['input_size'] = temp

    elif args.env_name == 'HumanoidTruncatedObsEnv-v2':
        if args.flow_option == 1:
            temp = 45 + 17 + 1
        elif args.flow_option == 0:
            temp = 45

        flow_args['input_size'] = temp

    flow_norm = 'none'
    print("input_size: ", flow_args['input_size'])

    rewarder_args = {'smooth': None, 'use_tanh': True, 'tanh_scale': [1,1],
                     'tanh_shift': True, 'flow_reg': True, 'flow_reg_weight': 1,
                     'rewarder_replay_size': None}

    flow1 = flow.Flow(flow_type=flow_type, flow_args=flow_args, flow_norm=flow_norm, env_name=args.env_name)
    flow2 = flow.Flow(flow_type=flow_type, flow_args=flow_args, flow_norm=flow_norm, env_name=args.env_name)
    couple_flow = CoupledFlow(flow1, flow2, args.lr, device=args.device, option=args.flow_option, env_name=args.env_name, **rewarder_args)

    rewarder = Rewarder(model=couple_flow, update_every=2560, update_iters=10,
                        update_batch_size=256, debug=False)
    # alpha = 0.1
    # sigma_max = 1.0
    # sigma_min = 0.1
    # model = FlowPolicy(alpha, sigma_max, sigma_min, action_sizes, state_sizes, output_sizes, device).to(device)

    train(args, env_sampler, predict_env, agent, env_pool, model_pool, couple_flow, rewarder, model_agent, rollout_schedule)


if __name__ == '__main__':
    main()
