#!/usr/bin/env python3
"""
Automated experiments for Dynamic Human-AI Co-Adaptation (CartPole-v1 domain).
Methods: baseline DQN and hybrid DQN + behavioral cloning (BC).
"""
import os
import json
import random
import argparse
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import matplotlib.pyplot as plt

# Replay buffer for RL
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

def collect_expert_data(env, n_episodes, save_path):
    data = []
    for ep in range(n_episodes):
        # reset environment (obs, info)
        out = env.reset()
        state = out[0] if isinstance(out, tuple) else out
        done = False
        while not done:
            # heuristic policy: push pole toward upright
            angle = state[2]
            action = 0 if angle < 0 else 1
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            data.append((state, action))
            state = next_state
    states, actions = zip(*data)
    np.savez(save_path, states=np.array(states), actions=np.array(actions))
    print(f"Saved expert data to {save_path}")

def train(method, env, device, expert_data, args):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = PolicyNet(state_dim, action_dim).to(device)
    target = PolicyNet(state_dim, action_dim).to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.replay_size)
    # expert data
    exp_states = torch.tensor(expert_data['states'], dtype=torch.float32).to(device)
    exp_actions = torch.tensor(expert_data['actions'], dtype=torch.long).to(device)
    # metrics
    rewards = []
    losses = []
    eps = args.eps_start
    total_steps = 0
    # initialize state (obs, info)
    out = env.reset()
    state = out[0] if isinstance(out, tuple) else out
    ep_reward = 0
    for step in range(1, args.max_steps + 1):
        total_steps += 1
        # epsilon-greedy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                qs = policy(torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0))
                action = qs.argmax().item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay.push(state, action, reward, next_state, done)
        state = next_state
        ep_reward += reward
        # decay eps
        eps = max(args.eps_end, eps * args.eps_decay)
        # training
        if len(replay) >= args.batch_size:
            trans = replay.sample(args.batch_size)
            s = torch.tensor(trans.state, dtype=torch.float32).to(device)
            a = torch.tensor(trans.action, dtype=torch.long).to(device)
            r = torch.tensor(trans.reward, dtype=torch.float32).to(device)
            ns = torch.tensor(trans.next_state, dtype=torch.float32).to(device)
            d = torch.tensor(trans.done, dtype=torch.float32).to(device)
            # Q targets
            with torch.no_grad():
                target_q = r + args.gamma * (1 - d) * target(ns).max(1)[0]
            q_vals = policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss_q = nn.functional.mse_loss(q_vals, target_q)
            loss = loss_q
            # BC loss for hybrid
            if method == 'hybrid':
                idx = torch.randperm(exp_states.size(0))[:args.bc_batch]
                bs = exp_states[idx]
                ba = exp_actions[idx]
                logits = policy(bs)
                loss_bc = nn.functional.cross_entropy(logits, ba)
                loss = loss_q + args.bc_coef * loss_bc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        # update target
        if step % args.target_update == 0:
            target.load_state_dict(policy.state_dict())
        # record
        if done:
            rewards.append(ep_reward)
            ep_reward = 0
            # reset environment (obs, info)
            out = env.reset()
            state = out[0] if isinstance(out, tuple) else out
        # logging intermediate
        if step % args.log_interval == 0:
            avg_r = np.mean(rewards[-10:]) if rewards else 0
            print(f"{method} step {step}/{args.max_steps} avg_reward {avg_r:.2f}")
        # stop at max_steps
    return policy, rewards, losses

def evaluate(policy, env, device, episodes=20):
    policy.eval()
    rewards = []
    with torch.no_grad():
        for _ in range(episodes):
            # reset environment (obs, info)
            out = env.reset()
            state = out[0] if isinstance(out, tuple) else out
            ep_r = 0
            done = False
            while not done:
                action = policy(torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)).argmax().item()
                next_state, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = next_state
                ep_r += r
            rewards.append(ep_r)
    return np.mean(rewards)

def plot_curves(curves, args):
    os.makedirs(args.out_dir, exist_ok=True)
    for name, data in curves.items():
        plt.figure()
        plt.plot(data)
        plt.title(f"{name} curve")
        plt.xlabel('Episode')
        plt.ylabel('Reward' if 'reward' in name else 'Loss')
        plt.savefig(os.path.join(args.out_dir, f"{name}.png"))
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=20000)
    parser.add_argument('--replay_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_end', type=float, default=0.01)
    parser.add_argument('--eps_decay', type=float, default=0.995)
    parser.add_argument('--target_update', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=2000)
    parser.add_argument('--bc_coef', type=float, default=1.0)
    parser.add_argument('--bc_batch', type=int, default=64)
    parser.add_argument('--expert_episodes', type=int, default=50)
    parser.add_argument('--out_dir', type=str, default='results')
    parser.add_argument('--log_file', type=str, default='log.txt')
    args = parser.parse_args()
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make('CartPole-v1')
    # prepare output
    os.makedirs(args.out_dir, exist_ok=True)
    # expert data
    expert_path = 'expert_data.npz'
    if not os.path.exists(expert_path):
        print('Collecting expert data...')
        # create directory if specified
        dirpath = os.path.dirname(expert_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        collect_expert_data(env, args.expert_episodes, expert_path)
    expert_data = np.load(expert_path)
    # run methods
    all_results = {}
    curves = {}
    for method in ['baseline', 'hybrid']:
        print(f"Training {method}...")
        policy, rewards, losses = train(method, env, device, expert_data, args)
        mean_eval = evaluate(policy, env, device)
        all_results[method] = {'final_eval_reward': float(mean_eval), 'train_rewards': rewards}
        curves[f'{method}_reward'] = rewards
        curves[f'{method}_loss'] = losses
    # save results
    with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    plot_curves(curves, args)
    # log
    with open(args.log_file, 'w') as f:
        f.write(json.dumps(all_results, indent=2))
    print('Done. Results are in', args.out_dir)

if __name__ == '__main__':
    main()
