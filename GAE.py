import torch
from collections import defaultdict
import numpy as np


def collect_trajectory(env, policy, rollout_length):
    trajectory = defaultdict(list, {"actions": []})
    observations = []
    rewards = []
    resets = []
    obs,_ = env.reset()

    for i in range(rollout_length):
        global counter
        observations.append(obs)
        act = policy.act(obs)
        for key, val in act.items():
            trajectory[key].append(val)
        last_action = trajectory["actions"][-1]

        obs, rew, terminated, truncated, _ = env.step(
            last_action.item()
        )
        done = np.logical_or(terminated, truncated)
        rewards.append(rew)
        resets.append(done)
        if terminated:
            obs,_ = env.reset()
            break
            
            
        
        # trajectory["state"]["latest_observation"] = obs
    trajectory.update(observations=observations, rewards=rewards, resets=resets)
    return trajectory 

device = torch.device("cpu")

class GAE:
    """Generalized Advantage Estimator."""

    def __init__(self, policy, gamma=0.99, lambda_=0.95):
        self.policy = policy
        self.gamma = gamma
        self.lambda_ = lambda_

    def __call__(self, trajectory):
        gamma = self.gamma
        lambda_ = self.lambda_

        rewards = trajectory["rewards"]
        values = trajectory["values"]
        terminal = trajectory["resets"]

        last_obs = torch.from_numpy(trajectory["observations"][-1]).float().to(device)
        last_value = self.policy.model.get_value(last_obs).item()

        T = len(rewards)
        advantages = [0] * T
        value_targets = [0] * T
        delta, gae = 0, 0

        for i in range(T-1, -1,-1):

          if i == T - 1:  # Last step
              next_value = last_value
              next_non_terminal = 1.0 
          else:
              next_value = values[i + 1]
              next_non_terminal = 1.0 - float(terminal[i + 1]) 

          delta = rewards[i] + gamma *  next_value * next_non_terminal - values[i]
          gae = delta + gamma * lambda_ * next_non_terminal * gae

          advantages[i] = gae
          value_targets[i] = gae + values[i]
        trajectory["advantages"]=torch.tensor(advantages)
        trajectory["value_targets"]=torch.tensor(value_targets)

        return trajectory