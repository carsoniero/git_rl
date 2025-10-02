import torch
import numpy as np


device = torch.device("cpu")

class PPO:
    def __init__(
        self, policy, optimizer, cliprange=0.2, value_loss_coef=0.25, max_grad_norm=0.5, entropy_coef = 0.01
    ):
        self.policy = policy
        self.optimizer = optimizer
        self.cliprange = cliprange
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
    

    def policy_loss(self, trajectory, act):

        states = torch.tensor(trajectory["observations"], dtype=torch.float32, device=device)
        
        actions = torch.tensor(trajectory["actions"], dtype=torch.long, device=device)

        act_output = act(states, training=True)
        distr = act_output["distribution"]

        new_log_probs = distr.log_prob(actions)

        old_log_probs = torch.as_tensor(trajectory['log_probs'], dtype=torch.float32, device=device)

        ratio = (new_log_probs - old_log_probs).exp()


        advantages = torch.tensor(trajectory["advantages"], dtype=torch.float32, device=device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        J = ratio * advantages
        J_clipped = torch.clip(ratio, 1 - self.cliprange, 1 + self.cliprange) * advantages

        entropy = distr.entropy().mean()

        policy_loss = -torch.mean(torch.min(J, J_clipped))

        total_policy_loss = policy_loss - entropy * self.entropy_coef

        return total_policy_loss

    def value_loss(self, trajectory, act):

        states = torch.tensor(trajectory["observations"], dtype=torch.float32, device=device)
        targets = torch.tensor(trajectory["value_targets"], dtype=torch.float32, device=device)

        act_output = act(states, training=True)
        values = act_output["values"]

        l_simple = (values - targets.detach()) ** 2

        values_old = torch.tensor(trajectory["values"], dtype=torch.float32, device=device)

        l_clipped = (
            values_old + torch.clip(values - values_old, -self.cliprange, self.cliprange) - targets
        ) ** 2

        loss = torch.mean(torch.max(l_simple, l_clipped.detach()))

        return loss 

    def loss(self, trajectory):
        p_loss = self.policy_loss(trajectory, self.policy.act)
        v_loss = self.value_loss(trajectory, self.policy.act)
        total_loss = p_loss + self.value_loss_coef * v_loss
        return total_loss

    def step(self, trajectory):
        loss_final = self.loss(trajectory)

        self.optimizer.zero_grad()
        loss_final.backward()
        
        self.optimizer.step()
