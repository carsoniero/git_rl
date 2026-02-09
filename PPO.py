import torch
import numpy as np

device = torch.device("cpu")


class PPO:
    def __init__(
        self,
        policy,
        optimizer,
        cliprange=0.2,
        value_loss_coef=0.25,
        max_grad_norm=0.5,
        entropy_coef=0.01,
        action_std=0.1,
    ):
        self.policy = policy
        self.optimizer = optimizer
        self.cliprange = cliprange
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.action_std = action_std

    # -----------------------------
    # Policy loss для continuous action
    # -----------------------------
    def policy_loss(self, trajectory):
        # Входы
        obs_map = torch.tensor(trajectory["obs_map"], dtype=torch.float32, device=device)
        state   = torch.tensor(trajectory["state"], dtype=torch.float32, device=device)
        cmd     = torch.tensor(trajectory["cmd"], dtype=torch.float32, device=device)
        actions = torch.tensor(trajectory["actions"], dtype=torch.float32, device=device)
        old_log_probs = torch.tensor(trajectory["log_probs"], dtype=torch.float32, device=device)

        # Прямой проход через сеть
        act_output = self.policy.act(obs_map, state, cmd, training=True)
        distr = act_output["distribution"]

        # Логарифм вероятности выбранных действий
        new_log_probs = distr.log_prob(actions).sum(dim=-1)

        # Показатель отношения вероятностей (PPO ratio)
        ratio = (new_log_probs - old_log_probs).exp()

        # Advantages
        advantages = torch.tensor(trajectory["advantages"], dtype=torch.float32, device=device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO clipped objective
        J = ratio * advantages
        J_clipped = torch.clamp(ratio, 1 - self.cliprange, 1 + self.cliprange) * advantages

        # Энтропия для exploration
        entropy = distr.entropy().sum(dim=-1).mean()

        policy_loss = -torch.mean(torch.min(J, J_clipped))
        total_policy_loss = policy_loss - self.entropy_coef * entropy

        return total_policy_loss

    # -----------------------------
    # Value loss (critic)
    # -----------------------------
    def value_loss(self, trajectory):
        obs_map = torch.tensor(trajectory["obs_map"], dtype=torch.float32, device=device)
        state   = torch.tensor(trajectory["state"], dtype=torch.float32, device=device)
        cmd     = torch.tensor(trajectory["cmd"], dtype=torch.float32, device=device)
        targets = torch.tensor(trajectory["value_targets"], dtype=torch.float32, device=device)

        act_output = self.policy.act(obs_map, state, cmd, training=True)
        values = act_output["values"]

        values_old = torch.tensor(trajectory["values"], dtype=torch.float32, device=device)

        # Clipped value loss
        l_simple = (values - targets.detach()) ** 2
        l_clipped = (values_old + torch.clamp(values - values_old, -self.cliprange, self.cliprange) - targets) ** 2

        loss = torch.mean(torch.max(l_simple, l_clipped.detach()))
        return loss

    # -----------------------------
    # Общая loss
    # -----------------------------
    def loss(self, trajectory):
        p_loss = self.policy_loss(trajectory)
        v_loss = self.value_loss(trajectory)
        total_loss = p_loss + self.value_loss_coef * v_loss
        return total_loss

    # -----------------------------
    # Шаг оптимизации
    # -----------------------------
    def step(self, trajectory):
        loss_final = self.loss(trajectory)

        self.optimizer.zero_grad()
        loss_final.backward()
        self.optimizer.step()
