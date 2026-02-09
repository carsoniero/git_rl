from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.distributions import Independent, Normal
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyModel(nn.Module):
    def __init__(self, action_dim=2, state_dim=12, cmd_dim=2):
        super().__init__()

        # -----------------------------
        # CNN для карты препятствий 36x36
        # -----------------------------
        # Вход: (batch, 1, 36, 36)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 16x18x18
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 32x9x9
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), # 32x5x5
            nn.ReLU()
        )

        self.cnn_out_dim = 32 * 5 * 5  # = 800

        # -----------------------------
        # MLP для состояния квадрокоптера
        # -----------------------------
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # -----------------------------
        # MLP для целевой команды C(t)
        # -----------------------------
        self.cmd_encoder = nn.Sequential(
            nn.Linear(cmd_dim, 32),
            nn.ReLU()
        )

        # -----------------------------
        # Общий MLP после объединения
        # -----------------------------
        self.fc = nn.Sequential(
            nn.Linear(self.cnn_out_dim + 32 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # -----------------------------
        # PPO heads
        # -----------------------------
        self.policy_head = nn.Linear(128, action_dim)  # 2 числа → ax, ay
        self.value_head = nn.Linear(128, 1)           # value function


    def get_policy(self, x):
        out = self.policy_model(x)
        return out
            
    def get_value(self, x):
        out = self.value_model(x)
        return out
    

    def forward(self, obs_map, state, cmd):
        """
        obs_map: (batch, 36, 36)
        state:   (batch, 12)
        cmd:     (batch, 2)
        """
        # Добавляем канал для CNN
        x = obs_map.unsqueeze(1)  # (B,1,36,36)
        x = self.cnn(x)
        x = x.reshape(x.size(0), -1)  # flatten → (B,800)

        # Кодирование состояния и команды
        s = self.state_encoder(state)  # (B,32)
        c = self.cmd_encoder(cmd)      # (B,32)

        # Объединяем все признаки
        z = torch.cat([x, s, c], dim=1)  # (B, 800+32+32=864)
        z = self.fc(z)                    # (B,128)

        # Выходы
        policy = self.get_policy(x)
        value = self.get_value(x)

        return policy, value


import torch
from torch.distributions import Normal

class Policy:
    def __init__(self, model, action_std=0.1, device="cpu"):
        self.model = model
        self.action_std = action_std
        self.device = device

    def act(self, obs_map, state, cmd, training=False):
        # Переводим все входы в тензор
        obs_map = torch.tensor(obs_map, dtype=torch.float32, device=self.device).unsqueeze(0)
        state   = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        cmd     = torch.tensor(cmd, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Прямой проход сети
        policy, value = self.model(obs_map, state, cmd)  # policy: (1,2), value: (1,1)

        # создаём нормальное распределение для непрерывных действий
        dist = Normal(policy, self.action_std)

        if training:
            actions = dist.rsample()  # для backprop через reparameterization trick
            log_probs = dist.log_prob(actions).sum(dim=-1)
            return {"distribution": dist, "values": value.squeeze()}
        
        else:
            actions = policy  # можно брать детерминированное действие
            log_probs = dist.log_prob(actions).sum(dim=-1)
            return {
                "actions": actions.detach().cpu().squeeze(0),
                "log_probs": log_probs.detach().cpu().squeeze(0),
                "values": value.detach().cpu().squeeze(0)
            }

    

# class Policy:
#     def __init__(self, model):
#         self.model = model

#     def act(self, inputs, training=False):

#         inputs = torch.tensor(inputs, dtype=torch.float32, device=device)
        
#         logits = self.model.get_policy(inputs)

#         dist = Categorical(logits=logits)
#         actions = dist.sample()
#         log_probs = dist.log_prob(actions)

#         values = self.model.get_value(inputs)

#         if training:
#             return {"distribution": dist, "values": values.squeeze()}
#         return {
#             "actions": actions.detach().cpu(),
#             "log_probs": log_probs.detach().cpu().squeeze(-1),
#             "values": values.detach().cpu().squeeze(-1),
#         }