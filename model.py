from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.distributions import Independent, Normal
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.h = 128

        self.policy_model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.h, self.h),
            nn.ReLU(),
            nn.Linear(self.h, action_dim)
        )

        self.value_model = nn.Sequential(
            nn.Linear(state_dim, self.h),
            nn.ReLU(),
            nn.Linear(self.h, self.h),
            nn.ReLU(),
            nn.Linear(self.h, 1)
        )

        self.action_dim = action_dim



    def get_policy(self, x):
        
        out = self.policy_model(x)

        return out

    def get_value(self, x):
        out = self.value_model(x)
        return out

    def forward(self, x):
        policy = self.get_policy(x)
        value = self.get_value(x)

        return policy, value
    

class Policy:
    def __init__(self, model):
        self.model = model

    def act(self, inputs, training=False):

        inputs = torch.tensor(inputs, dtype=torch.float32, device=device)
        
        logits = self.model.get_policy(inputs)

        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        values = self.model.get_value(inputs)

        if training:
            return {"distribution": dist, "values": values.squeeze()}
        return {
            "actions": actions.detach().cpu(),
            "log_probs": log_probs.detach().cpu().squeeze(-1),
            "values": values.detach().cpu().squeeze(-1),
        }