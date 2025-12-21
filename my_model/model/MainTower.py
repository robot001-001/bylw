# MainTower MLP for HSTU
import torch
from torch import nn

from typing import List


class MainTowerMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        units: List[int]
    ):
        super().__init__()
        self.mlp = nn.Sequential()
        i = -1
        for i in range(len(units)-1):
            self.mlp.add_module(f'linear_{i}', nn.Linear(in_features=in_features, out_features=units[i]))
            self.mlp.add_module(f'relu_{i}', nn.ReLU())
            in_features = units[i]

        self.mlp.add_module(f'linear_{i+1}', nn.Linear(in_features=in_features, out_features=units[i+1]))
        # self.mlp.add_module(f'softmax_{i+1}', nn.Softmax(dim=-1))

    def forward(self, x):
        return self.mlp(x)
    

if __name__ == '__main__':
    model = MainTowerMLP(in_features=128, units=[64, 32, 5])
    print(model)