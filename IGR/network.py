import numpy as np
import torch.nn as nn
import torch
from torch.autograd import grad
import torch.nn.functional as F
from general import *

def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0][:, -2:]
    return points_grad


class ImplicitNet(nn.Module):
    def __init__(
        self,
        d_in,
        dims,
        skip_in=(),
        geometric_init=True,
        radius_init=1,
        beta=100
    ):
        super().__init__()

        dims = [d_in] + dims + [1]

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)

            # if true preform preform geometric initialization
            if geometric_init:

                if layer == self.num_layers - 2:

                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)

                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, "lin" + str(layer), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)

        # vanilla relu
        else:
            self.activation = nn.ReLU()

    def forward(self, input):

        x = input

        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)

            if layer < self.num_layers - 2:
            # if layer < self.num_layers - 3:
                x = self.activation(x)

            # else:
            #     x = torch.tanh(x)

                ### Works bad
                # x = (torch.sigmoid(x) - 0.5) * 2.2
                
                # x = torch.clamp(x, -2.0, 2.0)

        return x

### PointNet Encoder
class STN3D(nn.Module):
    def __init__(self, input_channels=3):
        super(STN3D, self).__init__()
        self.input_channels = input_channels
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_channels * input_channels),
        )

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = self.mlp1(x)
        x = F.max_pool1d(x, num_points).squeeze(2)
        x = self.mlp2(x)
        I = torch.eye(self.input_channels).view(-1).to(x.device)
        x = x + I
        x = x.view(-1, self.input_channels, self.input_channels)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, embedding_size, input_channels=2, with_normals=False):
        super(PointNetEncoder, self).__init__()
        if not with_normals:
            self.input_channels = input_channels
        else:
            self.input_channels = input_channels * 2
        # self.stn1 = STN3D(input_channels)
        # self.stn2 = STN3D(64)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(self.input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.fc = nn.Linear(1024, embedding_size)

    def forward(self, x):
        batch_size = x.shape[0]
        num_points = x.shape[1]
        x = x[:, :, : self.input_channels]
        x = x.transpose(2, 1)  # transpose to apply 1D convolution
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = F.max_pool1d(x, num_points).squeeze(2)  # max pooling
        x = self.fc(x)

        x = F.normalize(x)

        return x

def get_learning_rate_schedules(schedule_specs):

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules

def add_latent(points, latent_codes):
    batch_size, num_of_points, dim = points.shape
    points = points.reshape(batch_size * num_of_points, dim)
    latent_codes = latent_codes.unsqueeze(1).repeat(1,num_of_points,1).reshape(batch_size * num_of_points, -1)
    out = torch.cat([latent_codes, points], 1)
    
    return out
