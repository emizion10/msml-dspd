import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class DiffusionEmbedding(nn.Module):
    ## dim=16, proj_dim=64
    def __init__(self, dim, proj_dim, max_steps=500):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(dim, max_steps), persistent=False
        )
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, dim, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(dim).unsqueeze(0)  # [1,dim]
        table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ResidualBlock(nn.Module):
    ## hidden_size =64, residual_channel=8
    def __init__(self, hidden_size, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            3, 
            padding=1,
            dilation=dilation,
            padding_mode="circular",
        )
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)
        self.conditioner_projection = nn.Conv1d(
            1, 2 * residual_channels, 1, padding=1, padding_mode="circular"
        ) 
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)
    ## x([1920, 8, 3]), conditioner([1920, 1, 1]), diffusion_step ([1920, 64])
    def forward(self, x, conditioner, diffusion_step):
        ## diffusion_step ([1920, 64]) => diffusion_step ([1920, 8,1])
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        ## conditioner([1920, 1, 1])=> conditioner([1920, 16, 3]) 
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step
        # y[1920,16,3]
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        y = F.leaky_relu(y, 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class CondUpsampler(nn.Module):
    def __init__(self, cond_length, target_dim):
        super().__init__()
        ## Replacing target_dim // 2 with 1 for univariate datasets
        self.linear1 = nn.Linear(cond_length, target_dim // 2 or 1)
        self.linear2 = nn.Linear(target_dim // 2 or 1, target_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.linear2(x)
        x = F.leaky_relu(x, 0.4)
        return x


class EpsilonTheta(nn.Module):
    def __init__(
        self,
        target_dim,
        cond_length,
        time_emb_dim=16,
        residual_layers=8,
        residual_channels=8,
        dilation_cycle_length=2,
        residual_hidden=64,
    ):
        super().__init__()
        self.input_projection = nn.Conv1d(
            1, residual_channels, 1, padding=1, padding_mode="circular"
        ) 
        self.diffusion_embedding = DiffusionEmbedding(
            time_emb_dim, proj_dim=residual_hidden
        )
        self.cond_upsampler = CondUpsampler(
            target_dim=target_dim, cond_length=cond_length
        )
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels=residual_channels,
                    dilation=2 ** (i % dilation_cycle_length),
                    hidden_size=residual_hidden,
                )
                for i in range(residual_layers)
            ]
        )
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 2) 
        self.output_projection = nn.Conv1d(residual_channels, 1, 2) 
        # self.output_projection = nn.Conv1d(residual_channels, 1, 3,padding=1) 

        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)
    ## inputs(bs*pl,1,f), time(bs*pl), cond(bs*pl,1,hidd_dim)
    def forward(self, inputs, time, cond):
        UNetmodel = UNet1D(1, 1)
        ## cond(bs*pl,1,hidd_dim) => cond(bs*pl,1,f)
        cond_up = self.cond_upsampler(cond)
        cond_up = cond_up.reshape(-1,1,cond_up.shape[1])

        time = time.reshape(-1,1,time.shape[1])
        diffusion_step = self.diffusion_embedding(time)
        output = UNetmodel(inputs,cond_up, diffusion_step)
        return output

        # x = self.input_projection(inputs)
        # x = F.leaky_relu(x, 0.4)
        ## diffusion_step(bs*pl,1)=>diffusion_step(bs*pl,1,bs)
      

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_up, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        ## x(1920,8,3)=>x(1920,8,1) [x(bs*pl,res_lay,lout from inp_proj)=>x(bs*pl,res_lay,lout from skip_pro)]
        x = self.skip_projection(x)
        x = F.leaky_relu(x, 0.4)
        x = self.output_projection(x)
        return x


class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet1D, self).__init__()

        self.diffusion_projection = nn.Linear(64, 1)

        self.e11 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1) 
        self.e12 = nn.Conv1d(64, 64, kernel_size=3, padding=1) 
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2) 

        self.diff_proj_e1 = nn.Conv1d(1,64,kernel_size=2,stride=2)
        self.cond_up_e1 = nn.Conv1d(1,64,kernel_size=2,stride=2)

        self.e21 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.diff_proj_e2 = nn.Conv1d(1, 128, kernel_size=4, stride=4)
        self.cond_up_e2 = nn.Conv1d(1, 128, kernel_size=4, stride=4)

        self.e31 = nn.Conv1d(128, 256, kernel_size=3, padding=1) 
        self.e32 = nn.Conv1d(256, 256, kernel_size=3, padding=1) 

        self.upconv1 = nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2)
        self.d11 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.d12 = nn.Conv1d(128, 128, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.d21 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.d22 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.relu= nn.ReLU()

        self.outconv = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x,cond_up, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step.squeeze(1))
        diffusion_step = diffusion_step.reshape(-1,1,diffusion_step.shape[1])

         ## [64,1,30]
        x = x + diffusion_step +  cond_up
        ## [64,64,30]
        xe11 = self.relu(self.e11(x))
        xe12 = self.relu(self.e12(xe11))
        ## [64,64,15]
        xp1 = self.pool1(xe12)

        ## Transforming time embedding & conditioner to required dimension after encoder 1
        diff_e1 = self.diff_proj_e1(diffusion_step)
        cond_up_e1 = self.cond_up_e1(cond_up)
        xp1 = xp1 + diff_e1 + cond_up_e1

        ## [64,128,15]
        xe21 = self.relu(self.e21(xp1))
        xe22 = self.relu(self.e22(xe21))
        ## [64,128,7]
        xp2 = self.pool2(xe22)

        ## Transforming time embedding & conditioner to required dimension after encoder 2
        diff_e2 = self.diff_proj_e2(diffusion_step)
        cond_up_e2 = self.cond_up_e2(cond_up)
        xp2 = xp2 + diff_e2 + cond_up_e2

        ## [64,256,7]
        xe31 = self.relu(self.e31(xp2))
        ## [64,256,7]
        xe32 = self.relu(self.e32(xe31))

        ## [64,128,15]
        xu1 = self.upconv1(xe32)
        ## [64,256,15]
        xu11 = torch.cat([xu1, xe22], dim=1)
        xd11 = self.relu(self.d11(xu11))
        ## [64,128,15]
        xd12 = self.relu(self.d12(xd11))

        ## [64,64,30]
        xu2 = self.upconv2(xd12)
        ## [64,128,30]
        xu22 = torch.cat([xu2, xe12], dim=1)
        xd21 = self.relu(self.d21(xu22))
        ## [64,64,30]
        xd22 = self.relu(self.d22(xd21))

        x = self.outconv(xd22)
        ## [64,1,30]
        return x