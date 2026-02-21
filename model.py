import torch.nn as nn
import torch
import math
from unet import Unet
from tqdm import tqdm

class MNISTDiffusion(nn.Module):
    def __init__(self, image_size, in_channels, time_embedding_dim=256, timesteps=1000, base_dim=32, dim_mults=[1, 2, 4, 8]):
        super().__init__()
        self.timesteps = timesteps
        self.in_channels = in_channels
        
        # 튜플 형태의 H, W 지원 (예: (208, 112))
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size

        betas = self._cosine_variance_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=-1)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))

        # out_channels를 in_channels와 동일하게 유지
        self.model = Unet(timesteps, time_embedding_dim, in_channels, in_channels, base_dim, dim_mults)

    def forward(self, x, noise):
        t = torch.randint(0, self.timesteps, (x.shape[0],)).to(x.device)
        x_t = self._forward_diffusion(x, t, noise)
        pred_noise = self.model(x_t, t)
        return pred_noise

    @torch.no_grad()
    def sampling(self, n_samples, clipped_reverse_diffusion=True, device="cuda"):
        # 직사각형 크기 적용
        x_t = torch.randn((n_samples, self.in_channels, self.image_size[0], self.image_size[1])).to(device)
        
        for i in tqdm(range(self.timesteps-1, -1, -1), desc="Sampling"):
            noise = torch.randn_like(x_t).to(device)
            t = torch.tensor([i for _ in range(n_samples)]).to(device)

            if clipped_reverse_diffusion:
                x_t = self._reverse_diffusion_with_clip(x_t, t, noise)
            else:
                x_t = self._reverse_diffusion(x_t, t, noise)

        x_t = (x_t + 1.) / 2.
        return x_t
    
    def _cosine_variance_schedule(self, timesteps, epsilon=0.008):
        steps = torch.linspace(0, timesteps, steps=timesteps+1, dtype=torch.float32)
        f_t = torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas = torch.clip(1.0-f_t[1:]/f_t[:timesteps], 0.0, 0.999)
        return betas

    def _forward_diffusion(self, x_0, t, noise):
        assert x_0.shape == noise.shape
        return self.sqrt_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1, 1) * x_0 + \
               self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1, 1) * noise

    @torch.no_grad()
    def _reverse_diffusion(self, x_t, t, noise): 
        pred = self.model(x_t, t)

        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        mean = (1. / torch.sqrt(alpha_t)) * (x_t - ((1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred)

        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t-1).reshape(x_t.shape[0], 1, 1, 1)
            std = torch.sqrt(beta_t * (1. - alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))
        else:
            std = 0.0

        return mean + std * noise 

    @torch.no_grad()
    def _reverse_diffusion_with_clip(self, x_t, t, noise): 
        pred = self.model(x_t, t)
        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        
        x_0_pred = torch.sqrt(1. / alpha_t_cumprod) * x_t - torch.sqrt(1. / alpha_t_cumprod - 1.) * pred
        x_0_pred.clamp_(-1., 1.)

        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t-1).reshape(x_t.shape[0], 1, 1, 1)
            mean = (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod)) * x_0_pred + \
                 ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod)) * x_t

            std = torch.sqrt(beta_t * (1. - alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))
        else:
            mean = (beta_t / (1. - alpha_t_cumprod)) * x_0_pred 
            std = 0.0

        return mean + std * noise