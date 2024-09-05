import torch
import numpy as np
import torch.nn as nn


class PredefinedNoiseSchedule(nn.Module):
    ''' Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules. '''

    def __init__(self, noise_schedule='cosine', timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'poly' in noise_schedule:
            splits = noise_schedule.split('_')
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, power=power)
        else:
            raise ValueError(noise_schedule)

        # print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        # print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = nn.Parameter(torch.from_numpy(-log_alphas2_to_sigmas2).float(), requires_grad=False)

    def forward(self, t_normalized):
        t_int = torch.round(t_normalized * self.timesteps).long()
        return self.gamma[t_int]

    def get_alpha(self, t_normalized):
        '''Computes alpha given gamma.'''
        t_int = torch.round(t_normalized * self.timesteps).long()
        return torch.sqrt(torch.sigmoid(-self.gamma[t_int]))

    def get_sigma(self, t_normalized):
        '''Computes sigma given gamma.'''
        t_int = torch.round(t_normalized * self.timesteps).long()
        return torch.sqrt(torch.sigmoid(self.gamma[t_int]))

    def get_alpha_t_given_s(self, s_normalized, t_normalized):
        ''' alpha t given s = alpha t / alpha s '''
        alpha_t = self.get_alpha(t_normalized=t_normalized)
        alpha_s = self.get_alpha(t_normalized=s_normalized)
        return alpha_t / alpha_s

    def get_sigma_t_given_s(self, s_normalized, t_normalized):
        ''' sigma_t_given_s = sqrt(1 - (alpha_t_given_s) ^2 ) '''
        return torch.sqrt(1 - (self.get_alpha_t_given_s(s_normalized, t_normalized)) ^ 2)


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    '''
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    '''
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    '''
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    '''
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def clip_noise_schedule(alphas2, clip_value=0.001):
    '''
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    '''
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def sample_gaussian_with_mask(size, mask=None):
    x = torch.randn(size, device=mask.device)
    x_masked = x * mask
    return x_masked
