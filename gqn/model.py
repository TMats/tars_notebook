import random
import torch
from torch import nn
from torch.nn import functional as F

from Tars.distributions import Normal
from Tars.losses.divergences import KullbackLeibler

from conv_lstm import Conv2dLSTMCell
from representation import Representation


# cores
class GeneratorCore(nn.Module):
    def __init__(self, v_dim, r_dim, z_dim, h_dim, SCALE):
        super(GeneratorCore, self).__init__()
        self.core = Conv2dLSTMCell(v_dim + r_dim + z_dim, h_dim, kernel_size=5, stride=1, padding=2)
        self.upsample = nn.ConvTranspose2d(h_dim, h_dim, kernel_size=SCALE, stride=SCALE, padding=0)

    def forward(self, z, v, r, h_g, c_g, u):
        h_g, c_g =  self.core(torch.cat([z, v, r], dim=1), [h_g, c_g])
        u = self.upsample(h_g) + u
        return h_g, c_g, u


class InferenceCore(nn.Module):
    def __init__(self, x_dim, v_dim, r_dim, h_dim):
        super(InferenceCore, self).__init__()
        self.core = Conv2dLSTMCell(h_dim + x_dim + v_dim + r_dim, h_dim, kernel_size=5, stride=1, padding=2)

    def forward(self, x, v, r, h_g, h_e, c_e):
        h_e, c_e = self.core(torch.cat([h_g, x, v, r], dim=1), [h_e, c_e])
        return h_e, c_e


# distributions
class Generator(Normal):
    def __init__(self, x_dim, h_dim):
        super(Generator, self).__init__(cond_var=["u", "sigma"],var=["x_q"])
        self.eta_g = nn.Conv2d(h_dim, x_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, u, sigma):
        mu = self.eta_g(u)
        return {"loc":mu, "scale":sigma}


class Prior(Normal):
    def __init__(self, z_dim, h_dim):
        super(Prior, self).__init__(cond_var=["h_g"],var=["z"])
        self.z_dim = z_dim
        self.eta_pi_mu = nn.Conv2d(h_dim, 2*z_dim, kernel_size=5, stride=1, padding=2)
        self.eta_pi_std = nn.Conv2d(h_dim, 2*z_dim, kernel_size=5, stride=1, padding=2)

    def forward(self, h_g):
        mu = self.eta_pi_mu(h_g)
        std = torch.exp(0.5*self.eta_pi_std(h_g))
        return {"loc":mu ,"scale":std}


class Inference(Normal):
    def __init__(self, z_dim, h_dim):
        super(Inference, self).__init__(cond_var=["h_i"],var=["z"])
        self.z_dim = z_dim
        self.eta_e_mu = nn.Conv2d(h_dim, 2*z_dim, kernel_size=5, stride=1, padding=2)
        self.eta_e_std = nn.Conv2d(h_dim, 2*z_dim, kernel_size=5, stride=1, padding=2)
        
    def forward(self, h_i):
        mu = self.eta_e_mu(h_i)
        std = torch.exp(0.5*self.eta_e_std(h_i))
        return {"loc":mu, "scale":std}


class GQN(nn.Module):
    def __init__(self, x_dim, v_dim, r_dim, h_dim, z_dim, L, SCALE):
        super(GQN, self).__init__()
        self.L = L
        self.h_dim = h_dim
        self.SCALE = SCALE

        self.representation = Representation(x_dim, v_dim, r_dim)
        self.generator_core = GeneratorCore(v_dim, r_dim, z_dim, h_dim, self.SCALE)
        self.inference_core = InferenceCore(x_dim, v_dim, r_dim, h_dim)

        self.upsample   = nn.ConvTranspose2d(h_dim, h_dim, kernel_size=SCALE, stride=SCALE, padding=0)
        self.downsample = nn.Conv2d(x_dim, x_dim, kernel_size=SCALE, stride=SCALE, padding=0)

        # distribution
        self.pi = Prior(z_dim, h_dim)
        self.q = Inference(z_dim, h_dim)
        self.g = Generator(x_dim, h_dim)

    def forward(self, images, viewpoints, sigma):
        # Number of context datapoints to use for representation
        batch_size, m, *_ = viewpoints.size()

        # Sample random number of views and generate representation
        n_views = random.randint(2, m-1)

        indices = torch.randperm(m)
        representation_idx, query_idx = indices[:n_views], indices[n_views]

        x, v = images[:, representation_idx], viewpoints[:, representation_idx]

        # Merge batch and view dimensions.
        _, _, *x_dims = x.size()
        _, _, *v_dims = v.size()

        x = x.view((-1, *x_dims))
        v = v.view((-1, *v_dims))

        # representation generated from input images
        # and corresponding viewpoints
        phi = self.representation(x, v)

        # Seperate batch and view dimensions
        _, *phi_dims = phi.size()
        phi = phi.view((batch_size, n_views, *phi_dims))

        # sum over view representations
        r = torch.sum(phi, dim=1)

        # Use random (image, viewpoint) pair in batch as query
        x_q, v_q = images[:, query_idx], viewpoints[:, query_idx]

        batch_size, _, h, w = x_q.size()
        kl = 0

        # Increase dimensions
        v_q = v_q.view(batch_size, -1, 1, 1).repeat(1, 1, h//self.SCALE, w//self.SCALE)
        if r.size(2) != h//self.SCALE:
            r = r.repeat(1, 1, h//self.SCALE, w//self.SCALE)

        # Reset hidden state
        hidden_g = x_q.new_zeros((batch_size, self.h_dim, h//self.SCALE, w//self.SCALE))
        hidden_i = x_q.new_zeros((batch_size, self.h_dim, h//self.SCALE, w//self.SCALE))

        # Reset cell state
        cell_g = x_q.new_zeros((batch_size, self.h_dim, h//self.SCALE, w//self.SCALE))
        cell_i = x_q.new_zeros((batch_size, self.h_dim, h//self.SCALE, w//self.SCALE))

        u = x_q.new_zeros((batch_size, self.h_dim, h, w))
        x_q_downsampled = self.downsample(x_q)

        kls = 0
        for _ in range(self.L):
            # kl
            z = self.q.sample({"h_i": hidden_i})["z"]
            kl = KullbackLeibler(self.q, self.pi)
            kl_estimated = kl.estimate({"h_i":hidden_i, "h_g":hidden_g})
            kls += kl_estimated
            # update state
            hidden_i, cell_i = self.inference_core(x_q_downsampled, v_q, r, hidden_g, hidden_i, cell_i)
            hidden_g, cell_g, u = self.generator_core(z, v_q, r, hidden_g, cell_g, u)

        x_sample = self.g.sample({"u": u, "sigma":sigma})
        x_reconst = x_sample["x_q"]
        x_nll = -torch.mean(self.g.log_likelihood({"u":x_sample["u"], "sigma":x_sample["sigma"], "x_q": x_q}), dim=0)
        kls = torch.mean(torch.sum(kls.view(batch_size, -1),dim=1), dim=0)
        return x_nll, kls, x_q, x_reconst
