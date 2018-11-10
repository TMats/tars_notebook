import argparse
import datetime
import math
import os
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from shepardmetzler import ShepardMetzler, Scene, transform_viewpoint
from model import GQN


seed = 1234
torch.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generative Query Network on Shepard Metzler Example')
    parser.add_argument('--gradient_steps', type=int, default=2*(10**6), help='number of gradient steps to run (default: 2 million)')
    parser.add_argument('--batch_size', type=int, default=36, help='size of batch (default: 36)')
    parser.add_argument('--data_dir', type=str, help='location of training data', default="train")
    parser.add_argument('--root_log_dir', type=str, help='root location of log', default='log')
    parser.add_argument('--log_interval', type=int, help='interval number of steps for logging', default=1000)
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # args
    train_data_dir = '/workspace/dataset/shepard_metzler_7_parts-torch/train'

    # number of workers to load data
    num_workers = 0

    # for logging
    log_interval_num = 500
    dir_name = str(datetime.datetime.now())
    log_dir = '/workspace/logs/'+ dir_name
    os.mkdir(log_dir)
    os.mkdir(log_dir+'/models')
    os.mkdir(log_dir+'/runs')

    # tensorboardX
    writer = SummaryWriter(log_dir=log_dir+'/runs')

    batch_size = 36
    gradient_steps = 2*(10**6)

    train_dataset = ShepardMetzler(root_dir=train_data_dir, target_transform=transform_viewpoint)

    # model settings
    xDim=3
    vDim=7
    rDim=256
    hDim=128
    zDim=64
    L=12
    SCALE = 4 # Scale of image generation process

    # model
    gqn=GQN(xDim,vDim,rDim,hDim,zDim, L, SCALE).to(device)
    gqn = nn.DataParallel(gqn, device_ids=[0, 1, 2])

    # Pixel variance
    sigma_f, sigma_i = 0.7, 2.0

    # Learning rate
    mu_f, mu_i = 5*10**(-5), 5*10**(-4)
    mu, sigma = mu_i, sigma_i

    optimizer = torch.optim.Adam(gqn.parameters(), lr=mu, betas=(0.9, 0.999))
    kwargs = {'num_workers':num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    # Number of gradient steps
    s = 0
    while True:
        for x, v in tqdm(loader):
            x = x.to(device)
            v = v.to(device)
            nll, kl, x_q, x_rec, x_gen = gqn(x, v, sigma)
            nll = nll.mean()
            kl = kl.mean()
            loss = nll + kl
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            writer.add_scalar('train_nll', nll, s)
            writer.add_scalar('train_kl', kl, s)
            writer.add_scalar('train_loss', loss, s)
            # Keep a checkpoint every n steps
            if s % log_interval_num == 0:
                torch.save(gqn, log_dir + "/models/model-{}.pt".format(s))
                writer.add_image('ground_truth', x_q[0], s)
                writer.add_image('reconstruction', x_rec[0], s)
                writer.add_image('generation', x_gen[0], s)

            if s >= gradient_steps:
                break

            s += 1

            # Anneal learning rate
            mu = max(mu_f + (mu_i - mu_f)*(1 - s/(1.6 * 10**6)), mu_f)
            optimizer.lr = mu
            # Anneal pixel variance
            sigma = max(sigma_f + (sigma_i - sigma_f)*(1 - s/(2 * 10**5)), sigma_f)

        if s >= gradient_steps:
            torch.save(gqn, log_dir + "/models/model-final.pt")
            break
    writer.close()

