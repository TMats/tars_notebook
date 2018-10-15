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

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # for logging
    log_dir = args.root_log_dir +'/'+str(datetime.datetime.now())
    os.mkdir(log_dir)
    os.mkdir(log_dir+'/models')
    os.mkdir(log_dir+'/runs')

    # tensorboardX
    writer = SummaryWriter(log_dir=log_dir+'/runs')

    train_dataset = ShepardMetzler(root_dir=args.data_dir, target_transform=transform_viewpoint)

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

    # Pixel variance
    sigma_f, sigma_i = 0.7, 2.0

    # Learning rate
    mu_f, mu_i = 5*10**(-5), 5*10**(-4)
    mu, sigma = mu_f, sigma_f

    optimizer = torch.optim.Adam(gqn.parameters(), lr=mu)
    kwargs = {'num_workers':args.workers, 'pin_memory': True} if torch.cuda.is_available() else {}
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # Number of gradient steps
    s = 0
    while True:
        for x, v in tqdm(loader):
            x = x.to(device)
            v = v.to(device)
            x_nll, kls, x_target, x_reconst = gqn(x, v, sigma)
            loss = x_nll + kls
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # Keep a checkpoint every n steps
            if s % args.log_interval == 0:
                torch.save(gqn, log_dir + "/models/model-{}.pt".format(s))
                writer.add_scalar('train_nll', x_nll, s)
                writer.add_scalar('train_kl', kls, s)
                writer.add_scalar('train_loss', loss, s)
                writer.add_image('target_image', x_target[0], s)
                writer.add_image('target_reconst', x_reconst[0], s)

            if s >= args.gradient_steps:
                break

            s += 1

            # Anneal learning rate
            mu = max(mu_f + (mu_i - mu_f)*(1 - s/(1.6 * 10**6)), mu_f)
            optimizer.lr = mu * math.sqrt(1 - 0.999**s)/(1 - 0.9**s)
            # Anneal pixel variance
            sigma = max(sigma_f + (sigma_i - sigma_f)*(1 - s/(2 * 10**5)), sigma_f)

        if s >= args.gradient_steps:
            torch.save(gqn, log_dir + "/models/model-final.pt")
            break
    writer.close()

