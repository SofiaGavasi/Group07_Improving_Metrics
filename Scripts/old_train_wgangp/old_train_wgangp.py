#  This code was derived from https://github.com/EmilienDupont/wgan-gp.git 
# Small changes had to be made since the repository was very old (and very old version) and some lines had to be updated



# Example: py Scripts/train_wgangp.py --dataset cifar10 --data-root data/CIFAR10 --epochs 1 --batch-size 64 --image-size 32 --out-dir outputs/wgangp_cifar10
from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.autograd import grad as torch_grad

from Models.wgangp import Generator, Discriminator
from Datasets.unified_dataset_loader import make_default_loader




class Trainer():
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                 gp_weight=10, critic_iterations=5, print_every=50,
                 use_cuda=False):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every


        # check this
        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def _critic_train_iteration(self, data):
        """ """
        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate probabilities on real and generated data
        # removed this line
        if self.use_cuda:
            data = data.cuda()
        d_real = self.D(data)
        d_generated = self.D(generated_data)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data)
        self.losses['GP'].append(gradient_penalty.item()) # changed to correspond to new version

        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()

        self.D_opt.step()

        # Record loss
        self.losses['D'].append(d_loss.item()) # also changed

    def _generator_train_iteration(self, data):
        """ """
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate loss and optimize
        d_generated = self.D(generated_data)
        g_loss = - d_generated.mean()
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.losses['G'].append(g_loss.item()) # also changed

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.detach() + (1 - alpha) * generated_data.detach() # changed .data to .detach(), a better option
        interpolated = interpolated.requires_grad_(True) # also chaned
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() if self.use_cuda else torch.ones(
                               prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().item()) # also changed

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            self._critic_train_iteration(data[0])
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(data[0])

            # if i % self.print_every == 0:
            #     print("Iteration {}".format(i + 1))
            #     print("D: {}".format(self.losses['D'][-1]))
            #     print("GP: {}".format(self.losses['GP'][-1]))
            #     print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
            #     if self.num_steps > self.critic_iterations:
            #         print("G: {}".format(self.losses['G'][-1]))

    def train(self, data_loader, epochs):
        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            self._train_epoch(data_loader)



    def sample_generator(self, num_samples):
        latent_samples = self.G.sample_latent(num_samples) # also removed variable
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_data = self.G(latent_samples)
        return generated_data

    def sample(self, num_samples):
        generated_data = self.sample_generator(num_samples)
        # Remove color channel
        return generated_data.detach().cpu().numpy()[:, 0, :, :] # changed to .detach()
    




def main():
    parser = argparse.ArgumentParser(description="Train DCGAN on MNIST/CIFAR-10.")
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar10"])
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--outf", type=str, default="outputs/wgangp")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    Path(args.outf).mkdir(parents=True, exist_ok=True)



    # repo_root = Path(__file__).resolve().parents[1]

    
    if args.dataset == "mnist":
        numChannels = 1
    else:
        numChannels = 3

    img_size = (args.image_size, args.image_size, numChannels)



    loader = make_default_loader(dataset_name=args.dataset, 
                                 data_root= args.data_root,
                                 image_size = args.image_size)
    dataset = loader.get_dataset()

    data_loader = DataLoader(dataset, batch_size= args.batch_size, shuffle=True)    

   


    generator = Generator(img_size=img_size, latent_dim=100, dim=16) 
    discriminator = Discriminator(img_size=img_size, dim=16)

    print(generator)
    print(discriminator)

    # Initialize optimizers
    betas = (.9, .99)
    G_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=betas)
    D_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=betas)

    # Train model
    trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                    use_cuda=torch.cuda.is_available())
    trainer.train(data_loader, args.epochs)

    # Save models
    name = 'mnist_model'
    torch.save(trainer.G.state_dict(), './gen_' + name + '_' + str(args.epochs) +'.pt')
    torch.save(trainer.D.state_dict(), './dis_' + name + '-' + str(args.epochs) +'.pt')


if __name__ == "__main__":
    main()






