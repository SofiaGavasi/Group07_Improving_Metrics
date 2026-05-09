"""
call each module in Models/ and print a quick status report.

Run from repo root:
    py Tests/call_models_files.py
"""

from __future__ import annotations
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Models import DCGANDiscriminator, DCGANGenerator
from Models.pretrained_wrappers import DDPMWrapper, StudioGANWrapper, StyleGAN2Wrapper
from Models.wgangp import WGANGPCritic, WGANGPGenerator, gradient_penalty




def call_dcgan_module():
    print("\n[Models/dcgan.py]")
    gen = DCGANGenerator(ngpu=0)
    dis = DCGANDiscriminator(ngpu=0)

    z = torch.randn(4, 100, 1, 1)
    fake = gen(z)
    score = dis(fake)
    print(f"Generator output shape: {tuple(fake.shape)}")
    print(f"Discriminator output shape: {tuple(score.shape)}")


def call_wgangp_module():
    print("\n[Models/wgangp.py]")
    wg_gen = WGANGPGenerator()
    wg_critic = WGANGPCritic()
    fake = wg_gen(torch.randn(2, 100, 1, 1))
    scores = wg_critic(torch.randn(2, 3, 32, 32))
    gp = gradient_penalty(
        critic=wg_critic,
        real=torch.randn(2, 3, 32, 32),
        fake=torch.randn(2, 3, 32, 32),
        device=torch.device("cpu"),
    )
    print(f"WGANGPGenerator output shape: {tuple(fake.shape)}")
    print(f"WGANGPCritic output shape: {tuple(scores.shape)}")
    print(f"gradient_penalty value: {float(gp.detach()):.4f}")


def call_pretrained_wrappers_module():
    print("\n[Models/pretrained_wrappers.py]")
    wrappers = [
        StudioGANWrapper("checkpoints/studiogan_dummy.ckpt"),
        DDPMWrapper("checkpoints/ddpm_dummy.ckpt"),
        StyleGAN2Wrapper("checkpoints/stylegan2_dummy.ckpt"),
    ]

    for wrapper in wrappers:
        try:
            _ = wrapper.sample(2, device=torch.device("cpu"))
        except NotImplementedError as exc:
            print(f"{wrapper.__class__.__name__}.sample -> TODO ({exc})")


def main():
    call_dcgan_module()
    call_wgangp_module()
    #call_pretrained_wrappers_module()


if __name__ == "__main__":
    main()
