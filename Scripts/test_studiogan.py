# py -m pip install seaborn matplotlib pandas scikit-learn tqdm pyyaml scipy kornia easydict PyYAML

import logging
import torch
import sys
from pathlib import Path
import torchvision.utils as vutils

# paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# need a logger
logger = logging.getLogger("studiogan")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


from Models.pretrained_wrappers import StudioGANWrapper

# different roots/paths 
repo = ROOT / "checkpoints" / "StudioGAN" / "CIFAR10" / "studioGAN_src"
ckpt = ROOT / "checkpoints" / "StudioGAN" / "CIFAR10" / "studioGAN_generator.pkl"


device = "cuda" if torch.cuda.is_available() else "cpu"

# model
model = StudioGANWrapper(
    repo_path=repo,
    ckpt_path=ckpt,
    config_name="SNGAN.yaml", 
    device=device,
    logger=logger
)

# sampling 
n_samples = 16
images = model.sample(n_samples)

print(f"Generated tensor shape: {images.shape}")

# save test image
output_path = ROOT / "output_test.png"
vutils.save_image(images, output_path, nrow=4, normalize=True)
