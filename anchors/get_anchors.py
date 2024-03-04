import os, argparse, time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import clip

import utils


def get_arguments():
    parser = argparse.ArgumentParser(description="Adversarial Training for CLIP.")
    parser.add_argument(
        "--modelname", default="ViT-B/16", choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"]
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="This is a photo of a {}",
        help="prompt to embed class label",
    )
    parser.add_argument("--classes", default="imagenet_classes.txt")
    parser.add_argument("--to", default="imagenet_anchors.npy")
    parser.add_argument("--bs", type=int, default=100)
    return parser.parse_args()


args = get_arguments()
device = utils.get_device()
# models
clip_model, clip_preprocess = clip.load(args.modelname, device=device)
clip_model.eval()

with open(args.classes, "r") as f:
    train_labels = list(
        map(lambda _: args.prompt.format(_.rstrip().replace("_", " ")), f.readlines())
    )
train_text_tokens = clip.tokenize(train_labels).to(device)
n_batches = (len(train_labels) - 1) // args.bs + 1
train_text_features = []
for idx in range(n_batches):
    X = train_text_tokens[idx*args.bs:(idx+1)*args.bs]
    with torch.no_grad():
        train_text_features.append(clip_model.encode_text(X).float().cpu())
np.save(args.to, torch.cat(train_text_features).numpy())
