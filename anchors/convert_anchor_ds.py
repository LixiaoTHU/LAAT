import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import math

def Rot_map(V):
    assert len(V.shape) == 1
    assert np.linalg.norm(V) - 1 < 1e-8
    n_dim = V.shape[0]
    Rot = np.eye(n_dim)
    Rot_inv = np.eye(n_dim)
    for rotate in range(n_dim-1):
        rot_mat = np.eye(n_dim)
        rot_norm = np.sqrt(V[rotate]**2 + V[rotate+1]**2)
        cos_theta = V[rotate+1]/rot_norm
        sin_theta = V[rotate]/rot_norm
        rot_mat[rotate,rotate] = cos_theta
        rot_mat[rotate,rotate+1] = - sin_theta
        rot_mat[rotate+1,rotate] = sin_theta
        rot_mat[rotate+1,rotate+1] = cos_theta

        V = np.dot(rot_mat, V)

        Rot = np.dot(rot_mat, Rot)
        Rot_inv = np.dot(Rot_inv,rot_mat.transpose())
    return Rot, Rot_inv

def convert(text_features, downstream_feature = None, dim = 512):
    '''text_features: Tensor, [N, dim]'''
    text_features = text_features /  text_features.norm(dim=-1, keepdim=True)
    print("original cos similarity: ", torch.mean(torch.matmul(text_features, text_features.T)))
    anchor = torch.mean(text_features, dim = 0)
    anchor = anchor / torch.norm(anchor)
    anchor = anchor.detach().cpu().numpy()

    anchor = anchor.astype(np.float64)
    anchor = anchor / np.linalg.norm(anchor)
    target = np.zeros(dim)
    target[0] += 1.0

    R_0, R_0_inv = Rot_map(target)
    R_X, _ = Rot_map(np.dot(R_0, anchor))
    R = np.dot(np.dot(R_0_inv, R_X), R_0)
    R = torch.from_numpy(R).to(text_features.device)

    new_text_features = torch.matmul(text_features.double(), R.T)
    anchor = torch.from_numpy(anchor).to(R.device)
    target = torch.matmul(anchor, R.T)

    similarity = new_text_features.matmul(target.T)
    mincos = torch.min(similarity)
    theta = torch.arccos(mincos)

    theta1 = torch.arccos(new_text_features[:,0])
    theta2 = 2 * math.pi * (theta1) / ((theta) * 4)

    converted = torch.zeros(new_text_features.shape).double().to(new_text_features.device)
    converted[:,1:] = new_text_features[:,1:] * torch.sin(theta2).unsqueeze(1) / torch.sin(theta1).unsqueeze(1)
    converted[:,0] = torch.cos(theta2)

    converted = torch.matmul(converted, R)

    print("converted cos similarity: ", torch.mean(torch.matmul(converted, converted.T)))
    if downstream_feature is not None:
        downstream_feature = downstream_feature /  downstream_feature.norm(dim=-1, keepdim=True)
        new_downstream_feature = torch.matmul(downstream_feature.double(), R.T)
        theta1 = torch.arccos(new_downstream_feature[:,0])
        theta2 = 2 * math.pi * (theta1) / ((theta) * 4)

        downstream_converted = torch.zeros(new_downstream_feature.shape).double().to(new_downstream_feature.device)
        downstream_converted[:,1:] = new_downstream_feature[:,1:] * torch.sin(theta2).unsqueeze(1) / torch.sin(theta1).unsqueeze(1)
        downstream_converted[:,0] = torch.cos(theta2)

        downstream_converted = torch.matmul(downstream_converted, R)

        return downstream_converted


    return converted

if __name__ == "__main__":
    import sys
    weight = np.load(sys.argv[1]+"_anchors.npy")
    weight_ds = np.load(sys.argv[2]+"_anchors.npy")
    text_features = torch.from_numpy(weight)
    ds_features = torch.from_numpy(weight_ds)
    np.save(f"{sys.argv[2]}_{sys.argv[1]}_ds.npy", convert(text_features, ds_features).float().numpy())
