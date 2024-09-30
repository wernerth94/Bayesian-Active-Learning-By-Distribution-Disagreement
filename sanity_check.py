import torch
import torch.nn as nn
import yaml
import experiment_util as util
from core.helper_functions import *
import sklearn.tree
import zuko
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from datasets import *
from core.data import normalize
from core.helper_functions import visualize_nf_prediction
from classifiers.classifier import GaussNN
from classifiers.normalizing_flow import EncNSF
from properscoring import crps_ensemble
import scipy
import random

# n_numbers = 1000
# repeats = 10
# dist = torch.distributions.Gamma(2.0, 2.0)
# nmbrs = dist.sample((n_numbers,))
# all_idxs = list(range(n_numbers))
# avrg = nmbrs.mean()
# full = (avrg - nmbrs).sum()
# test_range = torch.arange(2, n_numbers-1, 2)
# results = torch.zeros(len(test_range))
# for _ in tqdm(range(repeats)):
#     for idx, n in enumerate(test_range):
#         approx = 0
#         random.shuffle(all_idxs)
#         for i in range(int(n)):
#             approx += nmbrs[all_idxs[i]] - nmbrs[all_idxs[i+1]]
#         results[idx] += approx
# results /= repeats
# plt.plot(test_range, results)
# plt.hlines(full, test_range[0], test_range[-1])
# plt.show()
# exit(0)

# space_min = -10
# space_max = 10
# resolution = 4000
# y_space = torch.linspace(space_min, space_max, resolution).reshape(resolution, 1)
# diffs = []
# for i in range(400):
#     mu = torch.rand(1)
#     sig = torch.rand(1)
#     dist = torch.distributions.Normal(mu, sig)
#     lls = dist.log_prob(y_space)
#     likelihoods = torch.exp(lls)
#     entropy = - likelihoods * lls
#     trapz_entropy = torch.trapz(entropy, dx=(space_max - space_min) / resolution, dim=0)
#     diffs.append(torch.abs(trapz_entropy - dist.entropy()).item())
#
# plt.hist(diffs)
# plt.show()


class DummyDataset():
    pass

def split(x, y, p=0.8):
    ids = np.arange(len(x))
    np.random.shuffle(ids)
    cut = int(p * len(x))
    train_ids, test_ids = ids[:cut], ids[cut:]
    x_train, y_train = x[train_ids], y[train_ids]
    x_test, y_test = x[test_ids], y[test_ids]
    return x_train, y_train, x_test, y_test

eps = 5e-1
ana = []
man = []
x = torch.arange(0, 1, 0.01)
for m1 in x:
    # p = torch.distributions.Normal(torch.rand(1), torch.rand(1)+eps)
    # q = torch.distributions.Normal(torch.rand(1), torch.rand(1)+eps)
    p = torch.distributions.Normal(m1, 0.5)
    q = torch.distributions.Normal(1, 0.5)
    grid = torch.linspace(-0.5, 1.5, 200)

    analytic_kl = torch.log(q.scale/p.scale) + ((p.scale**2 + (p.mean - q.mean)**2) / (2 * q.scale**2)) - 0.5
    p_hat = p.log_prob(grid).exp()
    q_hat = q.log_prob(grid).exp()
    manual_kl = (p_hat * torch.log(p_hat / q_hat)).sum()
    ana.append(analytic_kl.item())
    man.append(manual_kl.item())
    # scipy_kl = scipy.special.rel_entr(p_hat.numpy(), q_hat.numpy()).sum()
    # print("analytic_kl", analytic_kl.item())
    # print("manual_kl", manual_kl.item())
    # print("scipy_kl", scipy_kl)
# print("pearson", scipy.stats.pearsonr(ana, man))
# plt.scatter(ana, man)
# plt.show()
plt.scatter(x, ana)
plt.title("Analytic")
plt.show()
plt.scatter(x, man)
plt.title("Manual")
plt.show()
exit(0)

pool_rng = np.random.default_rng(1)
model_rng = torch.Generator()
model_rng.manual_seed(1)
dataset_name = "sarcos"
with open(f"configs/{dataset_name}_gauss.yaml", 'r') as f:
    config = yaml.load(f, yaml.Loader)
dataset = Sarcos("../datasets", config, pool_rng, 0)

drop = 0.1124
wd = 0.0
lr = 0.0095
model = EncNSF(model_rng, 1, dataset.x_train.shape[-1], config["classifier"]["hidden"], dropout=drop,)
# opt = torch.optim.Adam([
#     { 'params': model.inpt.parameters(), 'weight_decay': wd}, # only apply L2 to the conditioner model
#     { 'params': model.hidden.parameters(), 'weight_decay': wd}, # only apply L2 to the conditioner model
#     { 'params': model.flow_head.parameters(), 'weight_decay': 0.0},
# ], lr=lr)
opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
ids = np.random.choice(len(dataset.x_train),
                       config["dataset"]["budget"] + config["dataset"]["initial_points_per_class"], replace=False)
train_loader = DataLoader(TensorDataset(dataset.x_train[ids], dataset.y_train[ids]), batch_size=64)
test_loader = DataLoader(TensorDataset(dataset.x_test, dataset.y_test), batch_size=256)
test_likes = []
test_maes = []
for epoch in range(400):
    train_loss = 0.0
    test_like = 0.0
    test_mae = 0.0
    test_crps = 0.0
    for (x, y) in train_loader:
        opt.zero_grad()
        loss_value = - model(x).log_prob(y)
        loss_value = loss_value.mean()
        train_loss += loss_value.item()
        loss_value.backward()
        opt.step()
    with torch.no_grad():
        for (x, y) in test_loader:
            loss = - model(x).log_prob(y)
            loss = loss.mean()
            test_like += loss.item()
            y_hat = model.predict(x)
            test_mae += torch.abs(y - y_hat).sum().item()
            cond_dist = model(x)
            sample = cond_dist.sample((64,))
            test_crps += crps_ensemble(y.squeeze(-1), sample.squeeze(-1).permute(1, 0)).sum().item()
    test_likes.append(test_like / len(dataset.y_test))
    test_maes.append(test_mae / len(dataset.y_test))
    print(f"Epoch: {epoch} trainloss: %1.4f, testloss %1.4f, CRPS %1.4f, test MAE %1.4f" % (train_loss / len(dataset.y_train),
                                                                                test_like / len(dataset.y_test),
                                                                                test_crps / len(dataset.y_test),
                                                                                test_mae / len(dataset.y_test)))
print("test_likes")
print(test_likes)
print("test_maes")
print(test_maes)

fig, ax1 = plt.subplots()
ax1.plot(test_maes, c='g')
ax1.set_ylabel('Test MAE', color='g')
ax2 = ax1.twinx()
ax2.plot(test_likes, 'b')
ax2.set_ylabel('Test Likelihood', color='b')
plt.savefig("raytune_output/%s_lr_%.2e_wd_%.2e_dr_%.2e.jpg"%(dataset_name, lr, wd, drop))
plt.show()

