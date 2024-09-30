import math

import zuko
import numpy as np
import torch
from torch import Tensor
from torch.distributions.utils import _sum_rightmost
import torch.nn as nn
from torch.optim import Optimizer
from core.agent import BaseAgent
from torch.utils.data import TensorDataset, DataLoader
from core.helper_functions import EarlyStopping
from core.helper_functions import visualize_nf_prediction
import matplotlib.pyplot as plt

class NF_Diff(BaseAgent):
    def __init__(self, agent_seed, config, query_size=1):
        super().__init__(agent_seed, config, query_size)
        self.data_loader_rng = torch.Generator()
        self.data_loader_rng.manual_seed(agent_seed)


    def _learn_flow_model(self, x_labeled: Tensor):
        self.density_model = zuko.flows.NSF(x_labeled.size(-1),
                                            transforms=3, hidden_features=[128] * 3)
        self.density_model = self.density_model.to(x_labeled.device)
        self.density_optim = torch.optim.AdamW(self.density_model.parameters(), lr=1e-3)

        cut_point = max(1, int(len(x_labeled)*0.8))
        ids = np.arange(len(x_labeled))
        np.random.shuffle(ids)
        x_train, x_val = x_labeled[:cut_point], x_labeled[cut_point:]
        drop_last = 32 < len(x_train)
        train_dataloader = DataLoader(TensorDataset(x_train),
                                      batch_size=32,
                                      drop_last=drop_last,
                                      generator=self.data_loader_rng,
                                      shuffle=True)
        val_dataloader = DataLoader(TensorDataset(x_val), batch_size=512)
        early_stop = EarlyStopping(patience=5, lower_is_better=False)
        val_loss_history = [] # used for debugging
        for epoch in range(40):
            for (x,) in train_dataloader:
                self.density_optim.zero_grad()
                loss = -self.density_model().log_prob(x)
                loss = loss.mean()
                loss.backward()
                self.density_optim.step()
            with torch.no_grad():
                val_loss = 0.0
                for (x,) in val_dataloader:
                    loss = self.density_model().log_prob(x)
                    val_loss += loss.mean().item()
                    val_loss_history.append(val_loss)
                if early_stop.check_stop(val_loss):
                    break
        pass


    def _get_density_probs(self, x_unlabeled:Tensor, batch_size=256):
        scores = torch.zeros(size=(0, 1)).to(x_unlabeled.device)
        loader = DataLoader(TensorDataset(x_unlabeled), batch_size=batch_size)
        density_dist = self.density_model()
        for (batch,) in loader:
            density_prob = density_dist.log_prob(batch)
            # density_prob = torch.exp(density_prob)
            scores = torch.cat([scores, density_prob.unsqueeze(-1)], dim=0)
        return scores.squeeze()


    def _get_log_prob_entropy(self, flow_model, x_unlabeled:Tensor, y_labeled: Tensor,
                              batch_size=256, eps=1e-7, resolution=500):

        scores = torch.zeros(size=(0, 1)).to(x_unlabeled.device)
        loader = DataLoader(TensorDataset(x_unlabeled), batch_size=batch_size)
        for (batch,) in loader:
            cond_distribution = flow_model(batch)
            y_space_base = torch.linspace(-1.0, 2.0, resolution).reshape(resolution, 1)
            y_space = torch.repeat_interleave(y_space_base, len(batch), dim=1).unsqueeze(-1)

            lls = cond_distribution.log_prob(y_space)
            lls = torch.exp(lls)
            total_prob = torch.trapz(lls, y_space.squeeze(-1), dim=0)
            if len(lls.size()) == 2:
                normalized_prob = lls / total_prob.reshape(1, lls.size(1))
            else:
                normalized_prob = lls / total_prob.reshape(1, lls.size(1), 1)
            # entropy = normalized_prob / (1 + normalized_prob)
            entropy = - normalized_prob * torch.log(normalized_prob + 1e-7)
            entropy = entropy.mean(dim=0)
            if len(entropy.size()) == 1:
                entropy = entropy.unsqueeze(-1)
            scores = torch.cat([scores, entropy], dim=0)
        return scores.squeeze()


    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: nn.Module, optimizer: Optimizer,
                      sample_size=10000) -> list[int]:

        self._learn_flow_model(x_labeled)
        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled),  sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]
            base_uncertainty = self._get_density_probs(x_unlabeled)
            entropies = self._get_log_prob_entropy(classifier, x_unlabeled, y_labeled)

            entropy_delta = entropies.max() - entropies.min()
            base_uncertainty /= base_uncertainty.max()
            base_uncertainty /= (1.0 / entropy_delta) # very heuristic
            scores = entropies - base_uncertainty
            chosen = torch.topk(scores, self.query_size).indices.tolist()
        return sample_ids[chosen]

class NF_Proxy(BaseAgent):
    def __init__(self, agent_seed, config, query_size=1):
        super().__init__(agent_seed, config, query_size)
        self.data_loader_rng = torch.Generator()
        self.data_loader_rng.manual_seed(agent_seed)


    def _learn_flow_model(self, x_labeled: Tensor):
        self.density_model = zuko.flows.NSF(x_labeled.size(-1),
                                            transforms=3, hidden_features=[128] * 3)
        self.density_model = self.density_model.to(x_labeled.device)
        self.density_optim = torch.optim.AdamW(self.density_model.parameters(), lr=1e-3)

        cut_point = max(1, int(len(x_labeled)*0.8))
        ids = np.arange(len(x_labeled))
        np.random.shuffle(ids)
        x_train, x_val = x_labeled[:cut_point], x_labeled[cut_point:]
        drop_last = 32 < len(x_train)
        train_dataloader = DataLoader(TensorDataset(x_train),
                                      batch_size=32,
                                      drop_last=drop_last,
                                      generator=self.data_loader_rng,
                                      shuffle=True)
        val_dataloader = DataLoader(TensorDataset(x_val), batch_size=512)
        early_stop = EarlyStopping(patience=5, lower_is_better=False)
        val_loss_history = [] # used for debugging
        for epoch in range(40):
            for (x,) in train_dataloader:
                self.density_optim.zero_grad()
                loss = -self.density_model().log_prob(x)
                loss = loss.mean()
                loss.backward()
                self.density_optim.step()
            with torch.no_grad():
                val_loss = 0.0
                for (x,) in val_dataloader:
                    loss = self.density_model().log_prob(x)
                    val_loss += loss.mean().item()
                    val_loss_history.append(val_loss)
                if early_stop.check_stop(val_loss):
                    break
        pass


    def _get_density_probs(self, x_unlabeled:Tensor, batch_size=256):
        scores = torch.zeros(size=(0, 1)).to(x_unlabeled.device)
        loader = DataLoader(TensorDataset(x_unlabeled), batch_size=batch_size)
        density_dist = self.density_model()
        for (batch,) in loader:
            density_prob = density_dist.log_prob(batch)
            # density_prob = torch.exp(density_prob)
            scores = torch.cat([scores, density_prob.unsqueeze(-1)], dim=0)
        return scores.squeeze()


    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: nn.Module, optimizer: Optimizer,
                      sample_size=10000) -> list[int]:

        self._learn_flow_model(x_labeled)
        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled),  sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]
            base_uncertainty = self._get_density_probs(x_unlabeled)
            scores = - base_uncertainty
            chosen = torch.topk(scores, self.query_size).indices.tolist()
        return sample_ids[chosen]


class NF_Conf(BaseAgent):
    def __init__(self, agent_seed, config, query_size=1):
        super().__init__(agent_seed, config, query_size)
        self.data_loader_rng = torch.Generator()
        self.data_loader_rng.manual_seed(agent_seed)
        self.flow_model = None


    def _get_log_probs(self, flow_model, x_unlabeled:Tensor, batch_size=256, sample_size=64):
        scores = torch.zeros(size=(0, 1)).to(x_unlabeled.device)
        loader = DataLoader(TensorDataset(x_unlabeled), batch_size=batch_size)
        for (batch,) in loader:
            cond_distribution = flow_model(batch)
            sample = cond_distribution.sample((sample_size,))
            probs = cond_distribution.log_prob(sample)
            max_prob = probs.max(dim=0).values
            if len(max_prob.shape) == 1:
                max_prob = max_prob.unsqueeze(-1)
            scores = torch.cat([scores, -max_prob], dim=0)
            # Variant: selecting samples with the lowest maximum probability, like least conf sampling
        return scores.squeeze()


    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: nn.Module, optimizer: Optimizer,
                      sample_size=10000) -> list[int]:

        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled),  sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]
            scores = self._get_log_probs(classifier, x_unlabeled)
            chosen = torch.topk(scores, self.query_size).indices.tolist()
        return sample_ids[chosen]


class NF_Entropy(NF_Conf):

    def _get_log_prob_entropy(self, flow_model, x_unlabeled:Tensor, y_labeled: Tensor,
                              batch_size=256, eps=1e-7, resolution=500,
                              space_min=0.1, space_max=0.9):
        # Variant: selecting samples with high entropy in the probability function
        scores = torch.zeros(size=(0, 1)).to(x_unlabeled.device)
        loader = DataLoader(TensorDataset(x_unlabeled), batch_size=batch_size)
        for (batch,) in loader:
            cond_distribution = flow_model(batch)
            y_space_base = torch.linspace(space_min, space_max, resolution).reshape(resolution, 1)
            y_space = torch.repeat_interleave(y_space_base, len(batch), dim=1).unsqueeze(-1)
            y_space = y_space.to(x_unlabeled.device)

            lls = cond_distribution.log_prob(y_space)
            likelihoods = torch.exp(lls)
            # visualize_nf_prediction(y_space_base, likelihoods)
            entropy = - likelihoods * lls
            entropy = torch.trapz(entropy, dx=(space_max - space_min) / resolution, dim=0)

            if len(entropy.size()) == 1:
                entropy = entropy.unsqueeze(-1)
            scores = torch.cat([scores, entropy], dim=0)
        return scores.squeeze()

    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: nn.Module, optimizer: Optimizer,
                      sample_size=10000) -> list[int]:

        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled),  sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]
            scores = self._get_log_prob_entropy(classifier, x_unlabeled, y_labeled)
            chosen = torch.topk(scores, self.query_size).indices.tolist()
        return sample_ids[chosen]


class GaussStd(BaseAgent):
    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: nn.Module, optimizer: Optimizer,
                      sample_size=10000) -> list[int]:
        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled),  sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]
            cond_distribution = classifier(x_unlabeled)
            scores = cond_distribution.stddev.squeeze(-1)
            chosen = torch.topk(scores, self.query_size).indices.tolist()
        return sample_ids[chosen]

class GaussEntropy(BaseAgent):
    def predict(self, x_unlabeled: Tensor,
                x_labeled: Tensor, y_labeled: Tensor,
                per_class_instances: dict,
                budget: int, added_images: int,
                initial_test_acc: float, current_test_acc: float,
                classifier: nn.Module, optimizer: Optimizer,
                sample_size=10000) -> list[int]:
        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled), sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]
            cond_distribution = classifier(x_unlabeled)
            scores = cond_distribution.entropy().squeeze(-1)
            chosen = torch.topk(scores, self.query_size).indices.tolist()
        return sample_ids[chosen]


class GaussLC(BaseAgent):
    def predict(self, x_unlabeled: Tensor,
                x_labeled: Tensor, y_labeled: Tensor,
                per_class_instances: dict,
                budget: int, added_images: int,
                initial_test_acc: float, current_test_acc: float,
                classifier: nn.Module, optimizer: Optimizer,
                sample_size=10000) -> list[int]:
        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled), sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]
            cond_distribution = classifier(x_unlabeled)
            mid_points = cond_distribution.mean
            lls = cond_distribution.log_prob(mid_points)
            scores = (1 - lls).squeeze(-1)
            chosen = torch.topk(scores, self.query_size).indices.tolist()
        return sample_ids[chosen]


class NF_Std(NF_Conf):

    def _get_log_prob_std(self, flow_model, x_unlabeled:Tensor, batch_size=256, sample_size=64, eps=1e-7):
        # Variant: selecting samples with high entropy in the normalized log probs
        scores = torch.zeros(size=(0, 1)).to(x_unlabeled.device)
        loader = DataLoader(TensorDataset(x_unlabeled), batch_size=batch_size)
        for (batch,) in loader:
            cond_distribution = flow_model(batch)
            sample = cond_distribution.sample((sample_size,))
            log_probs = cond_distribution.log_prob(sample)
            if len(log_probs.shape) == 3:
                log_probs = log_probs.squeeze(-1) # compat between NF and Gauss models
            std = torch.std(log_probs, dim=0)
            scores = torch.cat([scores, std.unsqueeze(1)], dim=0)
        return scores.squeeze()

    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: nn.Module, optimizer: Optimizer,
                      sample_size=10000) -> list[int]:
        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled),  sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]
            scores = self._get_log_prob_std(classifier, x_unlabeled)
            chosen = torch.topk(scores, self.query_size).indices.tolist()
        return sample_ids[chosen]


class NF_Sample_Std(NF_Conf):

    def _get_sample_std(self, flow_model, x_unlabeled:Tensor, batch_size=256, sample_size=64, eps=1e-7, conditional=True):
        # Variant: selecting samples with high entropy in the normalized log probs
        scores = torch.zeros(size=(0, 1)).to(x_unlabeled.device)
        loader = DataLoader(TensorDataset(x_unlabeled), batch_size=batch_size)
        for batch in loader:
            batch = batch[0]
            cond_distribution = flow_model(batch)
            sample = cond_distribution.sample((sample_size,))
            # log_probs = torch.nn.functional.softmax(log_probs, dim=0)
            std = torch.std(sample, dim=0)
            scores = torch.cat([scores, std], dim=0)
        return scores.squeeze()

    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: nn.Module, optimizer: Optimizer,
                      sample_size=10000) -> list[int]:

        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled),  sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]
            scores = self._get_sample_std(classifier, x_unlabeled)
            chosen = torch.topk(scores, self.query_size).indices.tolist()
        return sample_ids[chosen]


class NF_BALD(NF_Conf):
    @classmethod
    def inject_config(cls, config:dict):
        class_key = "classifier_embedded" if config["current_run_info"]["encoded"] else "classifier"
        config[class_key]["dropout"] = 0.3

    def _get_disagreement(self, flow_model, x_unlabeled:Tensor,
                          batch_size=256, sample_size=64, n_forward_passes=20):
        mode = flow_model.mode
        flow_model.train()
        scores = torch.zeros(size=(0, 1)).to(x_unlabeled.device)
        loader = DataLoader(TensorDataset(x_unlabeled), batch_size=batch_size)
        for (batch,) in loader:
            y_hat = torch.zeros(size=(len(batch), 0)).to(x_unlabeled.device)
            for i in range(n_forward_passes):
                cond_distribution = flow_model(batch)
                sample = cond_distribution.sample((sample_size,))
                probs = cond_distribution.log_prob(sample)
                max_prob = probs.max(dim=0).values
                if len(max_prob.shape) == 1:
                    max_prob = max_prob.unsqueeze(-1)
                y_hat = torch.cat([y_hat, max_prob], dim=1)
            disagreement = torch.std(y_hat, dim=1)
            scores = torch.cat([scores, disagreement.unsqueeze(-1)], dim=0)
        flow_model.mode = mode
        return scores.squeeze()

    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: nn.Module, optimizer: Optimizer,
                      sample_size=10000) -> list[int]:

        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled),  sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]
            scores = self._get_disagreement(classifier, x_unlabeled)
            chosen = torch.topk(scores, self.query_size).indices.tolist()
        return sample_ids[chosen]



class NF_Density(NF_Conf):

    def _scores(self, flow_model, x_unlabeled, batch_size=256):
        scores = torch.zeros(size=(0, 1)).to(x_unlabeled.device)
        loader = DataLoader(TensorDataset(x_unlabeled), batch_size=batch_size)
        for (batch,) in loader:
            dist = flow_model()
            z, log_det = dist.transform.call_and_ladj(batch)
            log_det = _sum_rightmost(log_det, dist.reinterpreted)
            ll = dist.base.log_prob(z)
            ll = torch.exp(ll)
            det = torch.exp(log_det)
            lls = - ll * det
            scores = torch.cat([scores, lls.unsqueeze(-1)], dim=0)
        return scores.squeeze()

    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: nn.Module, optimizer: Optimizer,
                      sample_size=10000) -> list[int]:

        if not isinstance(classifier, zuko.flows.NSF):
            raise NotImplementedError()
            # self._learn_flow_model(x_labeled, y_labeled)
        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled),  sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]
            if isinstance(classifier, zuko.flows.NSF):
                scores = self._scores(classifier, x_unlabeled)
            else:
                raise NotImplementedError()
                # scores = self._get_log_probs(self.flow_model, x_unlabeled)
            chosen = torch.topk(scores, self.query_size).indices.tolist()
        return sample_ids[chosen]


class NF_Density_STD(NF_Density):

    def __init__(self, agent_seed, config, query_size, noise_var=1.0):
        self.noise_var = noise_var
        super().__init__(agent_seed, config, query_size)

    def _scores(self, flow_model, x_unlabeled, batch_size=256, pertubations=50):
        pert_scores = torch.zeros(size=(len(x_unlabeled), pertubations)).to(x_unlabeled.device)
        for p in range(pertubations):
            scores = torch.zeros(size=(0, 1)).to(x_unlabeled.device)
            noise = torch.normal(0, self.noise_var, size=x_unlabeled.size())
            pertubation = x_unlabeled + noise
            loader = DataLoader(TensorDataset(pertubation), batch_size=batch_size)
            for (batch,) in loader:
                dist = flow_model()
                z, log_det = dist.transform.call_and_ladj(batch)
                log_det = _sum_rightmost(log_det, dist.reinterpreted)
                ll = dist.base.log_prob(z)

                ll = torch.exp(ll)
                det = torch.exp(log_det)
                lls = - ll * det
                scores = torch.cat([scores, lls.unsqueeze(-1)], dim=0)
            pert_scores[:, p] = scores.squeeze()
        final_scores = pert_scores.std(dim=-1)
        return final_scores
