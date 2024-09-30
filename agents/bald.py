import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from core.agent import BaseAgent
from torch.utils.data import TensorDataset, DataLoader

# Based on https://github.com/acl21/deep-active-learning-pytorch
# Author: Akshay L Chandra
class BALD_Point(BaseAgent):


    def __init__(self, agent_seed, config, query_size=1, dropout_trials=25):
        super().__init__(agent_seed, config, query_size)
        assert "current_run_info" in config and "encoded" in config["current_run_info"]
        self.dropout_trials = dropout_trials

    @classmethod
    def inject_config(cls, config:dict, dropout_rate=0.05):
        """
        Add dropout to classification model
        """
        class_key = "classifier_embedded" if config["current_run_info"]["encoded"] else "classifier"
        config[class_key]["dropout"] = max(config[class_key].get("dropout", 0.0), dropout_rate)


    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer,
                      sample_size=10000) -> list[int]:

        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled),  sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]

            classifier.train()
            for m in classifier.modules():
                if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                    m.eval()

            device = x_unlabeled.device
            x_sample = x_unlabeled
            y_hat_collection = torch.zeros( (len(x_sample), self.dropout_trials) ).to(device)
            for trial in range(self.dropout_trials):
                y_hat = self._predict(x_sample, classifier)
                y_hat_collection[:, trial] = y_hat.squeeze(-1)

            std = torch.std(y_hat_collection, dim=1)
            ids = torch.topk(std, self.query_size).indices.tolist()
            # taken from https://github.com/cure-lab/deep-active-learning/blob/main/query_strategies/batch_BALD.py
            # res = get_batchbald_batch(y_hat_collection, self.query_size, 100)
        # ids = res.indices
        return sample_ids[ids]



class BALD_Dist_Entropy(BALD_Point):

    @classmethod
    def inject_config(cls, config:dict, dropout_rate=0.05):
        """
        Add dropout to classification model
        """
        class_key = "classifier_embedded" if config["current_run_info"]["encoded"] else "classifier"
        config[class_key]["dropout"] = max(config[class_key].get("dropout", 0.0), dropout_rate)


    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer,
                      sample_size=10000) -> list[int]:

        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled),  sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]

            classifier.train()
            for m in classifier.modules():
                if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                    m.eval()

            device = x_unlabeled.device
            batch_size = 256
            resolution = 200
            space_min = 0.1
            space_max = 0.9
            predictions = torch.zeros((self.dropout_trials, len(x_unlabeled), resolution)).to(device)
            loader = DataLoader(TensorDataset(x_unlabeled), batch_size=batch_size)
            for trial in range(self.dropout_trials):
                all_likelihoods = torch.zeros((0, resolution)).to(device)
                for (batch,) in loader:
                    cond_distribution = classifier(batch)
                    y_space_base = torch.linspace(space_min, space_max, resolution).reshape(resolution, 1).to(device)
                    y_space = torch.repeat_interleave(y_space_base, len(batch), dim=1).unsqueeze(-1)
                    lls = cond_distribution.log_prob(y_space)
                    if len(lls.shape) > 2:
                        lls = lls.squeeze(-1)
                    likelihoods = torch.exp(lls)
                    all_likelihoods = torch.cat([all_likelihoods, likelihoods.permute(1, 0)], dim=0)
                predictions[trial] = all_likelihoods

            predictions = predictions.cpu()
            average_prediction = torch.mean(predictions, dim=0).cpu()
            volume = torch.trapz(average_prediction, dx=(space_max - space_min) / resolution, dim=1)
            average_prediction /= volume.unsqueeze(-1)

            average_entropy = - average_prediction * torch.log(average_prediction)
            average_entropy = torch.trapz(average_entropy, dx=(space_max - space_min) / resolution, dim=1)

            entropy = - predictions * torch.log(predictions)
            entropy = torch.trapz(entropy, dx=(space_max - space_min) / resolution, dim=2)

            scores = torch.sum(average_entropy - entropy, dim=0)

            ids = torch.topk(scores, self.query_size).indices.tolist()
        return sample_ids[ids]


class BALD_LST(BALD_Point):

    @classmethod
    def inject_config(cls, config:dict, dropout_rate=0.05):
        """
        Add dropout to classification model
        """
        class_key = "classifier_embedded" if config["current_run_info"]["encoded"] else "classifier"
        config[class_key]["dropout"] = max(config[class_key].get("dropout", 0.0), dropout_rate)


    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer,
                      sample_size=10000) -> list[int]:

        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled),  sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]

            classifier.train()
            for m in classifier.modules():
                if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                    m.eval()

            device = x_unlabeled.device
            batch_size = 256
            resolution = 200
            space_min = 0.1
            space_max = 0.9
            predictions = torch.zeros((self.dropout_trials, len(x_unlabeled), resolution)).to(device)
            loader = DataLoader(TensorDataset(x_unlabeled), batch_size=batch_size)
            for trial in range(self.dropout_trials):
                all_likelihoods = torch.zeros((0, resolution)).to(device)
                for (batch,) in loader:
                    cond_distribution = classifier(batch)
                    y_space_base = torch.linspace(space_min, space_max, resolution).reshape(resolution, 1).to(device)
                    y_space = torch.repeat_interleave(y_space_base, len(batch), dim=1).unsqueeze(-1)
                    lls = cond_distribution.log_prob(y_space)
                    if len(lls.shape) > 2:
                        lls = lls.squeeze(-1)
                    likelihoods = torch.exp(lls)
                    all_likelihoods = torch.cat([all_likelihoods, likelihoods.permute(1, 0)], dim=0)
                predictions[trial] = all_likelihoods

            predictions = predictions.cpu()
            average_prediction = torch.mean(predictions, dim=0).cpu()
            volume = torch.trapz(average_prediction, dx=(space_max - space_min) / resolution, dim=1)
            average_prediction /= volume.unsqueeze(-1)

            average_entropy = - average_prediction * torch.log(average_prediction)
            average_entropy = torch.trapz(average_entropy, dx=(space_max - space_min) / resolution, dim=1)

            entropy = - predictions * torch.log(predictions)
            entropy = torch.trapz(entropy, dx=(space_max - space_min) / resolution, dim=2)

            scores = torch.sum(average_entropy - entropy, dim=0)

            ids = torch.topk(scores, self.query_size).indices.tolist()
        return sample_ids[ids]


class BALD_Dist_Std(BALD_Point):

    @classmethod
    def inject_config(cls, config:dict, dropout_rate=0.05):
        """
        Add dropout to classification model
        """
        class_key = "classifier_embedded" if config["current_run_info"]["encoded"] else "classifier"
        config[class_key]["dropout"] = max(config[class_key].get("dropout", 0.0), dropout_rate)


    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer,
                      sample_size=10000) -> list[int]:

        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled),  sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]

            classifier.train()
            for m in classifier.modules():
                if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                    m.eval()

            device = x_unlabeled.device
            scores = torch.zeros( len(x_unlabeled) ).to(device)
            previous_std = None
            for trial in range(self.dropout_trials):
                samples = self._sample(x_unlabeled, classifier, 64)
                std = torch.std(samples, dim=1)
                if previous_std is None:
                    previous_std = std
                    continue
                scores += std - previous_std
                previous_std = std

            ids = torch.topk(scores, self.query_size).indices.tolist()
        return sample_ids[ids]





class NFlows_Out(BALD_Dist_Entropy):

    @classmethod
    def inject_config(cls, config:dict, dropout_rate=0.05):
        """
        Add dropout to classification model
        """
        class_key = "classifier_embedded" if config["current_run_info"]["encoded"] else "classifier"
        config[class_key]["dropout"] = max(config[class_key].get("dropout", 0.0), dropout_rate)


    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer,
                      sample_size=10000) -> list[int]:

        with (torch.no_grad()):
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled),  sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]

            classifier.train()
            for m in classifier.modules():
                if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                    m.eval()

            device = x_unlabeled.device
            batch_size = 256
            resolution = 200
            space_min = 0.1
            space_max = 0.9
            log_likelihoods = torch.zeros((self.dropout_trials, len(x_unlabeled), resolution)).to(device)
            loader = DataLoader(TensorDataset(x_unlabeled), batch_size=batch_size)
            for trial in range(self.dropout_trials):
                all_likelihoods = torch.zeros((0, resolution)).to(device)
                for (batch,) in loader:
                    cond_distribution = classifier(batch)
                    y_space_base = torch.linspace(space_min, space_max, resolution).reshape(resolution, 1).to(device)
                    y_space = torch.repeat_interleave(y_space_base, len(batch), dim=1).unsqueeze(-1)
                    lls = cond_distribution.log_prob(y_space)
                    if len(lls.shape) > 2:
                        lls = lls.squeeze(-1)
                    all_likelihoods = torch.cat([all_likelihoods, lls.permute(1, 0)], dim=0)
                log_likelihoods[trial] = all_likelihoods

            log_likelihoods = log_likelihoods.cpu()
            total_uncertainty = - log_likelihoods
            total_uncertainty = total_uncertainty.mean(-1) # average over samples
            total_uncertainty = total_uncertainty.mean(0) # average over ensemble members

            aleatoric_uncertainty = - log_likelihoods.mean(-1) # average over samples
            epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty
            scores = epistemic_uncertainty.mean(0) # average over components

            ids = torch.topk(scores, self.query_size).indices.tolist()
        return sample_ids[ids]
