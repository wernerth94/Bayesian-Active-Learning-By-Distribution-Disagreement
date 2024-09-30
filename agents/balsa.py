from agents.bald import BALD_Point
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.nn import Module
from scipy.stats import wasserstein_distance
from torch.utils.data import TensorDataset, DataLoader
from scipy.special import rel_entr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def do_bald_comp(predictions, average_prediction, dx):
    average_entropy = - average_prediction * torch.log(average_prediction)
    average_entropy = torch.trapz(average_entropy, dx=dx, dim=1)

    entropy = - predictions * torch.log(predictions)
    entropy = torch.trapz(entropy, dx=dx, dim=1)

    scores = average_entropy - entropy
    return scores


class BALSA_EMD(BALD_Point):


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
                      sample_size=5000) -> list[int]:

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
            wasser_dists = torch.zeros( len(x_sample) )
            previous_samples = None
            for trial in range(self.dropout_trials):
                samples = self._sample(x_sample, classifier, 64).cpu()
                if previous_samples is None:
                    previous_samples = samples
                    continue
                for i in range(len(samples)):
                    wasser_dists[i] += wasserstein_distance(previous_samples[i], samples[i])
                previous_samples = samples

            wasser_dists = wasser_dists.to(device)
            ids = torch.topk(wasser_dists, self.query_size).indices.tolist()

        return sample_ids[ids]


class BALSA_EMD_dual(BALD_Point):

    @classmethod
    def inject_config(cls, config: dict, dropout_rate=0.1):
        """
        Do not add additional dropout
        """
        # class_key = "classifier_embedded" if config["current_run_info"]["encoded"] else "classifier"
        # config[class_key]["dropout"] = max(config[class_key].get("dropout", 0.0), dropout_rate)

    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer,
                      sample_size=5000, dropout=0.1) -> list[int]:

        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled),  sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]

            classifier.train()
            previous_p = None
            for m in classifier.modules():
                if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                    m.eval()
                elif isinstance(m, torch.nn.Dropout):
                    if previous_p is None:
                        previous_p = m.p
                    m.p = max(dropout, m.p)

            device = x_unlabeled.device
            x_sample = x_unlabeled
            wasser_dists = torch.zeros( len(x_sample) )
            previous_samples = None
            for trial in range(self.dropout_trials):
                samples = self._sample(x_sample, classifier, 64).cpu()
                if previous_samples is None:
                    previous_samples = samples
                    continue
                for i in range(len(samples)):
                    wasser_dists[i] += wasserstein_distance(previous_samples[i], samples[i])
                previous_samples = samples

            wasser_dists = wasser_dists.to(device)
            ids = torch.topk(wasser_dists, self.query_size).indices.tolist()

            for m in classifier.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.p = previous_p

        return sample_ids[ids]


class BALSA_EMD_full(BALD_Point):

    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer,
                      sample_size=5000) -> list[int]:

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
            sample_data = None
            for trial in range(self.dropout_trials):
                samples = self._sample(x_sample, classifier, 64).cpu()
                if sample_data is None:
                    sample_data = samples.unsqueeze(0)
                else:
                    sample_data = torch.cat([sample_data, samples.unsqueeze(0).cpu()], dim=0)

            wasser_dists = torch.zeros(len(x_sample))
            for x in range(len(wasser_dists)):
                wasser_data = torch.zeros( (len(sample_data), len(sample_data)) )
                for i in range(len(sample_data)):
                    for j in range(len(sample_data)):
                        if j > i:
                            wasser_data[i, j] += wasserstein_distance(sample_data[i, x, :], sample_data[j, x, :])
                wasser_data += wasser_data.clone().T
                wasser_dists[x] = torch.mean(torch.sum(wasser_data, dim=1)).item()
            wasser_dists = wasser_dists.to(device)
            ids = torch.topk(wasser_dists, self.query_size).indices.tolist()

        return sample_ids[ids]


class BALSA_KL_Diff(BALD_Point):

    @classmethod
    def inject_config(cls, config: dict, dropout_rate=0.05):
        class_key = "classifier_embedded" if config["current_run_info"]["encoded"] else "classifier"
        config[class_key]["dropout"] = max(config[class_key].get("dropout", 0.0), dropout_rate)


    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer,
                      sample_size=5000) -> list[int]:

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
            predictions = torch.zeros( (self.dropout_trials, len(x_unlabeled), resolution) )
            loader = DataLoader(TensorDataset(x_unlabeled), batch_size=batch_size)
            for trial in range(self.dropout_trials):
                all_likelihoods = torch.zeros( (0, resolution) ).to(device)
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
            # BALSA KL norm
            # volume = torch.trapz(average_prediction, dx=(space_max - space_min) / resolution, dim=1)
            # average_prediction /= volume.unsqueeze(-1)

            scores = torch.zeros(size=(len(x_unlabeled), ))
            for trial in range(self.dropout_trials-1):
                kl = torch.log(predictions[trial] / average_prediction).sum(dim=1)

                baseline = rel_entr(predictions[trial], average_prediction).sum(dim=1)
                score = rel_entr(predictions[trial], predictions[trial+1]).sum(dim=1) - baseline
                scores += score

            ids = torch.topk(scores, self.query_size).indices.tolist()
        return sample_ids[ids]


class BALSA_KL_Anomaly(BALD_Point):

    @classmethod
    def inject_config(cls, config: dict, dropout_rate=0.05):
        class_key = "classifier_embedded" if config["current_run_info"]["encoded"] else "classifier"
        config[class_key]["dropout"] = max(config[class_key].get("dropout", 0.0), dropout_rate)


    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer,
                      sample_size=5000) -> list[int]:

        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled),  sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]

            classifier.train()
            for m in classifier.modules():
                if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                    m.eval()

            device = x_unlabeled.device
            predictions = torch.zeros( (self.dropout_trials, len(x_unlabeled), 64) ).to(device)
            for trial in range(self.dropout_trials):
                samples = self._sample(x_unlabeled, classifier, 64).permute(1,0).unsqueeze(-1)
                cond_dist = classifier(x_unlabeled)
                log_probs = cond_dist.log_prob(samples)
                predictions[trial] = torch.exp(log_probs).squeeze(-1).permute(1, 0)

            predictions = predictions.cpu()
            average_prediction = torch.mean(predictions, dim=0).cpu()
            lst_scores = torch.zeros(size=(len(x_unlabeled), ))
            for trial in range(self.dropout_trials):
                kl = torch.log(predictions[trial] / average_prediction).sum(dim=1)
                # lst_scores += kl
                lst_scores += -kl # complete anomaly

            ids = torch.topk(lst_scores, self.query_size).indices.tolist()
        return sample_ids[ids]


class BALSA_KL_Pairs(BALD_Point):

    @classmethod
    def inject_config(cls, config: dict, dropout_rate=0.05):
        class_key = "classifier_embedded" if config["current_run_info"]["encoded"] else "classifier"
        config[class_key]["dropout"] = max(config[class_key].get("dropout", 0.0), dropout_rate)


    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer,
                      sample_size=5000) -> list[int]:

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
            predictions = torch.zeros((self.dropout_trials, len(x_unlabeled), resolution))
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
            lst_scores = torch.zeros(size=(len(x_unlabeled),))
            for trial in range(self.dropout_trials - 1):
                kl = (predictions[trial] * torch.log(predictions[trial] / predictions[trial+1])).sum(dim=1)
                lst_scores += kl

            ids = torch.topk(lst_scores, self.query_size).indices.tolist()
        return sample_ids[ids]


class BALSA_KL_Grid(BALD_Point):

    @classmethod
    def inject_config(cls, config: dict, dropout_rate=0.05):
        class_key = "classifier_embedded" if config["current_run_info"]["encoded"] else "classifier"
        config[class_key]["dropout"] = max(config[class_key].get("dropout", 0.0), dropout_rate)


    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer,
                      sample_size=5000) -> list[int]:

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
            predictions = torch.zeros((self.dropout_trials, len(x_unlabeled), resolution))
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
            lst_scores = torch.zeros(size=(len(x_unlabeled), ))
            for trial in range(self.dropout_trials):
                kl = (predictions[trial] * torch.log(predictions[trial] / average_prediction)).sum(dim=1)
                lst_scores += kl

            ids = torch.topk(lst_scores, self.query_size).indices.tolist()
        return sample_ids[ids]


class BALSA_entropy(BALD_Point):

    @classmethod
    def inject_config(cls, config: dict, dropout_rate=0.05):
        class_key = "classifier_embedded" if config["current_run_info"]["encoded"] else "classifier"
        config[class_key]["dropout"] = max(config[class_key].get("dropout", 0.0), dropout_rate)


    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer,
                      sample_size=5000) -> list[int]:

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
            log_predictions = torch.zeros( (self.dropout_trials, len(x_unlabeled), resolution) )
            loader = DataLoader(TensorDataset(x_unlabeled), batch_size=batch_size)
            for trial in range(self.dropout_trials):
                all_likelihoods = torch.zeros( (0, resolution) ).to(device)
                for (batch,) in loader:
                    cond_distribution = classifier(batch)
                    y_space_base = torch.linspace(space_min, space_max, resolution).reshape(resolution, 1).to(device)
                    y_space = torch.repeat_interleave(y_space_base, len(batch), dim=1).unsqueeze(-1)
                    lls = cond_distribution.log_prob(y_space)
                    if len(lls.shape) > 2:
                        lls = lls.squeeze(-1)
                    all_likelihoods = torch.cat([all_likelihoods, lls.permute(1, 0)], dim=0)
                log_predictions[trial] = all_likelihoods

            average_log_prediction = torch.mean(log_predictions, dim=0)
            # Do I need to re-normalize with the area here?
            average_entropy = - torch.exp(average_log_prediction) * average_log_prediction
            average_entropy = torch.trapz(average_entropy, dx=(space_max - space_min) / resolution, dim=1)

            entropy = - torch.exp(log_predictions) * log_predictions
            entropy = torch.trapz(entropy, dx=(space_max - space_min) / resolution, dim=2)
            scores = torch.zeros(size=(len(x_unlabeled), ))
            for trial in range(self.dropout_trials-1):
                score = entropy[trial] - average_entropy
                scores += score

            ids = torch.topk(scores, self.query_size).indices.tolist()

        return sample_ids[ids]


class BALSA_KL_Grid_Norm(BALD_Point):

    @classmethod
    def inject_config(cls, config: dict, dropout_rate=0.05):
        class_key = "classifier_embedded" if config["current_run_info"]["encoded"] else "classifier"
        config[class_key]["dropout"] = max(config[class_key].get("dropout", 0.0), dropout_rate)


    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer,
                      sample_size=5000) -> list[int]:

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
            predictions = torch.zeros((self.dropout_trials, len(x_unlabeled), resolution))
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

            lst_scores = torch.zeros(size=(len(x_unlabeled), ))
            for trial in range(self.dropout_trials):
                kl = (predictions[trial] * torch.log(predictions[trial] / average_prediction)).sum(dim=1)
                lst_scores += kl

            ids = torch.topk(lst_scores, self.query_size).indices.tolist()
        return sample_ids[ids]


class BALSA_KL_Grid_dual(BALD_Point):

    @classmethod
    def inject_config(cls, config: dict, dropout_rate=0.05):
        pass
        # class_key = "classifier_embedded" if config["current_run_info"]["encoded"] else "classifier"
        # config[class_key]["dropout"] = max(config[class_key].get("dropout", 0.0), dropout_rate)


    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer,
                      sample_size=5000, dropout=0.05) -> list[int]:

        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled),  sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]

            classifier.train()
            previous_p = None
            for m in classifier.modules():
                if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                    m.eval()
                elif isinstance(m, torch.nn.Dropout):
                    if previous_p is None:
                        previous_p = m.p
                    m.p = max(dropout, m.p)

            device = x_unlabeled.device
            batch_size = 256
            resolution = 200
            space_min = 0.1
            space_max = 0.9
            predictions = torch.zeros((self.dropout_trials, len(x_unlabeled), resolution))
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
            # volume = torch.trapz(average_prediction, dx=(space_max - space_min) / resolution, dim=1)
            # average_prediction /= volume.unsqueeze(-1)

            lst_scores = torch.zeros(size=(len(x_unlabeled), ))
            for trial in range(self.dropout_trials):
                kl = (predictions[trial] * torch.log(predictions[trial] / average_prediction)).sum(dim=1)
                lst_scores += kl

            ids = torch.topk(lst_scores, self.query_size).indices.tolist()
        return sample_ids[ids]



class BALSA_KL_Pair_dual(BALD_Point):

    @classmethod
    def inject_config(cls, config: dict, dropout_rate=0.05):
        pass
        # class_key = "classifier_embedded" if config["current_run_info"]["encoded"] else "classifier"
        # config[class_key]["dropout"] = max(config[class_key].get("dropout", 0.0), dropout_rate)


    def predict(self, x_unlabeled: Tensor,
                      x_labeled: Tensor, y_labeled: Tensor,
                      per_class_instances: dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier: Module, optimizer: Optimizer,
                      sample_size=5000, dropout=0.05) -> list[int]:

        with torch.no_grad():
            sample_size = min(sample_size, len(x_unlabeled))
            sample_ids = np.random.choice(len(x_unlabeled),  sample_size, replace=False)
            x_unlabeled = x_unlabeled[sample_ids]

            classifier.train()
            previous_p = None
            for m in classifier.modules():
                if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                    m.eval()
                elif isinstance(m, torch.nn.Dropout):
                    if previous_p is None:
                        previous_p = m.p
                    m.p = max(dropout, m.p)

            device = x_unlabeled.device
            batch_size = 256
            resolution = 200
            space_min = 0.1
            space_max = 0.9
            predictions = torch.zeros((self.dropout_trials, len(x_unlabeled), resolution))
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
            lst_scores = torch.zeros(size=(len(x_unlabeled),))
            for trial in range(self.dropout_trials - 1):
                kl = (predictions[trial] * torch.log(predictions[trial] / predictions[trial + 1])).sum(dim=1)
                lst_scores += kl

            ids = torch.topk(lst_scores, self.query_size).indices.tolist()
        return sample_ids[ids]
