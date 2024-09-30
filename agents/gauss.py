import torch
from core.agent import BaseAgent
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np

class Gauss_Conf(BaseAgent):
    def __init__(self, agent_seed, config, query_size=1):
        super().__init__(agent_seed, config, query_size)
        self.data_loader_rng = torch.Generator()
        self.data_loader_rng.manual_seed(agent_seed)


    def _get_scores(self, x_unlabeled: Tensor, classifier: nn.Module, batch_size=256):
        scores = torch.zeros(size=(0, 1)).to(x_unlabeled.device)
        loader = DataLoader(TensorDataset(x_unlabeled), batch_size=batch_size)
        for (batch,) in loader:
            cond_distribution = classifier(batch)
            sample = cond_distribution.sample((sample_size,))
            probs = cond_distribution.log_prob(sample)
            probs = torch.pow(10, probs)
            probs /= probs.sum(dim=0)
            scores = torch.cat([scores, - probs.max(dim=0).values.unsqueeze(-1)], dim=0)
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
