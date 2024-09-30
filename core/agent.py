from typing import Union
from abc import ABC, abstractmethod
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class BaseAgent(ABC):

    def __init__(self, agent_seed, config:dict, query_size):
        self.agent_seed = agent_seed
        self.agent_rng = np.random.default_rng(agent_seed)
        self.config = config
        self.query_size = query_size
        self.name = str(self.__class__).split('.')[-1][:-2]
        print(f"Loaded Agent: {self.name}")

    @classmethod
    def inject_config(cls, config:dict):
        """
        This method can be used to change the dataset config.
        I.e. add dropout to the classification model
        """
        pass


    @abstractmethod
    def predict(self, x_unlabeled:Tensor,
                      x_labeled:Tensor, y_labeled:Tensor,
                      per_class_instances:dict,
                      budget:int, added_images:int,
                      initial_test_acc:float, current_test_acc:float,
                      classifier:Module, optimizer:Optimizer)->list[int]:
        """
        Sampling function for the acquisition function.
        Return one id, or list of ids from x_unlabeled
        """
        pass


    def _predict(self, x:Tensor, model:Module)->Tensor:
        with torch.no_grad():
            loader = DataLoader(TensorDataset(x),
                                batch_size=256)
            y_hat = None
            for batch in loader:
                batch = batch[0]
                emb_batch = model.predict(batch)
                if y_hat is None:
                    emb_dim = emb_batch.size(-1)
                    y_hat = torch.zeros((0, emb_dim)).to(emb_batch.device)
                y_hat = torch.cat([y_hat, emb_batch])
        return y_hat


    def _sample(self, x:Tensor, model:Module, sample_size)->Tensor:
        with torch.no_grad():
            loader = DataLoader(TensorDataset(x),
                                batch_size=256)
            samples = torch.zeros((0, sample_size)).to(x.device)
            for (batch,) in loader:
                dist = model(batch)
                s = dist.sample((sample_size,))
                samples = torch.cat([samples, s.squeeze(-1).permute(1,0)])
        return samples


    def _embed(self, x: Tensor, model: Module) -> Tensor:
        with torch.no_grad():
            loader = DataLoader(TensorDataset(x),
                                batch_size=512)
            emb_x = None
            for batch in loader:
                batch = batch[0]
                emb_batch = model._encode(batch)
                if emb_x is None:
                    emb_dim = emb_batch.size(-1)
                    emb_x = torch.zeros((0, emb_dim)).to(emb_batch.device)
                emb_x = torch.cat([emb_x, emb_batch])
        return emb_x

    def get_meta_data(self)->str:
        """
        Can be overwritten to provide additional information about the acquisition function.
        Contents will be stored into a meta.txt file for each run.
        """
        return f"{self.name}"
