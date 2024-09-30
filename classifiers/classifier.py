from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from core.data import BaseDataset
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from classifiers.seeded_layers import SeededLinear
from classifiers.lstm import BiLSTMModel
from classifiers.gauss_nn import GaussNN
from classifiers.normalizing_flow import EncNSF
import zuko
from properscoring import crps_ensemble


class LinearModel(nn.Module):
    def __init__(self, model_rng, input_size:int, num_classes:int, dropout=None):
        super().__init__()
        self.dropout = dropout
        self.out = SeededLinear(model_rng, input_size, num_classes)

    def _encode(self, x:Tensor)->Tensor:
        return x

    def forward(self, x:Tensor)->Tensor:
        if self.dropout is not None:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out(x)
        return x


def construct_model(model_rng, dataset:BaseDataset, model_config:dict, add_head=True) -> Tuple[nn.Module, int]:
        '''
        Constructs the model by name and additional parameters
        Returns model and its output dim
        '''
        x_shape = dataset.x_shape
        n_classes = dataset.n_classes
        model_type = model_config["type"].lower()
        dropout = model_config["dropout"] if "dropout" in model_config else 0.0
        if model_type == "linear":
            return LinearModel(model_rng, x_shape[-1], n_classes, dropout), \
                   n_classes
        elif model_type == "nsf":
            return EncNSF(model_rng, n_classes, x_shape[-1], model_config["hidden"], dropout=dropout,), \
                   n_classes if add_head else model_config["hidden"][-1]
        elif model_type == "nsf_density":
            return zuko.flows.NSF(x_shape[-1],
                                  transforms=len(model_config["hidden"]), hidden_features=model_config["hidden"],
                                  activation=torch.nn.ReLU), \
                   n_classes if add_head else model_config["hidden"][-1]
        elif model_type == "gauss":
            return GaussNN(model_rng, x_shape[-1], model_config["hidden"], dropout=dropout), \
                   n_classes if add_head else model_config["hidden"]
        else:
            raise NotImplementedError



def fit_and_evaluate(dataset:BaseDataset,
                     model_rng,
                     disable_progess_bar:bool=False,
                     max_epochs:int=1000,
                     patience:int=5):

    from core.helper_functions import EarlyStopping
    model = dataset.get_classifier(model_rng)
    model = model.to(dataset.device)
    if dataset.encoded:
        optim_cfg = dataset.config["optimizer_embedded"]
    else:
        optim_cfg = dataset.config["optimizer"]
    optimizer = torch.optim.NAdam(model.parameters(), lr=optim_cfg["lr"], weight_decay=optim_cfg["weight_decay"])

    train_dataloader = DataLoader(TensorDataset(dataset.x_train, dataset.y_train),
                                  batch_size=dataset.classifier_batch_size,
                                  shuffle=True)
    val_dataloader = DataLoader(TensorDataset(dataset.x_val, dataset.y_val), batch_size=512)
    test_dataloader = DataLoader(TensorDataset(dataset.x_test, dataset.y_test), batch_size=512)
    test_maes = []
    test_likelihoods = []
    test_crpss = []
    early_stop = EarlyStopping(patience=patience, lower_is_better=True)
    iterator = tqdm(range(max_epochs), disable=disable_progess_bar, miniters=2)
    for e in iterator:
        model.train()
        epoch_loss = 0.0
        for i, (batch_x, batch_y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss_value = - model(batch_x).log_prob(batch_y)
            loss_value = loss_value.mean()
            loss_value.backward()
            optimizer.step()
            epoch_loss += loss_value.item()
        epoch_loss = epoch_loss / len(train_dataloader)
        # early stopping on validation
        with torch.no_grad():
            model.eval()
            val_mae_sum = 0.0
            for batch_x, batch_y in val_dataloader:
                y_hat = model.predict(batch_x)
                val_mae_sum += torch.abs(batch_y - y_hat).sum().item()
            if early_stop.check_stop(val_mae_sum):
                print(f"early stop after {e} epochs")
                break

        # test loss and accuracy
        with torch.no_grad():
            model.eval()
            test_mae = 0.0
            test_likelihood = 0.0
            test_crps = 0.0
            for batch_x, batch_y in test_dataloader:
                cond_dist = model(batch_x)
                sample = cond_dist.sample((64,))
                argmax = cond_dist.log_prob(sample).argmax(dim=0)
                y_hat_max = torch.zeros((len(batch_x), model.features)).to(batch_x.device)
                for i in range(len(argmax)):
                    y_hat_max[i] = sample[argmax[i], i, :]
                test_mae += torch.abs(batch_y - y_hat_max).sum().item()
                test_likelihood += - cond_dist.log_prob(batch_y).sum().item()
                test_crps += crps_ensemble(batch_y.squeeze(-1).cpu(),
                                           sample.squeeze(-1).permute(1, 0).cpu()).sum().item()
        test_mae /= len(dataset.x_test)
        test_likelihood /= len(dataset.x_test)
        test_crps /= len(dataset.x_test)
        test_maes.append(test_mae)
        test_likelihoods.append(test_likelihood)
        test_crpss.append(test_crps)
        iterator.set_postfix({"train loss": "%1.4f"%epoch_loss, "test mae": "%1.4f"%test_mae, "test lh": "%1.4f"%test_likelihood})
    return test_maes, test_likelihoods, test_crpss


if __name__ == '__main__':
    import yaml
    import numpy as np
    from core.helper_functions import get_dataset_by_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = "splice"
    with open(f"configs/{dataset}.yaml", 'r') as f:
        config = yaml.load(f, yaml.Loader)
    DatasetClass = get_dataset_by_name(dataset)
    DatasetClass.inject_config(config)
    pool_rng = np.random.default_rng(1)
    dataset = DatasetClass("../datasets", config, pool_rng, encoded=0)
    dataset = dataset.to(device)
    model_rng = torch.Generator()
    model_rng.manual_seed(1)
    accs = fit_and_evaluate(dataset, model_rng)
    import matplotlib.pyplot as plt
    plt.plot(accs)

