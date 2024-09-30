import torch
from torch.distributions import transforms
import torch.nn as nn
from zuko.flows.autoregressive import Flow, MaskedAutoregressiveTransform
from zuko.transforms import MonotonicRQSTransform
from zuko.flows.core import Unconditional
from zuko.distributions import DiagNormal
from classifiers.seeded_layers import SeededLinear

class InvertibleDropout(nn.Module):
    def __init__(self, model_rng, p, *args, **kwargs):
        # transforms.Transform.__init__(self)
        # nn.Module.__init__(self)
        super().__init__()

        if p < 0.0 or p > 1.0:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.model_rng = model_rng
        self.p = p

    def forward(self, x: torch.Tensor):
        return self._call(x)

    def _call(self, x: torch.Tensor):
        if self.training:
            # cleaning input
            trunc_mask = (x > 1e-5).float()
            drop_mask = torch.rand(x.size(), generator=self.model_rng) > self.p
            drop_mask = drop_mask.to(x.device).float()
            inv_mask = 1.0 - drop_mask
            orig_values = x * inv_mask
            output = x * trunc_mask
            output *= drop_mask.float()
            output += orig_values * 1e-6
        else:
            # During evaluation (testing), just scale the input tensor by (1 - p)
            output = x * (1 - self.p)
        return output

    def _inverse(self, y: torch.Tensor):
        mask = torch.ones_like(y).to(y.device).float()
        mask *= (y < 1e-5).float()
        mask *= (y > 0.0).float()
        inv_mask = 1.0 - mask
        output = inv_mask * y + mask * y * 1e6
        return output

    def log_abs_det_jacobian(self, x, y):
        mask = torch.ones_like(y).to(y.device).float()
        mask *= (y < 1e-5).float()
        mask *= (y > 0.0).float()
        inv_mask = 1.0 - mask
        return torch.linalg.det(inv_mask)


class LazyDropout(nn.Module):

    def __init__(self, model_rng:torch.Generator(), dropout):
        super().__init__()
        self.dropout = dropout
        self.model_rng = model_rng


    def forward(self, *args, **kwargs) -> nn.Module:
        d = InvertibleDropout(model_rng=self.model_rng, p=self.dropout)
        d.domain = torch.distributions.constraints.real
        d.codomain = torch.distributions.constraints.real
        return d





class MAF(Flow):
    r"""
    All credits to the Zuko Library
    https://zuko.readthedocs.io/en/stable/
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        randperm: bool = False,
        **kwargs,
    ):
        orders = [
            torch.arange(features),
            torch.flipud(torch.arange(features)),
        ]

        n_transforms = transforms
        transforms = []
        for i in range(n_transforms):
            transforms.append(
                MaskedAutoregressiveTransform(
                    features=features,
                    context=context,
                    order=torch.randperm(features) if randperm else orders[i % 2],
                    **kwargs,)
            )


        base = Unconditional(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
            buffer=True,
        )

        super().__init__(transforms, base)


class NeuralSplineFlow(MAF):
    """
    All credits to the Zuko Library
    https://zuko.readthedocs.io/en/stable/
    """
    def __init__(
            self,
            features: int,
            context: int = 0,
            bins: int = 8,
            **kwargs,
    ):
        super().__init__(
            features=features,
            context=context,
            univariate=MonotonicRQSTransform,
            shapes=[(bins,), (bins,), (bins - 1,)],
            **kwargs,
        )


class EncNSF(nn.Module):
    def __init__(self, model_rng, features: int, context: int, hidden_sizes:list, dropout: float = 0.0,):
        super().__init__()
        self.features = features
        self.inpt = SeededLinear(model_rng, context, hidden_sizes[0])
        self.hidden = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            self.hidden.append(SeededLinear(model_rng, hidden_sizes[max(0, i - 1)], hidden_sizes[i]))
            self.hidden.append(nn.Dropout(p=dropout))

        self.flow_head = NeuralSplineFlow(features, hidden_sizes[-1],
                                          transforms=2, activation=torch.nn.ReLU)

    def _encode(self, x:torch.Tensor):
        x = self.inpt(x)
        for layer in self.hidden:
            x = layer(x)
        return x

    def forward(self, x:torch.Tensor):
        x = self._encode(x)
        cond_dist = self.flow_head(x)
        return cond_dist

    def predict(self, x:torch.Tensor, num_samples=64):
        x = self._encode(x)
        cond_dist = self.flow_head(x)
        y_hat = cond_dist.sample((num_samples,))
        argmax = cond_dist.log_prob(y_hat).argmax(dim=0)
        y_hat_max = torch.zeros((len(x), self.features)).to(x.device)
        for i in range(len(argmax)):
            y_hat_max[i] = y_hat[argmax[i], i, :]
        return y_hat_max


if __name__ == '__main__':
    import yaml
    import numpy as np
    from datasets.pakinsons import Pakinsons
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    torch.autograd.set_detect_anomaly(True)
    pool_rng = np.random.default_rng(1)
    model_rng = torch.Generator()
    model_rng.manual_seed(1)
    # lazy_d = InvertibleDropout(model_rng, 0.8)
    # drop_layer = lazy_d()
    # inpt = torch.ones(200, 200)
    # output = d(inpt.clone())
    # rev = d._inverse(output)
    # print(torch.sum(inpt - rev).item())
    with open(f"../configs/pakinsons.yaml", 'r') as f:
        config = yaml.load(f, yaml.Loader)
    dataset = Pakinsons("../../datasets", config, pool_rng, 0)
    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(61, 16)
            # self.d1 = CustomDropout(p=0.2)
            self.d1 = InvertibleDropout(model_rng=model_rng, p=0.2)
            self.l2 = nn.Linear(16, 32)
            # self.d2 = CustomDropout(p=0.2)
            self.d2 = InvertibleDropout(model_rng=model_rng, p=0.2)
            self.l3 = nn.Linear(32, 1)
        def forward(self, x):
            x = self.l1(x)
            x = torch.nn.functional.relu(x)
            x = self.d1(x)
            x = self.l2(x)
            x = torch.nn.functional.relu(x)
            x = self.d2(x)
            x = self.l3(x)
            return x

    model = MyModel()
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters())
    train_loader = DataLoader(TensorDataset(dataset.x_train, dataset.y_train), batch_size=64)
    test_loader = DataLoader(TensorDataset(dataset.x_test, dataset.y_test), batch_size=256)
    for epoch in range(10):
        train_loss = 0.0
        test_loss = 0.0
        for (x, y) in train_loader:
            opt.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            train_loss += loss.item()
            loss.backward()
            opt.step()
        with torch.no_grad():
            for (x, y) in test_loader:
                output = model(x)
                loss = loss_fn(output, y)
                test_loss += loss.item()
        print(f"Epoch: {epoch} trainloss: {train_loss}, testloss {test_loss}")
