from os.path import join
from functools import partial
from datetime import datetime
import numpy as np
import yaml
from tqdm import tqdm
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from properscoring import crps_ensemble

def subsample_dataset(dataset, size):
    assert size < len(dataset.x_train)
    ids = np.random.choice(len(dataset.x_train), size, replace=False)
    dataset.x_train = dataset.x_train[ids]
    dataset.y_train = dataset.y_train[ids]
    return dataset

def evaluate_classification_config(raytune_config, DatasetClass, config_file, cache_folder, benchmark_folder):
    with open(config_file, 'r') as f:
        config = yaml.load(f, yaml.Loader)

    # hidden_dims = [raytune_config["h1"]]
    # if raytune_config["h2"] > 0:
    #     hidden_dims.append(raytune_config["h2"])
    # config["classifier"]["hidden"] = hidden_dims
    # config["optimizer"]["type"] = raytune_config["type"]
    config["optimizer"]["lr"] = raytune_config["lr"]
    config["optimizer"]["weight_decay"] = raytune_config["weight_decay"]
    config["classifier"]["dropout"] = raytune_config["dropout"]

    like_sum = 0.0
    mae_sum = 0.0
    crps_sum = 0.0
    restarts = 4
    for i in range(restarts):
        pool_rng = np.random.default_rng(i)
        model_rng = torch.Generator()
        model_rng.manual_seed(i)
        dataset = DatasetClass(cache_folder, config, pool_rng, encoded=False)
        dataset = subsample_dataset(dataset, config["dataset"]["budget"] + config["dataset"]["initial_points_per_class"])
        model = dataset.get_classifier(model_rng)
        optimizer = dataset.get_optimizer(model)
        batch_size = dataset.classifier_batch_size

        data_loader_rng = torch.Generator()
        data_loader_rng.manual_seed(i)
        train_dataloader = DataLoader(TensorDataset(dataset.x_train, dataset.y_train),
                                      batch_size=batch_size,
                                      generator=data_loader_rng,
                                      shuffle=True, num_workers=2)
        test_dataloader = DataLoader(TensorDataset(dataset.x_test, dataset.y_test), batch_size=512,
                                     num_workers=4)

        # early_stop = EarlyStopping(patience=40)
        MAX_EPOCHS = 400
        TESTING_EPOCHS = 10
        avrg_test_like = 0.0
        avrg_test_mae = 0.0
        avrg_test_crps = 0.0
        for e in range(MAX_EPOCHS):
            for batch_x, batch_y in train_dataloader:
                optimizer.zero_grad()
                loss_value = - model(batch_x).log_prob(batch_y).mean()
                loss_value.backward()
                optimizer.step()
            if e >= MAX_EPOCHS - TESTING_EPOCHS:
                with torch.no_grad():
                    test_like = 0.0
                    test_mae = 0.0
                    test_crps = 0.0
                    total = 0.0
                    for batch_x, batch_y in test_dataloader:
                        total += batch_y.size(0)
                        cond_dist = model(batch_x)
                        sample = cond_dist.sample((64,))
                        y_hat = model.predict(batch_x)
                        test_mae += torch.abs(batch_y - y_hat).sum().item()
                        test_like += cond_dist.log_prob(batch_y).mean()
                        test_crps += crps_ensemble(batch_y.squeeze(-1), sample.squeeze(-1).permute(1, 0)).sum().item()
                    avrg_test_mae +=  test_mae / total
                    avrg_test_like += test_like / total
                    avrg_test_crps += test_crps / total
        like_sum += avrg_test_like / TESTING_EPOCHS
        mae_sum += avrg_test_mae  / TESTING_EPOCHS
        crps_sum += avrg_test_crps / TESTING_EPOCHS
    like_sum = float(like_sum/restarts)
    mae_sum = float(mae_sum/restarts)
    crps_sum = float(crps_sum/restarts)
    tune.report(likelihood=like_sum, mae=mae_sum, crps=crps_sum)


def tune_classification(num_samples, max_conc_trials, log_folder, config_file, cache_folder, DatasetClass, benchmark_folder):
    log_folder = join(log_folder, "classification")

    hp_space = {
        # "type": tune.choice(["NAdam", "Adam", "SGD"]),
        # "lr": tune.loguniform(1e-6, 1e-2),
        # "weight_decay": tune.loguniform(1e-6, 1e-2),
        "lr": tune.uniform(1e-6, 1e-3),
        "weight_decay": tune.uniform(1e-6, 1e-2),
        "dropout": tune.uniform(0.0, 0.3),
    }

    # fixes some parameters of the function
    evaluate_config = partial(evaluate_classification_config,
                              DatasetClass=DatasetClass,
                              config_file=config_file,
                              cache_folder=cache_folder,
                              benchmark_folder=benchmark_folder)

    algo = BayesOptSearch(utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0})
    analysis = tune.run(evaluate_config,
                        config=hp_space,
                        num_samples=num_samples,
                        metric="mae",
                        mode="min",
                        search_alg=algo,
                        local_dir=log_folder,
                        max_concurrent_trials=max_conc_trials,
                        verbose=1)
    df = analysis.dataframe()
    timestamp = str(datetime.now())[:-7]
    df.to_csv(join(log_folder, f"ray_tune_results_{timestamp}.csv"))

