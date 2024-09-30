import functools
from typing import Callable
import os
from os.path import join, exists
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

class EarlyStopping:
    def __init__(self, patience=7, lower_is_better=True):
        self.patience = patience
        self.lower_is_better = lower_is_better
        self.best_loss = torch.inf if lower_is_better else -torch.inf
        self.steps_without_improvement = 0
    def check_stop(self, loss_val):
        if (self.lower_is_better     and loss_val >= self.best_loss) or \
           (not self.lower_is_better and loss_val <= self.best_loss):
            self.steps_without_improvement += 1
            if self.steps_without_improvement > self.patience:
                return True
        else:
            self.steps_without_improvement = 0
            self.best_loss = loss_val
        return False

def save_meta_data(logpath, agent, env, dataset, config:dict):
    if not os.path.exists(logpath):
        os.makedirs(logpath, exist_ok=True)
    file = os.path.join(logpath, "meta.txt")
    if os.path.exists(file):
        os.remove(file)

    with open(file, "w") as f:
        if hasattr(dataset, "get_meta_data"):
            f.write("# Dataset: \n")
            f.write(f"{dataset.get_meta_data()} \n\n")
        if hasattr(agent, "get_meta_data"):
            f.write("# Agent: \n")
            f.write(f"{agent.get_meta_data()} \n\n")
        if hasattr(env, "get_meta_data"):
            f.write("# Environment: \n")
            f.write(f"{env.get_meta_data()} \n\n")

        f.write("# Config: \n")
        for key, value in config.items():
            f.write(f"{key}: {value} \n")


def _pad_nans_with_last_value(df:pd.DataFrame):
    max_len = len(df)
    for col in df:
        diff = max_len - sum(pd.notna(df[col]))
        if diff > 0:
            last_val = df[col][sum(pd.notna(df[col])) - 1]
            df[col] = pd.concat([df[col].iloc[:-diff], pd.Series([last_val]*diff)], ignore_index=True)
    return df


initial_pool_size = {
    "Pakinsons": 200,
    "Superconductors": 200,
    "Diamonds": 200,
    "Sarcos": 200,
    "BiModalSin": 1,
    "DensityBiModalSin": 1,
    "DensityGrid": 1
}

def get_init_pool_size(dataset_agent:str):
    dataset = dataset_agent.split("/")[0]
    if dataset not in initial_pool_size:
        print(f"Dataset {dataset} has no initial pool size")
        return 0
    else:
        return initial_pool_size[dataset]

def _moving_avrg(line, weight):
    moving_mean = line[0]
    result = [moving_mean]
    for i in range(1, len(line)):
        moving_mean = weight * moving_mean + (1 - weight) * line[i]
        result.append(moving_mean)
    return np.array(result)


def moving_avrg(trajectory, weight):
    # moving average for a tuple of trajectory and std
    stdCurve = trajectory[1]
    trajectory = trajectory[0]
    return _moving_avrg(trajectory, weight), _moving_avrg(stdCurve, weight)


def plot_upper_bound(axes, dataset, budget, color, alpha=0.8, percentile=0.99, linewidth=2, run_name="UpperBound", plot_percentile=False):
    if len(axes) == 1:
        files = ["accuracies.csv"]
    elif len(axes) == 2:
        files = ["accuracies.csv", "likelihoods.csv"]
    else:
        raise ValueError("Bad number of axes")
    for ax, file_name in zip(axes, files):
        file = os.path.join("/home/thorben/phd/projects/nf_active_learning/runs", dataset, run_name, file_name)
        all_runs = pd.read_csv(file, header=0, index_col=0)
        mean = np.mean(all_runs.values, axis=1)
        mean = [mean[0]]*budget
        x = np.arange(budget) + get_init_pool_size(dataset)
        ax.plot(x, mean, label="Full Dataset", linewidth=linewidth, c=color, alpha=alpha)
        if plot_percentile:
            mean_percentile = percentile * mean
            mean_percentile = [mean_percentile[0]]*budget
            ax.plot(x, mean_percentile, label="99% Percentile", linewidth=1, linestyle='--', c=color, alpha=0.6)

def plot_benchmark(dataset, color, display_name, smoothing_weight=0.0, alpha=0.8, linewidth=1.5, plot_std=False, show_auc=False, run_folder="runs"):
    full_name = f"{display_name}"
    file = os.path.join("/home/thorben/phd/projects/nf_active_learning", run_folder, dataset, "accuracies.csv")
    all_runs = pd.read_csv(file, header=0, index_col=0)
    if show_auc:
        values = all_runs.values
        auc = np.sum(values, axis=0) / values.shape[0]
        full_name += " - AUC: %1.3f"%(np.median(auc).item())
    # mean = np.median(all_runs.values, axis=1)
    mean = np.mean(all_runs.values, axis=1)
    std = np.std(all_runs.values, axis=1)
    curve = np.stack([mean, std])
    if smoothing_weight > 0.0:
        avrg_curve, std_curve = moving_avrg(curve, smoothing_weight)
    else:
        avrg_curve, std_curve = mean, std
    x = np.arange(len(avrg_curve)) + get_init_pool_size(dataset)
    if plot_std:
        if show_auc:
            avrg_std = round(sum(std) / len(std), 3)
            full_name += f"+-{avrg_std}"
        plt.fill_between(x, avrg_curve-std_curve, avrg_curve+std_curve, alpha=0.5, facecolor=color)
    plt.plot(x, avrg_curve, label=full_name, linewidth=linewidth, c=color, alpha=alpha)
    return len(x)

def plot_batch_benchmark(axes, dataset, color, display_name, alpha=0.8, linewidth=1.5,
                         plot_std=False, show_auc=True, moving_avrg=0.0, run_folder="runs"):
    if len(axes) == 1:
        files = ["accuracies.csv"]
    elif len(axes) == 2:
        files = ["accuracies.csv", "likelihoods.csv"]
    else:
        raise ValueError("Bad number of axes")
    full_name = f"{display_name}"
    for file_name, ax in zip(files, axes):
        file = os.path.join("/home/thorben/phd/projects/nf_active_learning", run_folder, dataset, file_name)
        if exists(file):
            all_runs = pd.read_csv(file, header=0, index_col=0)
            all_runs = all_runs.dropna(axis=0)
            if show_auc:
                values = all_runs.values
                auc = np.sum(values, axis=0) / values.shape[0]
                full_name += " - AUC: %1.4f"%(np.mean(auc).item())
            x = list(all_runs.index)
            x = [i + get_init_pool_size(dataset) for i in x]
            mean = np.mean(all_runs.values, axis=1)
            if moving_avrg > 0.0:
                mean = _moving_avrg(mean, moving_avrg)
            std = np.std(all_runs.values, axis=1)
            if moving_avrg > 0.0:
                std = _moving_avrg(std, moving_avrg)
            if plot_std:
                ax.fill_between(x, mean-std, mean+std, alpha=0.5, facecolor=color)
            ax.plot(x, mean, label=full_name, linewidth=linewidth, c=color, alpha=alpha)
    return len(x)

def plot_learning_curves(list_of_accs:list, out_file:str=None, y_label="validation accuracy"):
    for accs in list_of_accs:
        x = range(len(accs))
        plt.plot(x, accs, alpha=0.7)

    plt.xlabel("epochs")
    plt.ylabel(y_label)
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file, dpi=100, bbox_inches='tight')
    plt.clf()

def sort_by_run_id(x, y):
    """
    custom comparator for sorting run folders with syntax 'run_<id>'
    """
    if "_" not in x and "_" not in x:
        return 0
    elif "_" not in x:
        return 1
    elif "_" not in y:
        return -1
    else:
        x_id = int(x.split("_")[-1])
        y_id = int(y.split("_")[-1])
        if x_id > y_id:
            return 1
        elif x_id < y_id:
            return -1
        else:
            return 0


def collect_results(base_path, folder_prefix):
    result_acc = pd.DataFrame()
    result_like = pd.DataFrame()
    result_crps = pd.DataFrame()
    runs = sorted(os.listdir(base_path), key=functools.cmp_to_key(sort_by_run_id))
    column_names = []
    for run_folder in runs:
        if run_folder.startswith(folder_prefix):
            column_names.append(run_folder)
            acc_file_path = join(base_path, run_folder, "accuracies.csv")
            if exists(acc_file_path):
                accuracies = pd.read_csv(acc_file_path, header=0, index_col=0)
                result_acc = pd.concat([result_acc, accuracies], axis=1, ignore_index=True)

            like_file_path = join(base_path, run_folder, "likelihoods.csv")
            if exists(acc_file_path):
                likelihoods = pd.read_csv(like_file_path, header=0, index_col=0)
                result_like = pd.concat([result_like, likelihoods], axis=1, ignore_index=True)

            loss_file_path = join(base_path, run_folder, "crps.csv")
            if exists(loss_file_path):
                losses = pd.read_csv(loss_file_path, header=0, index_col=0)
                result_crps = pd.concat([result_crps, losses], axis=1, ignore_index=True)

    result_acc.columns = column_names
    result_acc.to_csv(join(base_path, "accuracies.csv"))
    result_like.columns = column_names
    result_like.to_csv(join(base_path, "likelihoods.csv"))
    result_crps.columns = column_names
    result_crps.to_csv(join(base_path, "crps.csv"))


def get_dataset_by_name(name:str)->Callable:
    import datasets
    # Tabular
    name = name.lower()
    # Regression
    if name == 'pakinsons':
        return datasets.Pakinsons
    elif name == 'superconductors':
        return datasets.Superconductors
    elif name == 'diamonds':
        return datasets.Diamonds
    elif name == 'sarcos':
        return datasets.Sarcos
    # Toy
    elif name == 'threeclust':
        return datasets.ThreeClust
    elif name == 'divergingsin':
        return datasets.DivergingSin
    elif name == 'bimodalsin':
        return datasets.BiModalSin
    elif name == 'densitybimodalsin':
        return datasets.DensityBiModalSin
    elif name == 'densitygrid':
        return datasets.DensityGrid

    else:
        raise ValueError(f"Dataset name '{name}' not recognized")


def get_agent_by_name(name:str)->Callable:
    import agents
    name = name.lower()
    if name == "random":
        return agents.RandomAgent
    elif name == "entropy":
        return agents.ShannonEntropy
    elif name == "margin":
        return agents.MarginScore
    elif name == "coreset":
        return agents.Coreset_Greedy
    elif name == "badge":
        return agents.Badge
    elif name == "typiclust":
        return agents.TypiClust
    elif name == "coregcn":
        return agents.CoreGCN
    elif name == "dsa":
        return agents.DSA
    elif name == "lsa":
        return agents.LSA

    elif name == "bald_point":
        return agents.BALD_Point
    elif name == "bald_dist_entropy":
        return agents.BALD_Dist_Entropy
    elif name == "bald_dist_std":
        return agents.BALD_Dist_Std
    elif name == "nflows_out":
        return agents.NFlows_Out

    elif name == "balsa_entropy":
        return agents.BALSA_entropy
    elif name == "balsa_emd":
        return agents.BALSA_EMD
    elif name == "balsa_emd_dual":
        return agents.BALSA_EMD_dual
    elif name == "balsa_full":
        return agents.BALSA_EMD_full
    elif name == "balsa_kl_diff":
        return agents.BALSA_KL_Diff
    elif name == "balsa_kl_pairs":
        return agents.BALSA_KL_Pairs
    elif name == "balsa_kl_grid":
        return agents.BALSA_KL_Grid
    elif name == "balsa_kl_anomaly":
        return agents.BALSA_KL_Anomaly

    elif name == "balsa_kl_grid_norm":
        return agents.BALSA_KL_Grid_Norm
    elif name == "balsa_kl_grid_dual":
        return agents.BALSA_KL_Grid_dual
    elif name == "balsa_kl_pairs_dual":
        return agents.BALSA_KL_Pair_dual

    elif name == "nf_conf":
        return agents.NF_Conf
    elif name == "nf_entropy":
        return agents.NF_Entropy
    elif name == "nf_std":
        return agents.NF_Std
    elif name == "nf_sample_std":
        return agents.NF_Sample_Std
    elif name == "nf_density":
        return agents.NF_Density
    elif name == "nf_density_std":
        return agents.NF_Density_STD
    elif name == "nf_diff":
        return agents.NF_Diff
    elif name == "nf_proxy":
        return agents.NF_Proxy
    elif name == "gauss_std":
        return agents.GaussStd
    elif name == "gauss_entropy":
        return agents.GaussEntropy
    elif name == "gauss_lc":
        return agents.GaussLC
    else:
        raise ValueError(f"Agent name '{name}' not recognized")


def visualize_nf_prediction(x, log_probs, idx=0, plot_kde=False):
    from scipy.stats import kde
    plt.clf()
    if plot_kde:
        # k = kde.gaussian_kde(x.squeeze(-1).T, bw_method='scott')
        # plt.scatter(x.squeeze(-1).T, k, label="kde")
        plt.hist(x[:, idx, 0], density=True, bins=100, label="Kde", color="orange")
    plt.scatter(x[:, idx], log_probs[:, idx], label="output", s=2)
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    for d in ["Pakinsons", "Diamonds", "Superconductors", "Sarcos"]:
        for a in ["BALSA_LST", "BALSA_LST_Grid"]:
            collect_results(f"../runs/{d}/1/{a}", "run_")

    # fig, axes = plt.subplots(1,2)
    # plot_batch_benchmark(axes, "Pakinsons/5/RandomAgent", "b", "random")
    # plot_upper_bound(axes, "Pakinsons", 800, "black", percentile=0.99)
    # plt.show()

    # from os.path import exists, join
    # from os import listdir
    #
    # for dataset_name in listdir("../runs"):
    #     dataset = join("../runs", dataset_name)
    #     for size_name in listdir(dataset):
    #         query_size = join(dataset, size_name)
    #         for agent_name in listdir(query_size):
    #             agent = join(query_size, agent_name)
    #             collect_results(agent, "run_")
