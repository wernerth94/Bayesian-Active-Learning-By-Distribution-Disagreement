import experiment_util
import argparse
import os
from os.path import *
import yaml
import numpy as np
from core.helper_functions import get_dataset_by_name
from raytune import pretext_encoder, embedded_classification, classification

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, default="../datasets")
parser.add_argument('--dataset', type=str, default="superconductors")
parser.add_argument("--variation", type=str, default="gauss")
parser.add_argument('--task', type=str, default="classification")
parser.add_argument('--num_trials', type=int, default=2)
parser.add_argument('--max_conc_trials', type=int, default=4)


if __name__ == '__main__':
    args = parser.parse_args()

    if experiment_util.is_cluster:
        project_folder = "normalizing_flows_al"
    else:
        project_folder = "nf_active_learning"
    base_path = os.path.split(os.getcwd())[0]
    cache_folder = join(base_path, "datasets")

    config_name = args.dataset.lower()
    if args.variation != "nf":
        config_name = f"{config_name}_{args.variation}"
    config_file = join(base_path, project_folder, f"configs/{config_name}.yaml")
    with open(config_file, 'r') as f:
        config = yaml.load(f, yaml.Loader)
    # check the dataset
    DatasetClass = get_dataset_by_name(args.dataset)
    dataset = DatasetClass(cache_folder, config, np.random.default_rng(1), encoded=False)
    # output
    if args.variation != "nf":
        output_folder = f"raytune_output_{args.variation}"
    else:
        output_folder = "raytune_output"
    output_folder = join(base_path, project_folder, output_folder)
    log_folder = join(output_folder, dataset.name)
    os.makedirs(log_folder, exist_ok=True)

    if args.task == "pretext":
        raise NotImplementedError()
        pretext_encoder.tune_pretext(args.num_trials, args.max_conc_trials, cache_folder, join(base_path, project_folder), log_folder, args.dataset)
    elif args.task == "embedded_classification":
        raise NotImplementedError()
        embedded_classification.tune_encoded_classification(args.num_trials, args.max_conc_trials, log_folder, config_file, cache_folder, DatasetClass, join(base_path, project_folder))
    elif args.task == "classification":
        classification.tune_classification(args.num_trials, args.max_conc_trials, log_folder, config_file, cache_folder, DatasetClass, join(base_path, project_folder))
    else:
        raise NotImplementedError
