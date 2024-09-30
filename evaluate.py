import math
import os
import time

import torch

import experiment_util as util
import argparse
from pprint import pprint
from tqdm import tqdm
import core
import yaml
from core.helper_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, default="../datasets")
parser.add_argument("--run_id", type=int, default=1)
parser.add_argument("--agent_seed", type=int, default=1)
parser.add_argument("--pool_seed", type=int, default=1)
parser.add_argument("--model_seed", type=int, default=1)
parser.add_argument("--agent", type=str, default="coregcn")
parser.add_argument("--dataset", type=str, default="pakinsons") # superconductors pakinsons diamonds sarcos
parser.add_argument("--variation", type=str, default="nf")
parser.add_argument("--query_size", type=int, default=50)
parser.add_argument("--encoded", type=int, default=0)
parser.add_argument("--restarts", type=int, default=1)
parser.add_argument("--record_points", type=int, default=0)
##########################################################
args = parser.parse_args()
args.encoded = bool(args.encoded)
args.record_points = bool(args.record_points)

run_id = args.run_id
max_run_id = run_id + args.restarts
while run_id < max_run_id:
    config_name = args.dataset.lower()
    if args.variation != "nf":
        config_name = f"{config_name}_{args.variation}"
    with open(f"configs/{config_name}.yaml", 'r') as f:
        config = yaml.load(f, yaml.Loader)
    config["current_run_info"] = args.__dict__
    print("Config:")
    pprint(config)
    print("Config End \n")

    pool_rng = np.random.default_rng(args.pool_seed + run_id)
    model_seed = args.model_seed + run_id
    # This is currently the only way to seed dropout masks in Python
    torch.random.manual_seed(args.model_seed + run_id)
    # Seed numpy-based algorithms like KMeans
    np.random.seed(args.model_seed + run_id)
    data_loader_seed = 1

    AgentClass = get_agent_by_name(args.agent)
    DatasetClass = get_dataset_by_name(args.dataset)

    # Inject additional configuration into the dataset config (See BALD agent)
    AgentClass.inject_config(config)
    DatasetClass.inject_config(config)

    dataset = DatasetClass(args.data_folder, config, pool_rng, args.encoded)
    dataset = dataset.to(util.device)
    env = core.environment.FlowALGame(dataset,
                                      pool_rng,
                                      model_seed=model_seed,
                                      data_loader_seed=data_loader_seed,
                                      device=util.device)
    agent = AgentClass(args.agent_seed, config, args.query_size)

    runs = "runs"
    if args.variation != "nf":
        runs = f"{runs}_{args.variation}"
    os.makedirs(runs, exist_ok=True)

    base_path = os.path.join(runs, dataset.name, str(args.query_size), agent.name)
    log_path = os.path.join(base_path, f"run_{run_id}")

    print(f"Starting run {run_id}")
    time.sleep(0.1) # prevents printing uglyness with tqdm

    selected_points = torch.zeros([0, 2])
    with core.EnvironmentLogger(env, log_path, util.is_cluster) as env:
        done = False
        dataset.reset()
        state = env.reset()
        iterations = math.ceil(env.env.budget / args.query_size)
        iterator = tqdm(range(iterations), miniters=2)
        for i in iterator:
            action = agent.predict(*state)
            if args.record_points:
                points = torch.cat([dataset.x_unlabeled[action], dataset.y_unlabeled[action]], dim=-1)
                selected_points = torch.cat([selected_points, points.cpu()], dim=0)
            state, reward, done, truncated, info = env.step(action)
            iterator.set_postfix({"mae": env.accuracies[1][-1]})
            if done or truncated:
                # triggered when sampling batch_size is >1
                break

    # collect results from all runs
    collect_results(base_path, "run_")
    save_meta_data(log_path, agent, env, dataset, config)
    if args.record_points:
        torch.save(selected_points, os.path.join(log_path, "selected_points.pt"))
    run_id += 1
