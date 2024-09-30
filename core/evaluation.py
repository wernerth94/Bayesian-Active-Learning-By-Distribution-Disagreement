import itertools, math
import os
from os.path import exists, join
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

name_corrections = {
    "RandomAgent": "Random",
    "Coreset_Greedy": "Coreset",
    "ShannonEntropy": "Entropy",
    "MarginScore": "Margin"
}

def sort_according_to_reference(unsorted_list:list, reference:list):
    result = []
    for entry in reference:
        if entry in unsorted_list:
            result.append(entry)
    return result

def _query_to_list(query, current_folder):
    if query is None:
        result_list = list(os.listdir(current_folder))
    elif isinstance(query, list):
        result_list = query
    elif isinstance(query, str):
        result_list = [query]
    else:
        raise ValueError(f"Query not recognized: {query}")
    return result_list


def generate_rank_leaderboard(precision=1, add_std=True, subsample_runs=None):
    datasets_raw = ["Splice", "DNA", "USPS", "Cifar10", "FashionMnist", "TopV2", "News",]
                    #"DivergingSin", "ThreeClust"]
    datasets_encoded = ["SpliceEncoded", "DNAEncoded", "USPSEncoded",
                        "Cifar10Encoded", "FashionMnistEncoded"]
    df_raw = combine_agents_into_df(dataset=datasets_raw, include_oracle=True, subsample_runs=subsample_runs)
    # df_raw = _insert_oracle_forecast(df_raw)
    df_raw = average_out_columns(df_raw, ["iteration"])
    df_raw = average_out_columns(df_raw, ["query_size"])
    df_raw = compute_ranks_over_trials(df_raw)
    print("range of stds", df_raw["rank_std"].min(), "-", df_raw["rank_std"].max())

    df_enc = combine_agents_into_df(dataset=datasets_encoded, include_oracle=True, subsample_runs=subsample_runs)
    df_enc = average_out_columns(df_enc, ["iteration", "query_size"])
    df_enc = compute_ranks_over_trials(df_enc)

    leaderboard = average_out_columns(df_raw, ["dataset"]).sort_values("rank")

    intersection = leaderboard["agent"].isin(df_enc["agent"])
    leaderboard = leaderboard[intersection]
    leaderboard.index = leaderboard["agent"]
    leaderboard = leaderboard.drop(["agent", "auc"], axis=1)
    # add single unencoded datasets
    for dataset in datasets_raw:
        values = []
        for index, _ in leaderboard.iterrows():
            sub_df = df_raw[(df_raw["agent"] == index) & (df_raw["dataset"] == dataset)]
            r = sub_df["rank"]
            r_std = sub_df["rank_std"]
            if len(r) == 0:
                print(f"No runs found for {index} on {dataset}")
                continue
            if add_std:
                values.append(f"{round(r.item(), precision)}+-{round(r_std.item(), 2)}")
            else:
                values.append(round(r.item(), precision))
        leaderboard[dataset] = values
    leaderboard["Unencoded"] = leaderboard["rank"].round(precision)
    # leaderboard["std"] = leaderboard["rank_std"].round(3) # not correct, we need to recompute the std, not just take the averages
    leaderboard = leaderboard.drop(["rank"], axis=1)
    leaderboard = leaderboard.drop(["rank_std"], axis=1)
    # add average of all encoded datasets
    df_enc = average_out_columns(df_enc, ["dataset"])
    values = []
    for index, _ in leaderboard.iterrows():
        r = df_enc[(df_enc["agent"] == index)]["rank"]
        values.append(round(r.item(), precision))
    leaderboard["Encoded"] = values
    leaderboard.to_csv("results/rank.csv")


def generate_auc_leaderboard(sorted_agents:list, precision=3, add_std=True, subsample_runs=None):
    def save_leaderboard(df, dataset_list, postfix):
        df["query_size"] = df["query_size"].astype(int)
        query_sizes = pd.unique(df["query_size"])
        for q in query_sizes:
            sub_df = df[df["query_size"] == q]
            sub_df = sub_df.drop("query_size", axis=1)
            available_datasets = pd.unique(sub_df["dataset"])
            available_datasets = sort_according_to_reference(list(available_datasets), dataset_list)
            available_agents = pd.unique(sub_df["agent"])
            available_agents = sort_according_to_reference(list(available_agents), sorted_agents)
            result_df = pd.DataFrame(columns=["agent"]+available_datasets)
            for agent in available_agents:
                auc_values = sub_df[sub_df["agent"] == agent]
                auc_values.index = auc_values["dataset"]
                auc_values = auc_values.drop(["agent", "dataset"], axis=1)
                auc_values = auc_values.transpose()
                auc_values["agent"] = agent
                result_df = pd.concat([result_df, auc_values])
            result_df.to_csv(f"results/auc{postfix}_qs{q}.csv")

    datasets_raw = ["Splice", "SpliceEncoded", "DNA", "DNAEncoded", "USPS", "USPSEncoded",
                    "Cifar10", "Cifar10Encoded", "FashionMnist", "FashionMnistEncoded",
                    "TopV2", "News",
                    "DivergingSin", "ThreeClust"]
    df_raw = combine_agents_into_df(dataset=datasets_raw, include_oracle=True, subsample_runs=subsample_runs)
    df_raw = average_out_columns(df_raw, ["iteration"])
    df_raw = std_for_column(df_raw, "auc")
    df_raw = average_out_columns(df_raw, ["trial"])
    df_raw["auc"] = df_raw["auc"].round(precision).astype(str) + "+-" + df_raw["auc_std"].round(precision).astype(str)
    df_raw = df_raw.drop("auc_std", axis=1)
    save_leaderboard(df_raw, datasets_raw, postfix="")

    # datasets_encoded = [
    #                     ]
    # df_enc = combine_agents_into_df(dataset=datasets_encoded, include_oracle=True, subsample_runs=subsample_runs)
    # df_enc = average_out_columns(df_enc, ["iteration"])
    # df_enc = std_for_column(df_enc, "auc")
    # df_enc = average_out_columns(df_enc, ["trial"])
    # df_enc["auc"] = df_enc["auc"].round(precision).astype(str) + "+-" + df_enc["auc_std"].round(precision).astype(str)
    # df_enc = df_enc.drop("auc_std", axis=1)
    # save_leaderboard(df_enc, datasets_encoded, postfix="_enc")



def compute_ranks_over_trials(df:pd.DataFrame):
    assert "trial" in df.columns
    df["rank"] = df.groupby(["dataset", "trial"])["auc"].rank(ascending=False)
    df = std_for_column(df, "rank")
    df = average_out_columns(df, ["trial"])
    return df



def combine_agents_into_df(dataset=None, query_size=None, agent=None,
                           max_loaded_runs=30, include_oracle=False,
                           subsample_runs=None, base_folder="runs", results_file="accuracies.csv"):
    def _load_trials_for_agent(dataset_name, query_size_name, agent_name):
        if query_size_name is not None:
            agent_folder = join(base_folder, dataset_name, query_size_name, agent_name)
        else:
            agent_folder = join(base_folder, dataset_name, agent_name)
        acc_file = join(agent_folder, results_file)
        if exists(acc_file):
            accuracies = pd.read_csv(acc_file, header=0, index_col=0).values
            if max_loaded_runs is not None:
                N = max_loaded_runs
            else:
                N = accuracies.shape[1]
            if subsample_runs is None:
                trials = range(N)
            else:
                # randomly chose a subset of runs
                trials = np.random.choice(N, subsample_runs, replace=False)
            # fill in runs for Oracle for the CD diagram
            while agent_name == "Oracle" and accuracies.shape[1] < N:
                diff = N - accuracies.shape[1]
                accuracies = np.concatenate([accuracies, accuracies[:, :diff]], axis=1)
            for i, trial in enumerate(trials):
                for iteration in range(accuracies.shape[0]):
                    if not np.isnan(accuracies[iteration, trial]):
                        df_data["dataset"].append(dataset_name)
                        if query_size_name is not None:
                            df_data["query_size"].append(query_size_name)
                        else:
                            df_data["query_size"].append(1)
                        df_data["agent"].append(name_corrections.get(agent_name, agent_name))
                        if subsample_runs:
                            df_data["trial"].append(i)
                        else:
                            df_data["trial"].append(trial)
                        df_data["iteration"].append(iteration)
                        df_data["auc"].append(accuracies[iteration, trial])



    df_data = {
        "dataset": [],
        "query_size": [],
        "agent": [],
        "trial": [],
        "iteration": [],
        "auc": []
    }
    dataset_list = _query_to_list(dataset, base_folder)
    for dataset_name in tqdm(dataset_list):
        dataset_folder = join(base_folder, dataset_name)
        query_size_list = _query_to_list(query_size, dataset_folder)
        for query_size_name in query_size_list:
            if query_size_name in ["UpperBound", "Oracle"]:
                continue
            query_size_folder = join(dataset_folder, query_size_name)
            agent_list = _query_to_list(agent, query_size_folder)
            for agent_name in agent_list:
                _load_trials_for_agent(dataset_name, query_size_name, agent_name)
        if include_oracle:
            _load_trials_for_agent(dataset_name, None, "Oracle")
    df = pd.DataFrame(df_data)
    df = df.sort_values(["dataset", "query_size", "agent", "trial", "iteration"])
    return df


def average_out_columns(df:pd.DataFrame, columns:list):
    result_df = df.copy(deep=True)
    for col in columns:
        other_columns = [c for c in result_df.columns if c not in [col, "auc", "rank", "rank_std"] ]
        result_list = []
        grouped_df = result_df.groupby(other_columns)
        for key, sub_df in grouped_df:
            mean = sub_df["auc"].mean()
            sub_df["auc"] = mean
            if "rank" in df.columns:
                mean = sub_df["rank"].mean()
                sub_df["rank"] = mean
            sub_df = sub_df.drop(col, axis=1) # drop averaged col from sub-dataframe
            sub_df = sub_df.drop(sub_df.index[1:]) # drop all the other useless rows
            result_list.append(sub_df)
        result_df = pd.concat(result_list)
    return result_df

def std_for_column(df:pd.DataFrame, column:str):
    result_df = df.copy(deep=True)
    other_columns = [c for c in result_df.columns if c not in [column, "trial", "auc", "rank"] ]
    result_list = []
    grouped_df = result_df.groupby(other_columns)
    for key, sub_df in grouped_df:
        std = sub_df["auc"].std()
        sub_df[f"{column}_std"] = std
        # if "rank" in df.columns:
        #     mean = sub_df["rank"].mean()
        #     sub_df["rank"] = mean
        # sub_df = sub_df.drop(column, axis=1) # drop averaged col from sub-dataframe
        # sub_df = sub_df.drop(sub_df.index[1:]) # drop all the other useless rows
        result_list.append(sub_df)
    result_df = pd.concat(result_list)
    return result_df

def find_missing_runs(num_trails=30):
    for run_folder in ["runs", "runs_gauss"]:
        if not exists(run_folder):
            print(run_folder, "not found")
        else:
            for dataset_folder in os.listdir(run_folder):
                dataset_folder_path = join(run_folder, dataset_folder)
                for query_size in os.listdir(dataset_folder_path):
                    if query_size not in ["UpperBound", "Oracle"]:
                        query_size_folder_path = join(dataset_folder_path, query_size)
                        for agent in os.listdir(query_size_folder_path):
                            agent_path = join(query_size_folder_path, agent, "accuracies.csv")
                            if not exists(agent_path):
                                print("Acc file not found for", agent_path)
                            else:
                                accuracies = pd.read_csv(agent_path, header=0, index_col=0).values
                                rows, cols = accuracies.shape
                                if cols < num_trails:
                                    print("not enough runs for", agent_path)

if __name__ == '__main__':
    find_missing_runs()
    # run = "runs/Splice"
    # for d in ["Diamonds", "Pakinsons", "Sarcos", "Superconductors"]:
    #     acc_file = join("runs", d, "UpperBound", "accuracies.csv")
    #     acc_nf = pd.read_csv(acc_file, header=0, index_col=0).values.mean()
    #     acc_file = join("runs_gauss", d, "UpperBound", "accuracies.csv")
    #     acc_gnn = pd.read_csv(acc_file, header=0, index_col=0).values.mean()
    #     print(d, acc_nf, acc_gnn)
        # df = average_out_columns(df, ["iteration"])
    # df = df.drop("dataset", axis=1)
    # from core.helper_functions import collect_results
    # collect_results("runs/Sarcos/1/BALD", "run")

