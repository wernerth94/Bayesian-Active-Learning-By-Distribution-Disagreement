import yaml
import experiment_util as util
import argparse
from classifiers.classifier import fit_and_evaluate
from core.helper_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, default='../datasets')
parser.add_argument("--run_id", type=int, default=2)
parser.add_argument("--model_seed", type=int, default=1)
parser.add_argument("--dataset", type=str, default="superconductors")
parser.add_argument("--variation", type=str, default="gauss")
parser.add_argument("--encoded", type=int, default=0)
parser.add_argument("--restarts", type=int, default=15)
parser.add_argument("--patience", type=int, default=25)
args = parser.parse_args()
args.encoded = bool(args.encoded)

run_id = args.run_id
max_run_id = run_id + args.restarts
while run_id < max_run_id:
    config_name = args.dataset.lower()
    if args.variation != "nf":
        config_name = f"{config_name}_{args.variation}"
    with open(f"configs/{config_name}.yaml", 'r') as f:
        config = yaml.load(f, yaml.Loader)

    pool_rng = np.random.default_rng(run_id)
    model_rng = torch.Generator()
    model_rng.manual_seed(args.model_seed + run_id)
    # This is currently the only way to seed dropout layers in Python
    torch.random.manual_seed(args.model_seed + run_id)

    DatasetClass = get_dataset_by_name(args.dataset)
    dataset = DatasetClass(args.data_folder, config, pool_rng=pool_rng, encoded=args.encoded)
    dataset = dataset.to(util.device)

    runs = "runs"
    if args.variation != "nf":
        runs = f"{runs}_{args.variation}"
    os.makedirs(runs, exist_ok=True)
    base_path = os.path.join(runs, dataset.name, "UpperBound")
    log_path = os.path.join(base_path, f"run_{run_id}")

    print(f"starting run {run_id}")
    save_meta_data(log_path, None, None, dataset, config)

    # some convoluted saving to make it compatible with collect_results
    # and to be consistent with logging of other runs
    test_maes, test_likelihoods, test_crps = fit_and_evaluate(dataset, model_rng, patience=args.patience)
    data_dict = {"1": [test_maes[-1]]}
    df = pd.DataFrame(data_dict)
    df.to_csv(os.path.join(log_path, "accuracies.csv"))
    data_dict = {"1": [test_likelihoods[-1]]}
    df = pd.DataFrame(data_dict)
    df.to_csv(os.path.join(log_path, "likelihoods.csv"))
    data_dict = {"1": [test_crps[-1]]}
    df = pd.DataFrame(data_dict)
    df.to_csv(os.path.join(log_path, "crps.csv"))

    # save learning curves for training evaluation
    plot_learning_curves([test_crps], os.path.join(log_path, "test_crps.jpg"), y_label="CRPS")
    plot_learning_curves([test_likelihoods], os.path.join(log_path, "test_nll.jpg"), y_label="NLL")
    plot_learning_curves([test_maes], os.path.join(log_path, "test_mae.jpg"), y_label="MAE")

    # collect results from all runs
    collect_results(base_path, "run_")
    run_id += 1
