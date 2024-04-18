import os
import sys
import csv
import json
import time
from collections import defaultdict

from tqdm import tqdm
from util import Logger, NumpyArrayEncoder

os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # avoids jax gpu warning
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision
# os.environ['JAX_DEBUG_NANS'] = 'True'  # debugs NaNs
# os.environ['JAX_DISABLE_JIT'] = 'True'  # disables jit for debugging

import warnings

# ??
warnings.filterwarnings("ignore", message="CUDA_ERROR_NO_DEVICE")
warnings.filterwarnings("ignore", message="No GPU/TPU found")

import numpy as np
import jax.numpy as jnp


from eval import eval_single_target
from parser import make_evaluation_parser
from bacadi.eval.target import Target


# from config.svgd import marginal_config, joint_config

#import wandb


def write_json(fp, data):
    with open(fp, 'w+') as f:
        for item in data:
            json.dump(item, f)
            f.write(os.linesep)


def read_csv(fp, delimiter=','):
    data = []
    with open(fp) as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for item in reader:
            for k,v in item.items():
                item[k] = v
            data.append(item)
    return data


def make_target(fp_data, fp_graph, fp_interv):
    """
    Subsample to 200 observational, 20 per interventional context, as per
    the paper
    """
    x_interv_data = np.load(fp_data)
    graph = np.load(fp_graph)

    with open(fp_interv) as f:
        # if >1 node intervened, formatted as a list
        lines = [line.strip() for line in f.readlines()]
    regimes = [tuple(sorted(int(x) for x in line.split(",")))
            if len(line) > 0 else () for line in lines]
    assert len(regimes) == len(x_interv_data)

    # get unique and map to nodes
    unique_regimes = sorted(set(regimes))  # 0 is obs because () first
    idx_to_regime = {i: reg for i, reg in enumerate(unique_regimes)}
    regime_to_idx = {reg: i for i, reg in enumerate(unique_regimes)}
    # convert to regime label tensor
    num_vars = x_interv_data.shape[1]
    args.n_vars = num_vars  # some use this version...
    interv_targets = np.zeros((len(unique_regimes), num_vars)).astype(int)
    for idx, regime in idx_to_regime.items():
        for node in regime:
            interv_targets[idx, node] = 1
    # expand to n_obs
    envs = [regime_to_idx[r] for r in regimes]

    # sampling!
    n_obs = 200
    n_int = 20
    idxs_to_keep = []
    env_to_idx = defaultdict(int)
    for idx, env in enumerate(envs):
        if env == 0:
            if env_to_idx[env] < n_obs:
                env_to_idx[env] += 1
                idxs_to_keep.append(idx)
        else:
            if env_to_idx[env] < n_int:
                env_to_idx[env] += 1
                idxs_to_keep.append(idx)

    envs = np.array([envs[i] for i in idxs_to_keep])
    x_interv_data = x_interv_data[idxs_to_keep, :]

    # bacadi.fit() only requires
    # - target.x_interv_data
    # - target.envs
    # eval requires
    # - target.g
    # - target.n_vars
    # - target.interv_targets
    target = Target(
        x_interv_data=x_interv_data,
        n_vars=num_vars,
        envs=envs,
        g=graph,
        interv_targets=interv_targets,
        n_observations=None,
        n_ho_observations=None,
        theta=None,
        x=None,
        x_ho=None,
        x_interv=None,
        interv_factors=None,
        x_ho_interv=None,
        x_ho_interv_data=None,
        interv_targets_ho=None,
        envs_ho=None,
        gt_posterior_obs=None,
        gt_posterior_interv=None)

    return target


def main(fp_template, fp_data, fp_graph, fp_interv, args):
    # if we use sergio simulator, force prior to be scale-free
    if args.simulator == 'sergio':
        args.graph_prior = 'sf'

    target = make_target(fp_data, fp_graph, fp_interv)

    # results is a dict that contains basically everything we're interested
    # including both metrics and predictions (around line 200)
    start = time.time()
    results = eval_single_target(target=target, args=args)
    end = time.time()

    # split into two separate results files for easier loading/comparison
    for setting in ["empirical", "mixture"]:
        our_results = {
            "true": results["g_gt"],
            "pred": results[f"g_{setting}"],
            "true_interv": results["interv_targets_gt"],
            "pred_interv": results[f"I_{setting}"],
            "shd": results["evals"][f"{setting}_eshd"],
            "auroc": results["evals"][f"{setting}_auroc"],
            "auprc": results["evals"][f"{setting}_auprc"],
            "interv_auroc": results["evals"][f"{setting}_intv_auroc"],
            "interv_auprc": results["evals"][f"{setting}_intv_auprc"],
            "time": end - start
        }
        for k, v in our_results.items():
            if isinstance(v, np.ndarray):
                our_results[k] = v.tolist()
            elif isinstance(v, jnp.ndarray):
                our_results[k] = v.tolist()
        print(our_results)
        fp_out = fp_template.format(setting)
        write_json(fp_out, [our_results])
        print(fp_out, "done")


if __name__ == '__main__':
    # I kept original main() and added my extra logic here

    parser = make_evaluation_parser()
    args = parser.parse_args()

    # manually set variables for my experiments
    args.model_seed = 0  # why do they need two seeds???
    args.seed = 0

    args.interv_data = True
    args.infer_interv = True

    graph_prior = "er"
    args.graph_prior = graph_prior
    args.bacadi_graph_prior = graph_prior
    args.joint_bacadi_graph_prior = graph_prior
    args.joint = True

    fp = "/data/rsg/chemistry/rmwu/src/sandbox/causal-target/data/test_180.csv"
    items_to_load = read_csv(fp)

    # save results here
    exp_root = "/data/scratch/rmwu/cache/causal_results/bacadi"
    # >>> uncomment desired setting
    items_to_load = items_to_load[0:50]
    #items_to_load = items_to_load[50:100]
    #items_to_load = items_to_load[100:150]
    #items_to_load = items_to_load[150:]
    items_to_load = items_to_load[::-1]
    # <<<
    # iterate through our test set
    for item in tqdm(items_to_load):
        if item["split"] != "test":
            continue
        # load data and filter
        fp_data = item["fp_data"]
        key = fp_data.split("/")[-2]
        # use all of Sachs
        if "sachs" not in item["fp_data"]:
            nodes = int(key.split("_")[0][1:])
            edges = int(key.split("_")[1][1:])
        else:
            key = "sachs"
        # check if output exists
        dataset_id = item["fp_data"].split("data_interv")[1].split(".")[0]

        # remove .npy filename
        data_path = item["fp_data"].rsplit("/", 1)[0]

        fp_template = f"{exp_root}-{{}}/{key}_{dataset_id}.json"

        # check if we already ran this experiment
        exist_count = 0
        for setting in ["empirical", "mixture"]:
            fp_out = fp_template.format(setting)
            if os.path.exists(fp_out):
                exist_count += 1
        if exist_count == 2:
            print(fp_out, "done")
            continue

        try:
            print(f"working on {fp_template}")
            main(fp_template,
                 item["fp_data"],
                 item["fp_graph"],
                 item["fp_regime"],
                 args)
        except Exception as e:
            print(item["fp_data"], "CRASHED:", e)
            raise
            continue

