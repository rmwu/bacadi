import os
from jax import jit, vmap
import jax.numpy as jnp
import pandas as pd
from bacadi.eval.metrics import expected_shd, expected_sid, kl_divergence, threshold_metrics, neg_ave_log_likelihood, neg_ave_log_marginal_likelihood
from bacadi.utils.func import joint_dist_to_marginal
from bacadi.eval.target import create_data_interv_tuples, save_pickle

STORE_ROOT = ['results']


def get_metrics(descr,
                dist,
                target,
                args,
                bacadi=None,
                use_cpdag=False,
                final=False):
    n_vars = target.n_vars
    metrics = {}

    # get targets used for inference to get interventional cpdag

    # I deleted the chunk that computes NLL and KL  because I do not care
    # about them

    dist_marginal = joint_dist_to_marginal(dist) if len(dist) >= 3 else dist

    if args.interv_data or args.infer_interv:
        interv_targets_train = target.interv_targets
        if (interv_targets_train[0] == 0).all():
            interv_targets_train = interv_targets_train[1:]
    else:
        interv_targets_train = jnp.zeros((1, n_vars))
    interv_target_list = [
        set([i_ for i_ in range(n_vars) if t[i_]])
        for t in interv_targets_train
    ]

    if final:
        esid = expected_sid(dist=dist_marginal,
                            g=target.g,
                            use_cpdag=use_cpdag,
                            interv_targets=interv_target_list)
        metrics[descr + 'esid'] = esid

    eshd = expected_shd(dist=dist_marginal,
                        g=target.g,
                        use_cpdag=use_cpdag,
                        interv_targets=interv_target_list)

    if final:
        thresh_metr = threshold_metrics(
            dist=dist_marginal,
            g=target.g,
            undirected_cpdag_oriented_correctly=use_cpdag)
    else:
        thresh_metr = threshold_metrics(dist=dist_marginal, g=target.g)

    metrics[descr + 'eshd'] = eshd
    metrics[descr + 'auroc'] = thresh_metr['roc_auc']
    metrics[descr + 'auprc'] = thresh_metr['prc_auc']
    metrics[descr + 'avgprec'] = thresh_metr['ave_prec']
    if args.infer_interv:
        interv_targets = target.interv_targets
        if (interv_targets[0] == 0).all():
            interv_targets = interv_targets[1:]
        metrics_i = threshold_metrics(dist=(dist[-2], dist[-1]),
                                      g=interv_targets,
                                      is_graph_distr=False)

        metrics[descr + 'intv_auroc'] = metrics_i['roc_auc']
        metrics[descr + 'intv_auprc'] = metrics_i['prc_auc']
        metrics[descr + 'intv_avgprec'] = metrics_i['ave_prec']
    return metrics


def process_incoming_result(result_df, incoming_dfs, args):
    """
    Appends new pd.DataFrame and saves the result in the existing .csv file
    """

    # concatenate existing and new results
    result_df = pd.concat([result_df, incoming_dfs], ignore_index=True)

    save_path = os.path.join(*STORE_ROOT)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # save to csv
    save_path = os.path.abspath(os.path.join(save_path, args.descr + '.csv'))
    result_df.to_csv(save_path)

    return result_df
