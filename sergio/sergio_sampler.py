import math
import functools
import copy
from multiprocessing import Pool
from collections import defaultdict

import numpy as np

from sergio.sergio_mod import Sergio as SergioSimulator



def grn_sergio(
    *,
    spec,
    rng,
    g,
    effect_sgn,
    toporder,
    n_vars,
    # specific inputs required in config
    b,  # function
    k_param,  # function
    k_sign_p,  # function
    hill,  # float
    decays,  # float?
    noise_params,  # float?
    cell_types,  # int
    noise_type='dpd',
    sampling_state=15,
    dt=0.01,
    # technical noise NOTE: not used for BaCaDI
    tech_noise_config=None,
    add_outlier_effect=False,
    add_lib_size_effect=False,
    add_dropout_effect=False,
    return_count_data=False,
    # interventions
    n_ko_genes=0,  # unique genes knocked out in all of data collected; -1 indicates all genes
):
    """
    SERGIO simulator for GRNs
    """

    # sample interaction terms K
    # NOTE k_param = uniform [1,5]
    k = np.abs(k_param(rng, shape=(n_vars, n_vars)))
    # NOTE if we knew real graph, could force sign
    if effect_sgn is None:
        effect_sgn = rng.binomial(
            1, k_sign_p(rng, shape=(n_vars, 1)), size=g.shape) * 2.0 - 1.0
    else:
        assert np.array_equal(g != 0, effect_sgn != 0)

    k = k * effect_sgn.astype(np.float32)
    assert np.array_equal(k != 0, effect_sgn != 0)
    assert set(np.unique(effect_sgn)).issubset({-1.0, 0.0, 1.0})

    # sample number of cell types to be sampled (i.e. number of unique master regulator reproduction rates)
    n_cell_types = cell_types
    assert n_cell_types > 0, "Need at least 1 cell type to be simulated"

    # master regulator basal reproduction rate
    basal_rates = b(rng,
                    shape=(n_vars,
                           n_cell_types))  # assuming 1 cell type is simulated

    # hill coeff
    hills = hill * np.ones((n_vars, n_vars))

    # sample targets for experiments performed (wild-type, knockout)
    ko_targets = []

    simulate_observ_data = spec['n_observations_obs'] > 0
    if simulate_observ_data:
        # repeat 10 times, number_sc per run
        for _ in range(50):
            ko_targets += [None]

    simulate_interv_data = spec['n_observations_int'] > 0
    # >>> modify this
    if simulate_interv_data:
        '''
        assert n_ko_genes != 0, f"Need n_ko_genes != 0 to have interventional data for SERGIO"
        if n_ko_genes == -1:
            n_ko_genes = n_vars
        ko_targets += sorted(
            rng.choice(n_vars, size=min(n_vars, n_ko_genes),
                       replace=False).tolist())
        '''
        # all single targets
        #ko_targets.extend([[i] for i in range(n_vars)])
        # ugh SERGIO is too slow
        # >>> if I only run 1 cell type, 1000 is ok.
        num_single_samples = n_vars #min(100, n_vars)
        ko_targets.extend(np.random.choice(n_vars, num_single_samples,
            replace=False).reshape(num_single_samples, 1).tolist())
        # multiple targets
        max_num_targets = 3
        num_comb_samples = min(100, (n_vars * (n_vars - 1)) // 2)
        for num_targets in range(2, max_num_targets+1):
            cur_targets = []
            while len(cur_targets) < num_comb_samples:
                targets = rng.choice(n_vars, size=num_targets,
                                     replace=False)
                targets = tuple(sorted(targets))
                if targets in cur_targets:
                    continue
                cur_targets.append(targets)
            ko_targets.extend(cur_targets)
    # <<<

    # simulate for wild type and each ko target
    data = defaultdict(lambda: defaultdict(list))
    # collect interv_targets
    interv_targets = []
    sergio_kwargs = {
        "number_bins":  n_cell_types,  # c
        "noise_params":  noise_params,
        "noise_type":  noise_type,
        "decays":  decays,
        "sampling_state":  sampling_state,
        "dt":  dt,
        "safety_steps":  10,
    }
    graph_kwargs = {
        "g": g,
        "k": k,
        "b": basal_rates,  # def of cell type
        "hill": hills
    }
    # >>>
    results = _simulate_parallel(ko_targets, n_vars,
                  sergio_kwargs, graph_kwargs, num_cpus=40)
    import pickle
    with open("/data/scratch/rmwu/cache/causal_data/sergio/cry.pkl",
              "wb+") as f:
        pickle.dump(results, f)
    #exit(0)
    # <<<
    import pickle
    with open("/data/scratch/rmwu/cache/causal_data/sergio/cry.pkl", "rb") as f:
        results = pickle.load(f)
    for x, ko_mask, env, kout, data_type, interv_factor in results:

        # advanced per process
        # advance rng outside for faithfullness/freshness of data in for loop
        #rng = copy.deepcopy(sim.rng)
        data[data_type]["x"].append(x)
        data[data_type]["ko_mask"].append(ko_mask)
        data[data_type]["env"].append(env)
        if type(interv_factor) is float:
            interv_factor = str(interv_factor)
        else:
            interv_factor = [str(f) for f in interv_factor]
            interv_factor = ",".join(interv_factor)
        data[data_type]["interv_factor"].append(interv_factor)
        interv_targets.append(kout)

    ## NOTE OWN CODE HERE
    # concatenate interventional data by interweaving rows to have balanced intervention target counts
    if simulate_observ_data:
        x_obs = np.concatenate(data["obs"]["x"], axis=-1)
        x_obs_msk = np.concatenate(data["obs"]["ko_mask"], axis=-1)
        x_obs_env = np.zeros((x_obs.shape[0], x_obs.shape[-1]), dtype=int)
        #x_obs = np.stack(data["obs"]["x"]).reshape(-1, n_vars, order="F")
        #x_obs_msk = np.stack(data["obs"]["ko_mask"]).reshape(-1,
        #                                                      n_vars,
        #                                                      order="F")
        #x_obs_env = np.zeros(x_obs.shape[0], dtype=int)
    else:
        x_obs = np.zeros((0, n_vars))  # dummy
        x_obs_msk = np.zeros((0, n_vars))  # dummy
        x_obs_env = np.zeros(0, dtype=int)  # dummy

    if simulate_interv_data:
        x_int = np.concatenate(data["int"]["x"], axis=-1)
        x_int_msk = np.concatenate(data["int"]["ko_mask"], axis=-1)
        x_int_env = np.concatenate(data["int"]["env"], axis=-1)
        #x_int = np.stack(data["int"]["x"]).reshape(-1, n_vars, order="F")
        #x_int_msk = np.stack(data["int"]["ko_mask"]).reshape(-1,
        #                                                      n_vars,
        #                                                      order="F")
        #x_int_env = np.stack(data["int"]["env"]).reshape(-1, order="F")
    else:
        x_int = np.zeros((0, n_vars))  # dummy
        x_int_msk = np.zeros((0, n_vars))  # dummy
        x_int_env = np.zeros(0, dtype=int)
    # clip number of observations to be invariant to n_cell_types due to rounding
    # [n_observations, n_vars]
    # >>> omg omg omg
    #x_obs = x_obs[:spec['n_observations_obs']]
    #x_int = x_int[:spec['n_observations_int']]
    #x_obs_env = x_obs_env[:spec['n_observations_obs']]
    #x_int_env = x_int_env[:spec['n_observations_int']]
    # <<<

    assert x_obs.size != 0 or x_int.size != 0, f"Need to sample at least some observations; " \
                                               f"got shapes x_obs {x_obs.shape} x_int {x_int.shape}"

    # collect data
    # boolean [n_env, d]
    interv_targets = np.stack(interv_targets)
    return dict(
        g=g,
        n_vars=n_vars,
        x_obs=x_obs,
        x_int=x_int,
        obs_envs=x_obs_env,
        int_envs=x_int_env,
        obs_interv=data["obs"]["interv_factor"],
        int_interv=data["int"]["interv_factor"],
        interv_targets=interv_targets,
    )


def _simulate_parallel(ko_targets, n_vars,
              sergio_kwargs, graph_kwargs, num_cpus):
    configs = [(i, ko, n_vars, sergio_kwargs, graph_kwargs)
               for i, ko in enumerate(ko_targets)]
    with Pool(num_cpus) as pool:
        results = pool.map(_simulate, configs)
    return results


def _simulate(config):
    i, ko_target, n_vars, sergio_kwargs, graph_kwargs = config
    print(i, "starting")
    eye = np.eye(n_vars)
    rng = np.random.default_rng(i)
    number_sc = 200  # >>> hardcoded
    if ko_target is None:
        # observational/wild type
        data_type = "obs"
        kout = np.zeros(n_vars).astype(bool)
        interv_factor = 1.0
        #number_sc = math.ceil(spec['n_observations_obs'] / n_cell_types)

    else:
        # interventional/knockout
        data_type = "int"
        kout = 0
        for gene in ko_target:
            kout = kout + eye[gene]
        kout = kout.astype(bool)
        # based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4830697
        interv_factor = rng.uniform(low=0.3, high=0.95, size=len(ko_target))
        interv_factor = interv_factor.tolist()
        # >>> NOTE: n_observations_int is SHARED
        #number_sc = math.ceil(spec['n_observations_int'] /
        #                      (n_cell_types * n_ko_genes))

    # setup simulator
    sim = SergioSimulator(
        rng=rng,
        number_genes=n_vars,  # d
        number_sc=number_sc,  # M
        # >>> kout vs. kdown
        #kout=kout,
        kdown=kout,
        interv_factor=interv_factor,
        # <<<
        **sergio_kwargs
    )

    sim.custom_graph(**graph_kwargs)

    # run steady-state simulations
    assert number_sc >= 1, f"Need to have number_sc >= 1: number_sc {number_sc} data_type {data_type}"
    sim.simulate()

    # Get the clean simulated expression matrix after steady_state simulations
    # shape: [number_bins (#cell types), number_genes, number_sc (#cells per type)]
    expr = sim.getExpressions()
    # >>>
    expr = sim.convert_to_UMIcounts(expr)
    # <<<

    # Aggregate by concatenating gene expressions of all cell types into 2d array
    # [number_genes (#genes), number_bins * number_sc (#cells  = #cell_types * #cells_per_type)]
    # >>>
    #expr_agg = np.concatenate(expr, axis=1)
    expr_agg = expr
    # <<<

    # Now each row represents a gene and each column represents a simulated single-cell
    # Gene IDs match their row in this expression matrix
    # [number_bins * number_sc, number_genes]
    # >>>
    x = expr_agg
    #x = expr_agg.T
    #x = rng.permutation(x, axis=0)
    # <<< wtf why would they permute it

    # generate intervention mask
    # [number_bins * number_sc, number_genes] with True/False depending on whether gene was knocked out
    # >>> [number_bins , number_sc, number_genes] with True/False depending on whether gene was knocked out
    ko_mask = np.tile(kout, (x.shape[0], x.shape[2], 1)).astype(np.float32)
    ko_mask = np.swapaxes(ko_mask, 1, 2)

    # environment indicator
    #env = np.array([i] * x.shape[0], dtype=int)
    env = np.ones((x.shape[0], x.shape[2]), dtype=int) * i
    #print(x.shape)
    #print(ko_mask.shape)
    #print(env.shape)

    print(i, "done")
    return x, ko_mask, env, kout, data_type, interv_factor


#####
# >>>
sergio_clean = functools.partial(
    grn_sergio,
    add_outlier_effect=False,
    add_lib_size_effect=False,
    add_dropout_effect=False,
    return_count_data=True,
)
#####
"""
THESE BELOW AREN'T USED
"""

sergio_clean_count = functools.partial(
    grn_sergio,
    add_outlier_effect=False,
    add_lib_size_effect=False,
    add_dropout_effect=False,
    return_count_data=True,
    n_ko_genes=0,
)

sergio_noisy = functools.partial(
    grn_sergio,
    add_outlier_effect=True,
    add_lib_size_effect=True,
    add_dropout_effect=True,
    return_count_data=False,
    n_ko_genes=0,
)

sergio_noisy_count = functools.partial(
    grn_sergio,
    add_outlier_effect=True,
    add_lib_size_effect=True,
    add_dropout_effect=True,
    return_count_data=True,
    n_ko_genes=0,
)

# interventional versions
kosergio_clean = functools.partial(
    grn_sergio,
    add_outlier_effect=False,
    add_lib_size_effect=False,
    add_dropout_effect=False,
    return_count_data=False,
)

kosergio_clean_count = functools.partial(
    grn_sergio,
    add_outlier_effect=False,
    add_lib_size_effect=False,
    add_dropout_effect=False,
    return_count_data=True,
)

kosergio_noisy = functools.partial(
    grn_sergio,
    add_outlier_effect=True,
    add_lib_size_effect=True,
    add_dropout_effect=True,
    return_count_data=False,
)

kosergio_noisy_count = functools.partial(
    grn_sergio,
    add_outlier_effect=True,
    add_lib_size_effect=True,
    add_dropout_effect=True,
    return_count_data=True,
)
