from copy import deepcopy
import os
import pickle
from collections import namedtuple
from warnings import warn
import numpy as onp
# >>>
import numpy as jnp
import numpy as np
#import jax.numpy as jnp
# <<<
from jax import random, vmap, jit
from jax.scipy.special import logsumexp

from bacadi.graph.graph import ErdosReniDAGDistribution, ScaleFreeDAGDistribution, UniformDAGDistributionRejection
from bacadi.kernel.joint import JointAdditiveFrobeniusSEKernel, JointMultiplicativeFrobeniusSEKernel
from bacadi.kernel.interv import JointAdditiveFrobeniusSEKernel as JointAdditiveInterv
from bacadi.kernel.interv import MarginalAdditiveFrobeniusSEKernel as MarginalAdditiveInterv
from bacadi.kernel.marginal import FrobeniusSquaredExponentialKernel
from bacadi.utils.graph import graph_to_mat, adjmat_to_str, make_all_dags, mat_to_graph

from bacadi.models.linearGaussian import LinearGaussian, LinearGaussianJAX
from bacadi.models.linearGaussianEquivalent import BGe, BGeJAX, NewBGe
from bacadi.models.nonlinearGaussian import DenseNonlinearGaussianJAX
from bacadi.models.sobolev import SobolevGaussianJAX

from bacadi.utils.func import bit2id

from sergio.sergio_sampler import sergio_clean

STORE_ROOT = ['store']

if not os.path.exists(os.path.join(*STORE_ROOT)):
    os.makedirs(os.path.join(*STORE_ROOT))


Target = namedtuple('Target', (
    'passed_key',               # jax.random key passed _into_ the function generating this object
    'graph_model',
    'n_vars',
    'n_observations',
    'n_ho_observations',
    'g',                        # [n_vars, n_vars]
    'theta',                    # PyTree
    'x',                        # [n_observation, n_vars]    data
    'x_ho',                     # [n_ho_observation, n_vars] held-out data
    'x_interv',                 # list of (interv dict, interventional data)
    'x_interv_data',            # (n_obs stacked, n_vars) interventional data (same data as above)
    'interv_targets',           # (n_env, n_vars)
    'interv_factors',           # (n_env,)
    'envs',                     # (n_obs) giving env that sample i came from
    'x_ho_interv',              # a tuple (interv dict, held-out interventional data)
    'x_ho_interv_data',         # (n_obs stacked, n_vars) heldout interventional data (same data as above)
    'envs_ho',                  # (n_obs) giving env that sample i came from
    'interv_targets_ho',        # (n_env, n_vars)
    'gt_posterior_obs',         # ground-truth posterior with observational data
                                # (log distribution tuple as e.g. given by `particle_marginal_empirical`), or None
    'gt_posterior_interv'       # ground-truth posterior with interventional data
                                # (log distribution tuple as e.g. given by `particle_marginal_empirical`), or None
))


def save_pickle(obj, relpath):
    """Saves `obj` to `path` using pickle"""
    # >>>
    folder, seed = relpath
    #folder = os.path.join("/data/rsg/chemistry/rmwu/src/pkg/dcdi/data/sergio", folder)
    folder = os.path.join("/data/scratch/rmwu/cache/causal_data/sergio", folder)
    # <<<
    if not os.path.exists(folder):
        os.makedirs(folder)
        print("Created", folder)
    # save graph
    graph = obj["g"]
    fp_graph = os.path.join(folder, f"DAG{seed}.npy")
    np.save(fp_graph, graph)
    # save datasets, observational and interventional
    # data_interv already contains all of data
    data = np.transpose(obj["x"], axes=[0,2,1])  # (cell type, cells, genes)
    data_interv = np.transpose(obj["x_interv_data"], axes=[0,2,1])
    #print(data.shape, data_interv.shape)
    assert data_interv.shape[0] == data.shape[0], "# cell types not same"
    num_celltypes, num_cells_obs, num_genes = data.shape
    _, num_cells_int, _ = data_interv.shape
    # reshape to (num_cells, genes)
    # >>> DO NOT DO THIS
    data = data.reshape(-1, num_genes)
    data_interv = data_interv.reshape(-1, num_genes)
    fp_data = os.path.join(folder, f"data{seed}.npy")
    fp_data_interv = os.path.join(folder, f"data_interv{seed}.npy")
    # >>>
    #np.save(fp_data, data)
    #np.save(fp_data_interv, data_interv)
    # <<<
    # regimes and interventions
    targets = obj["interv_targets"]
    interventions = []
    for row in targets:
        row = np.nonzero(row)[0].reshape(-1).tolist()
        row = [str(i) for i in row]
        interventions.append(",".join(row))
    regimes = obj["envs"].reshape(-1).tolist()  # expand to length of data
    ints_long = [interventions[r] for r in regimes]
    interv_factors = [obj["interv_factors"][r] for r in regimes]
    # cell types
    celltypes_obs, celltypes_int = [], []
    for i in range(num_celltypes):
        celltypes_obs.extend([i] * num_cells_obs)
        celltypes_int.extend([i] * num_cells_int)

    assert len(regimes) == len(ints_long) == len(interv_factors)
    assert len(celltypes_int) == len(regimes)
    fp_attr = os.path.join(folder, f"attributes_interv{seed}.tsv")
    data = [{"regime": regime,
             "intervention": interv,
             "interv_factor": interv_factor,
             "celltype": celltype} for regime, interv, interv_factor, celltype
            in zip(regimes, ints_long, interv_factors, celltypes_int)]
    import csv
    with open(fp_attr, "w+") as f:
        writer = csv.DictWriter(f, fieldnames=data[0], delimiter="\t")
        writer.writeheader()
        for item in data:
            writer.writerow(item)

    #fp_intervention = os.path.join(folder, f"intervention{seed}.csv")
    #fp_regime = os.path.join(folder, f"regime{seed}.csv")
    #fp_interv = os.path.join(folder, f"interv_factor{seed}.csv")
    #with open(fp_intervention, "w+") as f:
    #    for row in ints_long:
    #        f.write(row)
    #        f.write(os.linesep)
    #with open(fp_regime, "w+") as f:
    #    for row in regimes:
    #        f.write(str(row))
    #        f.write(os.linesep)
    #with open(fp_interv, "w+") as f:
    #    for row in interv_factors:
    #        f.write(str(row))
    #        f.write(os.linesep)
    # save cell types
    #fp_obs = os.path.join(folder, f"celltype{seed}.csv")
    #fp_int = os.path.join(folder, f"celltype_interv{seed}.csv")
    #with open(fp_obs, "w+") as f:
    #    for row in celltypes_obs:
    #        f.write(str(row))
    #        f.write(os.linesep)
    #with open(fp_int, "w+") as f:
    #    for row in celltypes_int:
    #        f.write(str(row))
    #        f.write(os.linesep)
    return
    # <<<
    save_path = os.path.abspath(os.path.join(
        *STORE_ROOT, relpath + '.pkl'
    ))
    with open(save_path, 'wb') as fp:
        pickle.dump(obj, fp)

def load_pickle(relpath):
    """Loads object from `path` using pickle"""
    load_path = os.path.abspath(os.path.join(
        *STORE_ROOT, relpath + '.pk'
    ))
    with open(load_path, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def options_to_str(**options):
    #return '-'.join(['{}={}'.format(k, v) for k, v in options.items()])
    seed = options["seed"]
    del options["seed"]
    folder = '-'.join(['{}={}'.format(k, v) for k, v in options.items()])
    return folder, int(seed)

def hparam_dict_to_str(d):
    """
    Converts hyperparameter dictionary into human-readable string
    """
    strg = '_'.join([k + '=' + str(v) for k, v, in d.items()
                     if type(v) in [bool, int, float, str, dict]])
    return strg

def interv_list_to_tuple(x_interv, n_vars):
    """
    Transform list of pairs (intervention dict, samples) to a
    tuple of jnp.arrays (data, intervention_targets)

    Args:
        x_interv: list of (intervention dict, x_) of length k,
                  where the dict is of node indices to values, and the x_ correspond
                  to samples from the interventional distribution
        n_vars (int): number of variables

    Returns:
        tuple (data, interv_targets)
        where data is a jnp.array of shape [sum_{n_obs}, n_vars]
        and interv_targets is a boolean mask of shape [sum_{n_obs}, n_vars]
    """
    data = jnp.concatenate([interv_set[1] for interv_set in x_interv])
    interv_targets = jnp.concatenate([
        jnp.array([[i in interv_set[0].keys() for i in range(n_vars)]
                   for _ in interv_set[1]]) for interv_set in x_interv
    ])
    env_size = jnp.array([interv_set[1].shape[0] for interv_set in x_interv])
    envs = jnp.repeat(jnp.arange(len(x_interv)), env_size)
    return (data, interv_targets, envs)


def create_data_interv_tuples(target):
    """
        Create tuples of the form (data, interv_targets, environments) for the
        train and held-out datasamples of the target.

    Args:
        target (Target)

    Returns:
        tuple (x, x_ho, X_interv, X_ho_interv)
    """
    # train, held-out and interventional samples
    n_vars = target.n_vars
    x = jnp.array(target.x)
    x_ho = jnp.array(target.x_ho)
    x_interv = target.x_interv
    x_ho_interv = target.x_ho_interv

    X_interv = interv_list_to_tuple(x_interv, n_vars)

    X_ho_interv = interv_list_to_tuple(x_ho_interv, n_vars)

    return (x, x_ho, X_interv, X_ho_interv)


def make_graph_model(*, n_vars, graph_prior_str, edges_per_node=2, n_edges=None):
    """
    Instantiates graph model

    Args:
        n_vars: number of variables
        graph_prior_str: specifier (`er`, `sf`)

    Returns:
        `GraphDistribution`
    """
    if graph_prior_str == 'er':
        graph_model = ErdosReniDAGDistribution(
            n_vars=n_vars,
            n_edges=n_edges if n_edges is not None else edges_per_node * n_vars)

    elif graph_prior_str == 'sf':
        graph_model = ScaleFreeDAGDistribution(
            n_vars=n_vars,
            n_edges_per_node=edges_per_node)

    else:
        assert n_vars <= 5
        graph_model = UniformDAGDistributionRejection(
            n_vars=n_vars)

    return graph_model


def make_sergio(n_vars,
                seed=1,
                sergio_hill=2,
                sergio_decay=0.8,
                sergio_noise_params=1.0,
                sergio_cell_types=10,
                sergio_k_lower_lim=1,
                sergio_k_upper_lim=5,
                graph_prior_edges_per_node=2,
                n_observations=100,
                n_ho_observations=100,
                n_intervention_sets=5,
                n_interv_obs=10):
    key = random.PRNGKey(seed)
    graph_model = make_graph_model(
        n_vars=n_vars,
        graph_prior_str='sf',
        edges_per_node=graph_prior_edges_per_node)

    key, subk = random.split(key)
    g_gt = graph_model.sample_G(subk)
    g_gt_mat = jnp.array(graph_to_mat(g_gt))

    n_ko_genes = n_intervention_sets  # >>> not used here
    rng = onp.random.default_rng(seed)
    passed_rng = deepcopy(rng)

    def k_param(rng, shape):
        return rng.uniform(
            low=sergio_k_lower_lim,  # default 1
            high=sergio_k_upper_lim,  # default 5
            size=shape)

    # MLE from e.coli
    def k_sign_p(rng, shape):
        return rng.beta(a=0.2588, b=0.2499, size=shape)

    def b(rng, shape):
        return rng.uniform(low=1, high=3, size=shape)

    n_obs = n_observations
    n_obs_ho = n_ho_observations
    spec = {}
    # number of intv
    spec['n_observations_int'] = 2 * n_intervention_sets * n_interv_obs
    # double to have heldout dataset
    spec['n_observations_obs'] = 2 * n_observations + n_ho_observations

    data = sergio_clean(
        spec=spec,
        rng=rng,  # function?
        g=g_gt_mat,
        effect_sgn=None,  # TODO
        toporder=None,
        n_vars=n_vars,
        b=b,  # function
        k_param=k_param,  # function
        k_sign_p=k_sign_p,  # function
        hill=sergio_hill,  # default 2
        decays=sergio_decay,  # default 0.8
        noise_params=sergio_noise_params,  # default 1.0
        cell_types=sergio_cell_types,  # default 10
        n_ko_genes=n_ko_genes)

    # extract data
    # obs
    # >>> ugh why do they do these weird things omg
    #x = jnp.array(data['x_obs'][:n_obs])
    #x_ho = jnp.array(data['x_obs'][n_obs:(n_obs + n_obs_ho)])
    # interv
    #x_interv_obs = jnp.array(data['x_obs'][-n_obs:])
    #x_interv, x_ho_interv = jnp.split(data['x_int'], 2)
    #envs, envs_ho = jnp.split(data['int_envs'], 2)
    x = jnp.array(data['x_obs'])
    x_interv_obs = x  # jnp.array(data['x_obs'][-n_obs:])
    x_interv = data["x_int"]
    # pickle will transpose. currently (cell type, gene, obs)
    x_interv_data = jnp.concatenate([x_interv_obs, x_interv], axis=-1)
    envs = jnp.concatenate([data["obs_envs"],
                            data["int_envs"]], axis=-1).astype(int)
    interv_factors = data["obs_interv"] + data["int_interv"]

    # ho interv
    #x_ho_interv = jnp.array(x_ho_interv)
    #envs_ho = jnp.array(envs_ho, dtype=int)
    interv_targets = data['interv_targets']

    # standardize
    # >>>
    #mean = x_interv_data.mean(0)
    #std = x_interv_data.std(0)
    #print(x.shape, x_interv_data.shape)
    # >>> I CAN STANDARDIZE IN MY OWN WORKFLOW
    # do not destroy distributions here
    #mean = x_interv_data.mean(0, keepdims=True).mean(-1, keepdims=True)
    #std = x_interv_data.std(0, keepdims=True).mean(-1, keepdims=True)
    # <<<

    #x = (x - mean) / jnp.where(std == 0.0, 1.0, std)
    #x_interv_data = (x_interv_data - mean) / jnp.where(std == 0.0, 1.0, std)
    #x_ho = (x_ho - mean) / jnp.where(std == 0.0, 1.0, std)
    #x_ho_interv = (x_ho_interv - mean) / jnp.where(std == 0.0, 1.0, std)

    target = Target(
        passed_key=passed_rng,
        graph_model=graph_model,
        n_vars=n_vars,
        n_observations=n_observations,
        #n_ho_observations=n_ho_observations,
        n_ho_observations=None,
        g=g_gt_mat,
        theta=None,
        x=x,
        #x_ho=x_ho,
        x_ho=None,
        x_interv=[{} for _ in interv_targets],  # dummy
        x_interv_data=x_interv_data,
        interv_targets=interv_targets,
        envs=envs,
        interv_factors=interv_factors,
        x_ho_interv=[{} for _ in interv_targets],  # dummy
        #x_ho_interv_data=x_ho_interv,
        x_ho_interv_data=None,
        interv_targets_ho=interv_targets,
        #envs_ho=envs_ho,
        envs_ho=None,
        gt_posterior_obs=None,
        gt_posterior_interv=None)
    return target

