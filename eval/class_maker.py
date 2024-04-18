import os
from jax import random
import jax.numpy as jnp

#from bacadi.inference.joint_dibs_svgd import JointDiBS
from bacadi.inference.bacadi_joint import BaCaDIJoint
#from bacadi.inference.marginal_dibs_svgd import MarginalDiBS
#from bacadi.inference.bacadi_marginal import BaCaDIMarginal
from bacadi.eval.target import load_pickle, make_graph_model, make_sergio, options_to_str, save_pickle

from bacadi.models.linearGaussian import LinearGaussian
from bacadi.models.nonlinearGaussian import DenseNonlinearGaussianJAX
from bacadi.models.sobolev import SobolevGaussianJAX
from bacadi.utils.graph import adjmat_to_str


def linear_chain(n_vars):
    return [[1. if i == j - 1 else 0. for j in range(n_vars)]
            for i in range(n_vars)]


def tree_graph(n_vars):
    return [[
        1. if j == 2 * i + 1 or j == 2 * i + 2 else 0. for j in range(n_vars)
    ] for i in range(n_vars)]


def no_intervention_targets(n_vars):
    return jnp.zeros(n_vars).astype(bool)


DIBS_PAPER_EXAMPLE = [
    [0., 2., -2., 0.],
    [0., 0., 0., 3.],
    [0., 0., 0., 1.],
    [0., 0., 0., 0.],
]


def make_target(args, load=True):
    """
    Create synthetic Bayesian network
    TODO proper docstring
    """
    n_vars = args.n_vars

    key = random.PRNGKey(args.seed)

    filename = args.simulator
    if args.simulator == 'synthetic':
        filename += f'_{args.joint_inference_model}'
    # >>>
    #filename += '_' + \
    filename = options_to_str(
                p=n_vars,
                e=args.graph_prior_edges_per_node*n_vars,
                #n_sc="10000C+200P-count",
                n_sc="10000C+200P-count-1_celltype",
                #n_sc="testing",
                #graph_type=args.target_graph,
                #graph=args.graph_prior,
                #n_obs=args.n_observations,
                #n_interv_obs=args.n_interv_obs,
                #n_ho=args.n_ho_observations,
                #interventions=args.intervention_type,
                #intervention_val=args.intervention_val,
                #intervention_noise=args.intervention_noise,
                #sergio_hill=args.sergio_hill,
                seed=args.seed,
            )
    # >>>
    folder, seed = filename
    folder = os.path.join("/data/rsg/chemistry/rmwu/src/pkg/dcdi/data/sergio", folder)
    fp_regime = os.path.join(folder, f"regime{seed}.csv")
    if os.path.exists(fp_regime):
        print("Already done", fp_regime)
        return None, None
    # <<<
    if load:
        try:
            target = load_pickle(filename)

            print(
                'Loaded {}-vars, {}-edge graph with {} generative model:\t{}'.
                format(
                    target.n_vars,
                    jnp.sum(target.g).item(), args.joint_inference_model
                    if args.simulator == 'synthetic' else 'sergio',
                    adjmat_to_str(target.g)))
            return target, filename

        except FileNotFoundError:
            print('Loading failed: ' + filename + ' does not exist.')
            print('Generating from scratch...')

    ############################ SERGIO GRN
    if args.simulator == 'sergio':
        target = make_sergio(
            n_vars,
            seed=args.seed,
            sergio_hill=args.sergio_hill,
            sergio_decay=args.sergio_decay,
            sergio_noise_params=args.sergio_noise_params,
            sergio_cell_types=args.sergio_cell_types,
            sergio_k_lower_lim=args.sergio_k_lower_lim,
            sergio_k_upper_lim=args.sergio_k_upper_lim,
            graph_prior_edges_per_node=args.graph_prior_edges_per_node,
            n_observations=args.n_observations,
            n_ho_observations=args.n_ho_observations,
            n_intervention_sets=args.n_intervention_sets,
            n_interv_obs=args.n_interv_obs)
        # >>>
        #filename = "/data/rsg/chemistry/rmwu/src/pkg/bacadi/sergio.pkl"
        #filename = "/data/rsg/chemistry/rmwu/src/pkg/bacadi/sergio_10000.pkl"
        target = target._asdict()
        target["graph_model"] = str(target["graph_model"])
        # <<<
        save_pickle(target, filename)
        print(f"Wrote target file to {filename}.")
        return target, filename

    # >>> I deleted the rest


def make_nointerv_bacadi(args, model_param, callback, key):
    # no interventions reduces to DiBS
    if args.joint:
        dibs = JointDiBS(
            random_state=key,
            model_param=model_param,
            kernel=args.joint_bacadi_kernel,
            graph_prior=args.joint_bacadi_graph_prior,
            edges_per_node=args.joint_bacadi_graph_prior_edges_per_node,
            model_prior=args.joint_bacadi_inference_model,
            alpha_linear=args.joint_bacadi_alpha_linear,
            beta_linear=args.joint_bacadi_beta_linear,
            tau=args.joint_bacadi_tau_linear,
            h_latent=args.joint_bacadi_h_latent,
            h_theta=args.joint_bacadi_h_theta,
            optimizer=dict(name=args.bacadi_optimizer,
                           stepsize=args.bacadi_optimizer_stepsize),
            n_grad_mc_samples=args.joint_bacadi_n_grad_mc_samples,
            n_acyclicity_mc_samples=args.joint_bacadi_n_acyclicity_mc_samples,
            grad_estimator_z=args.joint_bacadi_grad_estimator_z,
            score_function_baseline=args.joint_bacadi_score_function_baseline,
            n_steps=args.joint_bacadi_n_steps,
            n_particles=args.n_particles,
            callback_every=args.callback_every,
            callback=callback if args.callback else None,
            verbose=args.verbose,
        )
    else:
        dibs = MarginalDiBS(
            random_state=key,
            model_param=model_param,
            kernel=args.bacadi_kernel,
            graph_prior=args.bacadi_graph_prior,
            edges_per_node=args.bacadi_graph_prior_edges_per_node,
            model_prior=args.bacadi_inference_model,
            alpha_linear=args.bacadi_alpha_linear,
            beta_linear=args.bacadi_beta_linear,
            tau=args.bacadi_tau_linear,
            h_latent=args.bacadi_h_latent,
            optimizer=dict(name=args.bacadi_optimizer,
                           stepsize=args.bacadi_optimizer_stepsize),
            n_grad_mc_samples=args.bacadi_n_grad_mc_samples,
            n_acyclicity_mc_samples=args.bacadi_n_acyclicity_mc_samples,
            grad_estimator_z=args.bacadi_grad_estimator_z,
            score_function_baseline=args.bacadi_score_function_baseline,
            n_steps=args.bacadi_n_steps,
            n_particles=args.n_particles,
            callback_every=args.callback_every,
            callback=callback if args.callback else None,
            verbose=args.verbose,
        )

    return dibs


def make_interv_bacadi(args, model_param, callback, key):
    """
    Finally, this is the model factory
    """
    # this is the default option
    if args.joint:
        bacadi = BaCaDIJoint(
            random_state=key,
            model_param=model_param,
            kernel=args.joint_bacadi_interv_kernel,
            graph_prior=args.joint_bacadi_graph_prior,
            edges_per_node=args.joint_bacadi_graph_prior_edges_per_node,
            model_prior=args.joint_bacadi_inference_model,
            alpha_linear=args.joint_bacadi_alpha_linear,
            beta_linear=args.joint_bacadi_beta_linear,
            tau=args.joint_bacadi_tau_linear,
            h_latent=args.joint_bacadi_h_latent,
            h_theta=args.joint_bacadi_h_theta,
            h_interv=args.joint_bacadi_h_interv,
            interv_per_env=args.bacadi_interv_per_env,
            lambda_regul=args.bacadi_lambda_regul,
            optimizer=dict(name=args.bacadi_optimizer,
                           stepsize=args.bacadi_optimizer_stepsize),
            n_grad_mc_samples=args.joint_bacadi_n_grad_mc_samples,
            n_acyclicity_mc_samples=args.joint_bacadi_n_acyclicity_mc_samples,
            grad_estimator_z=args.joint_bacadi_grad_estimator_z,
            score_function_baseline=args.joint_bacadi_score_function_baseline,
            n_steps=args.joint_bacadi_n_steps,
            n_particles=args.n_particles,
            callback_every=args.callback_every,
            callback=callback if args.callback else None,
            verbose=args.verbose,
        )
    # this is not the default option
    else:
        bacadi = BaCaDIMarginal(
            random_state=key,
            model_param=model_param,
            kernel=args.bacadi_interv_kernel,
            graph_prior=args.bacadi_graph_prior,
            edges_per_node=args.bacadi_graph_prior_edges_per_node,
            model_prior=args.bacadi_inference_model,
            alpha_linear=args.bacadi_alpha_linear,
            beta_linear=args.bacadi_beta_linear,
            tau=args.bacadi_tau_linear,
            h_latent=args.bacadi_h_latent,
            h_interv=args.bacadi_h_interv,
            interv_per_env=args.bacadi_interv_per_env,
            lambda_regul=args.bacadi_lambda_regul,
            optimizer=dict(name=args.bacadi_optimizer,
                           stepsize=args.bacadi_optimizer_stepsize),
            n_grad_mc_samples=args.bacadi_n_grad_mc_samples,
            n_acyclicity_mc_samples=args.bacadi_n_acyclicity_mc_samples,
            grad_estimator_z=args.bacadi_grad_estimator_z,
            score_function_baseline=args.bacadi_score_function_baseline,
            n_steps=args.bacadi_n_steps,
            n_particles=args.n_particles,
            callback_every=args.callback_every,
            callback=callback if args.callback else None,
            verbose=args.verbose,
        )

    return bacadi


def make_bacadi(target, args, callback, key):
    # >>> this the default I think
    if args.joint:
        # this is for linear data
        if args.joint_bacadi_inference_model == "lingauss":
            model_param = dict(
                obs_noise=args.bacadi_lingauss_obs_noise,
                mean_edge=args.bacadi_lingauss_mean_edge,
                sig_edge=args.bacadi_lingauss_sig_edge,
                init_sig_edge=args.bacadi_lingauss_init_sig_edge,
                interv_mean=args.intervention_val,
                interv_noise=args.intervention_noise,
                interv_prior_mean=args.interv_prior_mean,
                interv_prior_std=args.interv_prior_std)
        # this is for non-linear data
        elif args.joint_bacadi_inference_model == 'fcgauss':
            model_param = dict(
                obs_noise=args.bacadi_fcgauss_obs_noise,
                sig_param=args.bacadi_fcgauss_sig_param,
                init_sig_param=args.bacadi_fcgauss_init_sig_param,
                hidden_layers=(args.bacadi_fcgauss_n_neurons, ) *
                args.bacadi_fcgauss_hidden_layers,
                init=args.bacadi_fcgauss_init_param,
                activation=args.bacadi_fcgauss_activation,
                interv_mean=args.intervention_val,
                interv_noise=args.intervention_noise,
                interv_prior_mean=args.interv_prior_mean,
                interv_prior_std=args.interv_prior_std)
        elif args.joint_bacadi_inference_model == 'sobolevgauss':
            model_param = dict(
                obs_noise=args.bacadi_sobolevgauss_obs_noise,
                sig_param=args.bacadi_sobolevgauss_sig_param,
                mean_param=args.bacadi_sobolevgauss_mean_param,
                n_vars=args.n_vars,
                n_exp=args.bacadi_sobolevgauss_n_exp,
                init_sig_param=args.bacadi_sobolevgauss_init_sig_param,
                init=args.bacadi_sobolevgauss_init_param,
                interv_mean=args.intervention_val,
                interv_noise=args.intervention_noise,
                interv_prior_mean=args.interv_prior_mean,
                interv_prior_std=args.interv_prior_std)
    # >>> this is not the default
    else:
        if args.simulator == 'synthetic':
            # for the BGe prior mean vector over interventions, take the mean of the intervention
            # values where node i has been intervened on
            interv_means = jnp.zeros((len(target.x_interv) - 1, args.n_vars),
                                     dtype=float)
            for i, interv_set in enumerate(
                    target.x_interv[1:]):  # assume first is observational
                for k, v in interv_set[0].items():
                    if k in range(args.n_vars):  # intv on k in i-th env
                        if type(v) == dict:  # mean, noise dict
                            interv_means = interv_means.at[i, k].set(v['mean'])
                        else:  # direct value
                            interv_means = interv_means.at[i, k].set(v)
            interv_means = jnp.where(
                (interv_means != 0).sum(0) != 0,
                interv_means.sum(0) / (interv_means != 0).sum(0), -1e8)
        else:  # for sergio, take the argument
            interv_means = float(args.intervention_val)
        model_param = dict(alpha_mu=args.bacadi_bge_alpha_mu,
                           alpha_lambd=args.n_vars +
                           args.bacadi_bge_alpha_lambd_add,
                           interv_mean=interv_means,
                           interv_noise=args.intervention_noise)

    if args.infer_interv:
        return make_interv_bacadi(args, model_param, callback, key)
    else:
        return make_nointerv_bacadi(args, model_param, callback, key)

