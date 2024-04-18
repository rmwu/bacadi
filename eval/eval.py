import copy
import functools
import pandas as pd
import torch
from timeit import default_timer as timer
import time
from datetime import timedelta

from class_maker import make_bacadi, make_target
#from baselines.bootstrap import run_bootstrap
from bacadi.utils.func import expected_graph, expected_interv
from result import get_metrics
from jax import random


def callback(target, args, **kwargs):
    if args.method == "bacadi":
        zs = kwargs["zs"]
        gs = kwargs["model"].particle_to_g_lim(zs)
        probs = kwargs["model"].edge_probs(zs, kwargs["t"])

        bacadi_empirical = kwargs["model"].particle_empirical()
        bacadi_mixture = kwargs["model"].particle_mixture()

        metrics = {}
        metrics.update(get_metrics("bacadi_", bacadi_empirical, target, args))
        metrics.update(get_metrics("bacadi+_", bacadi_mixture, target, args))
    else:
        raise ValueError(f"{args.method} not yet implemented in callback")
    return


def eval_single_target(*,  # out of all the coding conventions, they chose to
                       # enforce that their parameters are named ???
                       target,
                       #method,
                       #graph_prior_str,
                       #target_seed,
                       #model_seed,
                       #group_id,
                       args,):
                       #load=True):
    args = copy.deepcopy(args)
    # >>> this logic is ridiculous and circular wtf
    # they literally extract method from args and rename it the same thing
    # either pass the variable or pass the args! don't do both! don't modify
    # args!
    #args.method = method
    #args.seed = target_seed
    #args.model_seed = model_seed
    #args.graph_prior = graph_prior_str
    #args.bacadi_graph_prior = graph_prior_str
    #args.joint_bacadi_graph_prior = graph_prior_str
    # <<<
    method = args.method

    # Instead of crafting synthetic data from scratch, load from disk!!
    # The original design is so ridiculous in terms of reproducibility.
    # Who expects to create NEW data on the fly???
    # Loading also should NOT be a flag to a data generation function!

    # >>> target should be a dict with all the keys -> np arrays / lists
    # we pass it as input, rather than creating it anew each time
    #target, filename = make_target(args, load)
    # <<<

    data_type = 'interventional' if args.interv_data or args.infer_interv else 'observational'

    key = random.PRNGKey(args.model_seed)

    metrics = {}
    #start = timer()  # this is absolutely ridiculous why do you have two timers
    #t_before = time.time()  # named different things?

    #####################
    if method == "bacadi":
        bacadi = make_bacadi(target=target,
                             args=args,
                             callback=functools.partial(callback,
                                                        target=target,
                                                        args=args),
                             key=key)

        # >>> we are here!!
        if args.infer_interv:
            # data and datapoint -> environment mappin
            bacadi.fit(target.x_interv_data, target.envs)
        else:
            if args.interv_data:
                interv_targets = target.interv_targets
                # drop obs. setting
                if (interv_targets[0] == 0).all():
                    interv_targets = interv_targets[1:]
                bacadi.fit(target.x_interv_data, interv_targets, target.envs)
            else:
                bacadi.fit(target.x)
        # ... eval
        dist_empirical = bacadi.particle_empirical()
        dist_mixture = bacadi.particle_mixture()
        metrics.update(
            get_metrics("empirical_",
                        dist_empirical,
                        target,
                        args,
                        bacadi=bacadi,
                        final=True))
        metrics.update(
            get_metrics("mixture_",
                        dist_mixture,
                        target,
                        args,
                        bacadi=bacadi,
                        final=True))
    #####################
    elif method == "DCDI-G" or method == "DCDI-DSF":
        # adjust some default hparams, taken from dcdi/main
        if args.dcdi_lr_reinit is None:
            args.dcdi_lr_reinit = args.dcdi_lr

        # Use GPU
        if args.dcdi_gpu:
            if args.dcdi_float:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        else:
            if args.dcdi_float:
                torch.set_default_tensor_type('torch.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.DoubleTensor')

        dist_empirical, dist_mixture, metrics_bootstrap = run_bootstrap(
            key, args, target, learner_str='dcdi')
        metrics.update(metrics_bootstrap)
        metrics.update(
            get_metrics("empirical_",
                        dist_empirical,
                        target,
                        args,
                        final=True))
        metrics.update(
            get_metrics("mixture_", dist_mixture, target, args, final=True))
    #####################
    elif method == "JCI-PC":
        dist_empirical, dist_mixture, metrics_bootstrap = run_bootstrap(
            key, args, target, learner_str='jci-pc')
        metrics.update(metrics_bootstrap)
        metrics.update(
            get_metrics("empirical_",
                        dist_empirical,
                        target,
                        args,
                        use_cpdag=True,
                        final=True))
        metrics.update(
            get_metrics("mixture_",
                        dist_mixture,
                        target,
                        args,
                        use_cpdag=True,
                        final=True))
    #####################
    elif method == "IGSP":
        dist_empirical, dist_mixture, metrics_bootstrap = run_bootstrap(
            key, args, target, learner_str='igsp')
        metrics.update(metrics_bootstrap)
        metrics.update(
            get_metrics("empirical_",
                        dist_empirical,
                        target,
                        args,
                        use_cpdag=True,
                        final=True))
        metrics.update(
            get_metrics("mixture_",
                        dist_mixture,
                        target,
                        args,
                        use_cpdag=True,
                        final=True))
    #####################
    else:
        raise ValueError(f"{method} method not implemented yet")
    #####
    #t_after = time.time()
    #end = timer()
    #delta = timedelta(seconds=end - start)
    # wow ok I got it, so the two timers = they don't know how to strftime

    results = dict(
        params=args.__dict__,
        #target_filename=filename,
        #duration=t_after - t_before,
        evals=metrics,
        g_gt=target.g,
        interv_targets_gt=target.interv_targets.astype(int),
        g_empirical=expected_graph(dist_empirical, args.n_vars),
        g_mixture=expected_graph(dist_mixture, args.n_vars),
        I_empirical=expected_interv(dist_empirical) if args.infer_interv else 0,
        I_mixture=expected_interv(dist_mixture) if args.infer_interv else 0,
    )

    return results
