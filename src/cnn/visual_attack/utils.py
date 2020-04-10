import numpy as np


def parse_ord(ord_str):
    if ord_str == 'inf':
        return np.inf
    else:
        return int(ord_str)


def set_attack_paths(args,
                     path_output_images_attack,
                     path_output_classes_attack,
                     path_output_features_attack):

    if args.attack_type == 'fgsm':
        params = {
            "eps": args.eps / 255,  #
            "clip_min": None,
            "clip_max": None,
            "ord": parse_ord(args.l),  #
            "y_target": None
        }

        if args.defense:
            path_args = args.dataset, \
                        args.model_dir, \
                        args.attack_type, \
                        args.origin_class, \
                        args.target_class, \
                        'eps' + str(args.eps), \
                        'it' + str(args.it), \
                        'l' + str(args.l), \
                        'XX'

        else:
            path_args = args.dataset, \
                        args.attack_type, \
                        args.origin_class, \
                        args.target_class, \
                        'eps' + str(args.eps), \
                        'it' + str(args.it), \
                        'l' + str(args.l), \
                        'XX'

    elif args.attack_type == 'pgd':
        params = {
            "eps": args.eps / 255,
            "eps_iter": args.eps / 255 / 6,  #
            "nb_iter": 10,  #
            "ord": parse_ord(args.l),  #
            "clip_min": None,
            "clip_max": None,
            "y_target": None,
            "rand_init": None,
            "rand_init_eps": None,
            "clip_grad": False,
            "sanity_checks": True
        }

        if args.defense:
            path_args = args.dataset, \
                        args.model_dir, \
                        args.attack_type, \
                        args.origin_class, \
                        args.target_class, \
                        'eps' + str(args.eps), \
                        'eps_it' + str(args.eps), \
                        'nb_it' + str(params["nb_iter"]), \
                        'l' + str(params["ord"])

        else:
            path_args = args.dataset, \
                        args.attack_type, \
                        args.origin_class, \
                        args.target_class, \
                        'eps' + str(args.eps), \
                        'eps_it' + str(args.eps), \
                        'nb_it' + str(params["nb_iter"]), \
                        'l' + str(params["ord"])

    elif args.attack_type == 'cw':
        # 'n_classes': 1000
        params = {'max_iterations': 1000, 'learning_rate': 5e-3,
                  'binary_search_steps': 5, 'confidence': 0,
                  'abort_early': True, 'initial_const': 1e-2,
                  'y_target': args.target_class
                  # ,      'clip_min': None, 'clip_max': None
                  }

        if args.defense:
            path_args = args.dataset, \
                        args.model_dir, \
                        args.attack_type, \
                        args.origin_class, \
                        args.target_class, \
                        'conf' + str(params["confidence"]), \
                        'lr' + str(params["learning_rate"]), \
                        'c' + str(params["initial_const"]), \
                        'max_it' + str(params["max_iterations"])

        else:
            path_args = args.dataset, \
                        args.attack_type, \
                        args.origin_class, \
                        args.target_class, \
                        'conf' + str(params["confidence"]), \
                        'lr' + str(params["learning_rate"]), \
                        'c' + str(params["initial_const"]), \
                        'max_it' + str(params["max_iterations"])

    elif args.attack_type == 'jsma':
        params = {
            "theta": 1.0,  #
            "gamma": 1.0,  #
            # "clip_min": args.clip_min,
            # "clip_max": args.clip_max,
            "y_target": None,
            "symbolic_impl": True  #
        }

        if args.defense:
            path_args = args.dataset, \
                        args.model_dir, \
                        args.attack_type, \
                        args.origin_class, \
                        args.target_class, \
                        'th' + str(params["theta"]), \
                        'ga' + str(params["gamma"]), \
                        'symb' + str(params["symbolic_impl"]), \
                        'XX'

        else:
            path_args = args.dataset, \
                        args.attack_type, \
                        args.origin_class, \
                        args.target_class, \
                        'th' + str(params["theta"]), \
                        'ga' + str(params["gamma"]), \
                        'symb' + str(params["symbolic_impl"]), \
                        'XX'

    elif args.attack_type == 'zoo':
        params = {
            "theta": 1.0,  #
            "gamma": 1.0,  #
            # "clip_min": args.clip_min,
            # "clip_max": args.clip_max,
            "y_target": None,
            "symbolic_impl": True  #
        }

        if args.defense:
            path_args = args.dataset, \
                        args.model_dir, \
                        args.attack_type, \
                        args.origin_class, \
                        args.target_class, \
                        'th' + str(params["theta"]), \
                        'ga' + str(params["gamma"]), \
                        'symb' + str(params["symbolic_impl"]), \
                        'XX'

        else:
            path_args = args.dataset, \
                        args.attack_type, \
                        args.origin_class, \
                        args.target_class, \
                        'th' + str(params["theta"]), \
                        'ga' + str(params["gamma"]), \
                        'symb' + str(params["symbolic_impl"]), \
                        'XX'

    elif args.attack_type == 'spsa':
        params = {
            "eps": args.eps / 255,
            "nb_iter": args.nb_iter,
            "y_target": None
        }

        if args.defense:
            path_args = args.dataset, \
                        args.model_dir, \
                        args.attack_type, \
                        args.origin_class, \
                        args.target_class, \
                        'eps' + str(args.eps), \
                        'delta0.01', \
                        'nb_iter' + str(args.nb_iter), \
                        'XX'

        else:
            path_args = args.dataset, \
                        args.attack_type, \
                        args.origin_class, \
                        args.target_class, \
                        'eps' + str(args.eps), \
                        'delta0.01', \
                        'nb_iter' + str(args.nb_iter), \
                        'XX'

    else:
        raise NotImplementedError('Unknown attack type.')

    path_output_images_attack = path_output_images_attack.format(*path_args)
    path_output_classes_attack = path_output_classes_attack.format(*path_args)
    path_output_features_attack = path_output_features_attack.format(*path_args)

    return params, path_output_images_attack, path_output_classes_attack, path_output_features_attack
