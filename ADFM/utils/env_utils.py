import copy
def create_env(args, render_mode='particle'):
    from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
    assert args.env_name == 'ClothDrop'

    env_args = copy.deepcopy(env_arg_dict[args.env_name])  # Default args
    env_args['cached_states_path'] = args.cached_states_path
    env_args['num_variations'] = args.num_variations

    env_args['render'] = True
    env_args['headless'] = True
    env_args['render_mode'] = render_mode
    env_args['observation_mode'] = args.observation_mode

    env_args['camera_name'] = 'default_camera'
    env_args['camera_width'] = 360
    env_args['camera_height'] = 360

    env_args['num_picker'] = args.num_picker # 2 pickers
    env_args['picker_radius'] = args.picker_radius
    env_args['picker_threshold'] = args.picker_threshold
    env_args['action_repeat'] = args.action_repeat
    env_args['env_shape'] = args.env_shape

    env_args['vary_cloth_size'] = args.vary_cloth_size
    env_args['vary_stiffness'] = args.vary_stiffness
    env_args['vary_orientation'] = args.vary_orientation
    env_args['vary_mass'] = args.vary_mass
    if 'target_type' in args:
        env_args['target_type'] = args.target_type

    return SOFTGYM_ENVS[args.env_name](**env_args)

def create_env_plan(args):
    from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
    # assert args.env_name == 'ClothDrop'

    env_args = copy.deepcopy(env_arg_dict[args.env_name])  # Default args
    env_args['cached_states_path'] = args.cached_states_path
    env_args['num_variations'] = args.num_variations

    env_args['render'] = True
    env_args['headless'] = True
    env_args['render_mode'] = 'cloth'
    env_args['observation_mode'] = args.observation_mode

    env_args['camera_name'] = 'default_camera'
    env_args['camera_width'] = 360
    env_args['camera_height'] = 360

    env_args['num_picker'] = args.num_picker # 2 pickers
    env_args['picker_radius'] = args.picker_radius
    env_args['picker_threshold'] = args.picker_threshold
    env_args['action_repeat'] = args.action_repeat
    env_args['env_shape'] = args.env_shape
    env_args['vary_cloth_size'] = args.vary_cloth_size
    env_args['vary_stiffness'] = args.vary_stiffness
    env_args['vary_orientation'] = args.vary_orientation
    env_args['vary_mass'] = args.vary_mass
    env_args['target_type'] = args.target_type

    env = SOFTGYM_ENVS[args.env_name](**env_args)

    render_env_kwargs = copy.deepcopy(env_args)
    render_env_kwargs['render_mode'] = 'particle'
    render_env = SOFTGYM_ENVS[args.env_name](**render_env_kwargs)

    return env, render_env