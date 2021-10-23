from gym.envs.registration import register as gym_register

env_list = []


def register(id, *args, **kwargs):  # pylint: disable=redefined-builtin
    assert id.startswith('POMDP-')
    assert id not in env_list
    env_list.append(id)
    gym_register(id, *args, **kwargs)
