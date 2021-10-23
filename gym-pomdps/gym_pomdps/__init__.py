import re

from pkg_resources import resource_exists, resource_filename, resource_listdir

from .envs import *
from .envs.registration import env_list, register
from .wrappers import *

__version__ = '1.0.0'

extension = 'pomdp'


def list_pomdps():
    return list(env_list)


def is_pomdp(filename):  # pylint: disable=redefined-outer-name
    return filename.casefold().endswith(
        f'.{extension.casefold()}'
    ) and resource_exists('gym_pomdps.pomdps', filename)


for filename in (
    filename
    for filename in resource_listdir('gym_pomdps', 'pomdps')
    if filename.casefold().endswith(f'.{extension.casefold()}')
):
    path = resource_filename('gym_pomdps.pomdps', filename)
    name, _ = filename.rsplit('.', 1)  # remove .pomdp extension
    version = 0

    # extract version if any
    m = re.fullmatch(r'(?P<name>.*)\.v(?P<version>\d+)', name)
    if m is not None:
        name, version = m['name'], int(m['version'])

    with open(path) as f:
        text = f.read()

    register(
        id=f'POMDP-{name}-continuing-v{version}',
        entry_point='gym_pomdps.envs:POMDP',
        kwargs=dict(text=text, episodic=False),
    )

    register(
        id=f'POMDP-{name}-episodic-v{version}',
        entry_point='gym_pomdps.envs:POMDP',
        kwargs=dict(text=text, episodic=True),
    )
