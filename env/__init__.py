import gym
import sys
from .ant import AntTruncatedObsEnv
from .humanoid import HumanoidTruncatedObsEnv
from .swimmer import SwimmerTruncatedEnv
MBPO_ENVIRONMENT_SPECS = (
	{
        'id': 'AntTruncatedObsEnv-v2',
        'entry_point': AntTruncatedObsEnv,
    },
	 {
         'id': 'HumanoidTruncatedObsEnv-v2',
         'entry_point': HumanoidTruncatedObsEnv,
     },
    {
         'id': 'SwimmerTruncatedEnv-v2',
         'entry_point': SwimmerTruncatedEnv,
     },
)

def register_mbpo_environments():
    for mbpo_environment in MBPO_ENVIRONMENT_SPECS:
        gym.register(**mbpo_environment)


    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in MBPO_ENVIRONMENT_SPECS)

    return gym_ids