from typing import Type

import stable_baselines3.common.policies as sb3_policies

from mav_baselines.torch.recurrent_ppo.recurrent.policies import (
    RecurrentActorCriticCnnPolicy,
    RecurrentActorCriticPolicy,
    RecurrentMultiInputActorCriticPolicy,
)
from stable_baselines3.common.policies import BasePolicy


def _noop_register_policy(name: str, policy: Type[BasePolicy]) -> None:
    """Fallback for Stable-Baselines3 builds without ``register_policy``."""

    pass


register_policy = getattr(sb3_policies, "register_policy", _noop_register_policy)

MlpLstmPolicy = RecurrentActorCriticPolicy
CnnLstmPolicy = RecurrentActorCriticCnnPolicy
MultiInputLstmPolicy = RecurrentMultiInputActorCriticPolicy

register_policy("MlpLstmPolicy", RecurrentActorCriticPolicy)
register_policy("CnnLstmPolicy", RecurrentActorCriticCnnPolicy)
register_policy("MultiInputLstmPolicy", RecurrentMultiInputActorCriticPolicy)
