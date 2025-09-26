from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, List, Tuple, Type, Union, Optional

import gymnasium as gym
import torch as th
from torch import nn
import warnings

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device


def _get_image_observation_space(
    observation_space: gym.spaces.Space,
    image_key: Optional[str] = None,
) -> gym.spaces.Box:
    """Select an image-like Box subspace from the observation space.

    Preference order:
    1) If ``image_key`` is provided and exists, return it.
    2) First subspace recognized by SB3 as image via ``is_image_space``.
    3) First 3D Box subspace (with a warning) as a fallback.
    Raises a ValueError if nothing suitable is found.
    """

    # Dict observation: try provided key, then auto-detect
    if isinstance(observation_space, gym.spaces.Dict):
        if image_key and image_key in observation_space.spaces:
            return observation_space.spaces[image_key]
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                return subspace
        for key, subspace in observation_space.spaces.items():
            if isinstance(subspace, gym.spaces.Box) and subspace.shape is not None and len(subspace.shape) == 3:
                warnings.warn(
                    (
                        "Observation key '{0}' is not recognised as an image by"
                        " stable-baselines3; treating it as channels-first"
                        " image for the recurrent encoder."
                    ).format(key),
                    stacklevel=2,
                )
                return subspace
        raise ValueError(
            "Encoder expects at least one image-like Box space in the Dict"
            f" observation space; available keys: {list(observation_space.spaces.keys())}"
        )

    # Single Box observation
    if isinstance(observation_space, gym.spaces.Box) and observation_space.shape is not None and len(observation_space.shape) == 3:
        if not is_image_space(observation_space):
            warnings.warn(
                "Observation space is treated as an image although it is not"
                " recognised as such by stable-baselines3 (check dtype/bounds).",
                stacklevel=2,
            )
        return observation_space

    # Nothing suitable found
    raise ValueError(
        "Encoder expects an image-like gym.spaces.Box with 3 dimensions (C, H, W)"
        " or a Dict containing such a subspace."
    )

class Encoder(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        features_dim: int = 64,
        image_key: Optional[str] = None,
    ):
        image_observation_space = _get_image_observation_space(observation_space, image_key=image_key)
        super(Encoder, self).__init__(image_observation_space, features_dim)
        n_input_channels = image_observation_space.shape[0]
        self.conv1 = nn.Conv2d(n_input_channels, 8, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
        # Compute shape by doing one forward pass
        # with th.no_grad():
        #     n_flatten = self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(th.as_tensor(image_observation_space.sample()[None][:, :1, :, :]).float())))))).shape
        self.linear = nn.Linear(2*2*256, features_dim)
        self.fc_logsigma = nn.Linear(2*2*256, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = nn.functional.relu(self.conv1(observations))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = nn.functional.relu(self.conv5(x))
        x = nn.functional.relu(self.conv6(x))
        mu = self.linear(x.view(observations.size(0), -1))
        # logsigma = self.fc_logsigma(x.view(observations.size(0), -1))
        # sigma = logsigma.exp()
        # eps = th.randn_like(sigma)
        # z = eps.mul(sigma).add_(mu)
        return mu
    
class Decoder(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        lstm_hidden_dim: int = 64,
        image_key: Optional[str] = None,
    ) -> None:
        super(Decoder, self).__init__()
        image_observation_space = _get_image_observation_space(observation_space, image_key=image_key)
        n_input_channels = image_observation_space.shape[0]
        self.fc = nn.Linear(lstm_hidden_dim, 2*2*256)
        self.deconv1 = nn.ConvTranspose2d(2*2*256, 128, kernel_size=5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2)
        self.deconv5 = nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2)
        self.deconv6 = nn.ConvTranspose2d(8, n_input_channels, kernel_size=4, stride=2)

    def forward(self, latent)-> th.Tensor:
        x = nn.functional.relu(self.fc(latent))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = nn.functional.relu(self.deconv1(x))
        x = nn.functional.relu(self.deconv2(x))
        x = nn.functional.relu(self.deconv3(x))
        x = nn.functional.relu(self.deconv4(x))
        x = nn.functional.relu(self.deconv5(x))
        reconstruction = th.sigmoid(self.deconv6(x))
        return reconstruction

class MultiExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observation spaces.
    Builds a feature extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    """

    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 64):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super(MultiExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                extractors[key] = Encoder(subspace, features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
                continue
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += 6

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)
