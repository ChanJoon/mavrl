""" Some data loading utilities """
from bisect import bisect
from os import listdir, path
from os.path import join, isdir
from tqdm import tqdm
import torch
import torch.utils.data
import numpy as np
import rosbag
from cv_bridge import CvBridge
import cv2
import rospy

class _RolloutDataset(torch.utils.data.Dataset): # pylint: disable=too-few-public-methods
    def __init__(self, root, transform, buffer_size=200, train=True): # pylint: disable=too-many-arguments
        self._transform = transform

        self._files = [
            join(root, sd, ssd)
            for sd in listdir(root) if isdir(join(root, sd))
            for ssd in listdir(join(root, sd))]

        if train:
            self._files = self._files[:-600]
        else:
            self._files = self._files[-600:]

        self._cum_size = None
        self._buffer = None
        self._buffer_fnames = None
        self._buffer_index = 0
        self._buffer_size = buffer_size

    def load_next_buffer(self):
        """ Loads next buffer """
        self._buffer_fnames = self._files[self._buffer_index:self._buffer_index + self._buffer_size]
        self._buffer_index += self._buffer_size
        self._buffer_index = self._buffer_index % len(self._files)
        self._buffer = []
        self._cum_size = [0]

        # progress bar
        pbar = tqdm(total=len(self._buffer_fnames),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        pbar.set_description("Loading file buffer ...")

        for f in self._buffer_fnames:
            with np.load(f) as data:
                self._buffer += [{k: np.copy(v) for k, v in data.items()}]
                self._cum_size += [self._cum_size[-1] +
                                   self._data_per_sequence(data['rewards'].shape[0])]
            pbar.update(1)
        pbar.close()

    def __len__(self):
        # to have a full sequence, you need self.seq_len + 1 elements, as
        # you must produce both an seq_len obs and seq_len next_obs sequences
        if not self._cum_size:
            self.load_next_buffer()
        return self._cum_size[-1]

    def __getitem__(self, i):
        # binary search through cum_size
        file_index = bisect(self._cum_size, i) - 1
        seq_index = i - self._cum_size[file_index]
        data = self._buffer[file_index]
        return self._get_data(data, seq_index)

    def _get_data(self, data, seq_index):
        pass

    def _data_per_sequence(self, data_length):
        pass

class RosbagDataset(torch.utils.data.Dataset):
    def __init__(self, rosbag_folder, image_topic, buffer_size=10, transform=None, train=True):

        self.image_topic = image_topic
        self.transform = transform
        self.bag_paths = [
            join(rosbag_folder, sd, ssd)
            for sd in listdir(rosbag_folder) if isdir(join(rosbag_folder, sd))
            for ssd in listdir(join(rosbag_folder, sd))]
        if train:
            self.bag_paths = self.bag_paths[:-5]
        else:
            self.bag_paths = self.bag_paths[-5:]

        self.bridge = CvBridge()
        self._cum_size = None
        self._buffer = None
        self._buffer_fnames = None
        self._buffer_index = 0
        self._buffer_size = buffer_size
        # rospy.init_node('rosbag_dataset', anonymous=True)

    def load_next_buffer(self):
        """ Loads next buffer """
        self._buffer_fnames = self.bag_paths[self._buffer_index:(self._buffer_index + self._buffer_size)]
        self._buffer_index += self._buffer_size
        self._buffer_index = self._buffer_index % len(self.bag_paths)
        self._buffer = []
        self._cum_size = [0]

        # progress bar
        pbar = tqdm(total=len(self._buffer_fnames),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        pbar.set_description("Loading file buffer ...")

        for f in self._buffer_fnames:
            with rosbag.Bag(f, "r") as bag:
                current_data = []
                for _, msg, _ in bag.read_messages(topics=[self.image_topic]):
                    try:
                        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") / 1000.0
                        cv_image = (np.minimum(cv_image, 12.0)) / 12.0 * 255.0
                        shape = cv_image.shape
                        cv_image = cv_image[:, int((shape[1]-shape[0])/2 - 1) : int((shape[1]+shape[0])/2 - 1)]
                        dim = (256, 256)
                        cv_image = cv2.resize(cv_image, dim, interpolation=cv2.INTER_AREA)
                        # convert to uint8
                        cv_image = cv_image.astype(np.uint8)
                        if self.transform:
                            cv_image = self.transform(cv_image)
                        current_data += [cv_image]
                    except Exception as e:
                        print(f"Error extracting image: {str(e)}")
                self._cum_size += [self._cum_size[-1] + len(current_data)]
                self._buffer.append(current_data)
            pbar.update(1)
        pbar.close()
    
    def __len__(self):
        # to have a full sequence, you need self.seq_len + 1 elements, as
        # you must produce both an seq_len obs and seq_len next_obs sequences
        if not self._cum_size:
            self.load_next_buffer()
        return self._cum_size[-1]
                
    def __getitem__(self, i):
        # binary search through cum_size
        file_index = bisect(self._cum_size, i) - 1
        seq_index = i - self._cum_size[file_index]
        data = self._buffer[file_index]
        return self._get_data(data, seq_index)

    # def _data_per_sequence(self, data_length):
    #     return data_length

    def _get_data(self, data, seq_index):
        
        return data[seq_index]
    
class RosbagSequenceDataset(RosbagDataset):
    def __getitem__(self, index):
        return np.array(self._buffer[index]).squeeze()
    
    def __len__(self):
        return len(self._buffer)

class RolloutLSTMSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, root, device, train=True):
        self._files = [
            join(root, sd, ssd)
            for sd in listdir(root) if isdir(join(root, sd))
            for ssd in listdir(join(root, sd))]
        print("len(self._files): ", len(self._files))
        if train:
            self._files = self._files[:-10]
        else:
            self._files = self._files[-10:]

        self._cum_size = None
        self._buffer = None
        self._buffer_fnames = None
        self._buffer_index = 0
        self._buffer_size = 20
        self.device = device

    def load_next_buffer(self):
        """ Loads next buffer """
        self._buffer_fnames = self._files[self._buffer_index:self._buffer_index + self._buffer_size]
        self._buffer_index += self._buffer_size
        self._buffer_index = self._buffer_index % len(self._files)
        self._buffer = []
        self._cum_size = 0

        # progress bar
        # pbar = tqdm(total=len(self._buffer_fnames),
        #             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        # pbar.set_description("Loading file buffer ...")

        for f in self._buffer_fnames:
            with np.load(f, allow_pickle=True) as data:
                # if v is also a dict, then we need to copy each element of v, else just copy v
                self._buffer += [{k: np.copy(v) if type(v) is not dict else {kk: np.copy(vv) for kk, vv in v.items()} for k, v in data.items()}]
                self._cum_size += 1
        #     pbar.update(1)
        # pbar.close()

    def __len__(self):
        if not self._cum_size:
            self.load_next_buffer()
        return len(self._buffer)
    
    def __getitem__(self, i):
        # print(self._buffer[i].keys())
        observations = {key: torch.tensor(obs, dtype=torch.float32, device=self.device) for (key, obs) in self._buffer[i]['observations'].item().items()}
        lstm_states = (torch.tensor(self._buffer[i]['lstm_states'][0], dtype=torch.float32, device=self.device), 
                       torch.tensor(self._buffer[i]['lstm_states'][1], dtype=torch.float32, device=self.device))
        episode_starts = torch.tensor(self._buffer[i]['episode_starts'], dtype=torch.float32, device=self.device)
        return observations, lstm_states, episode_starts
    
class RolloutSequenceDataset(_RolloutDataset): # pylint: disable=too-few-public-methods
    """ Encapsulates rollouts.

    Rollouts should be stored in subdirs of the root directory, in the form of npz files,
    each containing a dictionary with the keys:
        - observations: (rollout_len, *obs_shape)
        - actions: (rollout_len, action_size)
        - rewards: (rollout_len,)
        - terminals: (rollout_len,), boolean

     As the dataset is too big to be entirely stored in rams, only chunks of it
     are stored, consisting of a constant number of files (determined by the
     buffer_size parameter).  Once built, buffers must be loaded with the
     load_next_buffer method.

    Data are then provided in the form of tuples (obs, action, reward, terminal, next_obs):
    - obs: (seq_len, *obs_shape)
    - actions: (seq_len, action_size)
    - reward: (seq_len,)
    - terminal: (seq_len,) boolean
    - next_obs: (seq_len, *obs_shape)

    NOTE: seq_len < rollout_len in moste use cases

    :args root: root directory of data sequences
    :args seq_len: number of timesteps extracted from each rollout
    :args transform: transformation of the observations
    :args train: if True, train data, else test
    """
    def __init__(self, root, seq_len, transform, buffer_size=200, train=True): # pylint: disable=too-many-arguments
        super().__init__(root, transform, buffer_size, train)
        self._seq_len = seq_len

    def _get_data(self, data, seq_index):
        obs_data = data['observations'][seq_index:seq_index + self._seq_len + 1]
        obs_data = self._transform(obs_data.astype(np.float32))
        obs, next_obs = obs_data[:-1], obs_data[1:]
        action = data['actions'][seq_index+1:seq_index + self._seq_len + 1]
        action = action.astype(np.float32)
        reward, terminal = [data[key][seq_index+1:
                                      seq_index + self._seq_len + 1].astype(np.float32)
                            for key in ('rewards', 'terminals')]
        # data is given in the form
        # (obs, action, reward, terminal, next_obs)
        return obs, action, reward, terminal, next_obs

    def _data_per_sequence(self, data_length):
        return data_length - self._seq_len

class RolloutObservationDataset(_RolloutDataset): # pylint: disable=too-few-public-methods
    """ Encapsulates rollouts.

    Rollouts should be stored in subdirs of the root directory, in the form of npz files,
    each containing a dictionary with the keys:
        - observations: (rollout_len, *obs_shape)
        - actions: (rollout_len, action_size)
        - rewards: (rollout_len,)
        - terminals: (rollout_len,), boolean

     As the dataset is too big to be entirely stored in rams, only chunks of it
     are stored, consisting of a constant number of files (determined by the
     buffer_size parameter).  Once built, buffers must be loaded with the
     load_next_buffer method.

    Data are then provided in the form of images

    :args root: root directory of data sequences
    :args seq_len: number of timesteps extracted from each rollout
    :args transform: transformation of the observations
    :args train: if True, train data, else test
    """
    def _data_per_sequence(self, data_length):
        return data_length

    def _get_data(self, data, seq_index):
        return self._transform(data['observations'][seq_index])
    