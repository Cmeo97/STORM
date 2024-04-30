import numpy as np
import random
import unittest
import torch
from einops import rearrange
import copy
import pickle
import os

class ReplayBuffer():
    def __init__(self, obs_shape, num_envs, max_length=int(1E6), warmup_length=50000, store_on_gpu=False, device='cuda:0') -> None:
        self.store_on_gpu = store_on_gpu
        if store_on_gpu:
            self.obs_buffer = torch.empty((max_length//num_envs, num_envs, *obs_shape), dtype=torch.uint8, device=device, requires_grad=False)
            self.action_buffer = torch.empty((max_length//num_envs, num_envs), dtype=torch.float32, device=device, requires_grad=False)
            self.reward_buffer = torch.empty((max_length//num_envs, num_envs), dtype=torch.float32, device=device, requires_grad=False)
            self.termination_buffer = torch.empty((max_length//num_envs, num_envs), dtype=torch.float32, device=device, requires_grad=False)
        else:
            self.obs_buffer = np.empty((max_length//num_envs, num_envs, *obs_shape), dtype=np.uint8)
            self.action_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)
            self.reward_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)
            self.termination_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)

        self.length = 0
        self.num_envs = num_envs
        self.last_pointer = -1
        self.max_length = max_length
        self.warmup_length = warmup_length
        self.external_buffer_length = None

    def load_trajectory(self, path, image_size, device):
        buffer = pickle.load(open(path, "rb"))
        if buffer['obs'].shape[-2] != image_size:
            try:
                import cv2
            except ImportError:
                raise print(
                    "opencv is not install, run `pip install gym[other]`"
                )

            observation = cv2.resize(
                observation, (image_size, image_size), interpolation=cv2.INTER_AREA
            )
        if self.store_on_gpu:
            self.external_buffer = {name: torch.from_numpy(buffer[name]).to(device) for name in buffer}
        else:
            self.external_buffer = buffer
        self.external_buffer_length = self.length = self.external_buffer["obs"].shape[0]
        print('Trajectory lengh: ', self.external_buffer_length)

    def save_trajectory(self, path, kwargs):
        obj = {'obs': self.obs_buffer, 'actions': self.action_buffer, 'rewards': self.reward_buffer, 'terminations': self.termination_buffer}
        kwargs.setdefault('protocol', 4)
        file = path + '/trajectory.pkl'
        if not os.path.exists(file):
            os.makedirs(path)
        if file is None:
            return pickle.dumps(obj, **kwargs)
        elif isinstance(file, str):
            with open(file, 'wb') as f:
                pickle.dump(obj, f, **kwargs)
        elif hasattr(file, 'write'):
            pickle.dump(obj, file, **kwargs)
        else:
            raise TypeError('"file" must be a filename str or a file-object')

    def sample_external(self, batch_size, batch_length, device="cuda:0"):
        indexes = np.random.randint(0, self.external_buffer_length+1-batch_length, size=batch_size)
        if self.store_on_gpu:
            obs = torch.stack([self.external_buffer["obs"][idx:idx+batch_length] for idx in indexes])
            action = torch.stack([self.external_buffer["action"][idx:idx+batch_length] for idx in indexes])
            reward = torch.stack([self.external_buffer["reward"][idx:idx+batch_length] for idx in indexes])
            termination = torch.stack([self.external_buffer["done"][idx:idx+batch_length] for idx in indexes])
        else:
            obs = np.stack([self.external_buffer["obs"][idx:idx+batch_length] for idx in indexes])
            action = np.stack([self.external_buffer["action"][idx:idx+batch_length] for idx in indexes])
            reward = np.stack([self.external_buffer["reward"][idx:idx+batch_length] for idx in indexes])
            termination = np.stack([self.external_buffer["done"][idx:idx+batch_length] for idx in indexes])
        return obs, action, reward, termination

    def ready(self):
        return self.length * self.num_envs > self.warmup_length

    @torch.no_grad()
    def sample(self, batch_size, external_batch_size, batch_length, device="cuda:0"):
        if self.store_on_gpu:
            obs, action, reward, termination = [], [], [], []
            if batch_size > 0 and self.external_buffer_length is None:
                for i in range(self.num_envs):
                    indexes = np.random.randint(0, self.length+1-batch_length, size=batch_size//self.num_envs)
                    obs.append(torch.stack([self.obs_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    action.append(torch.stack([self.action_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    reward.append(torch.stack([self.reward_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    termination.append(torch.stack([self.termination_buffer[idx:idx+batch_length, i] for idx in indexes]))

            if self.external_buffer_length is not None and external_batch_size > 0:
                external_obs, external_action, external_reward, external_termination = self.sample_external(
                    external_batch_size, batch_length, device)
                obs.append(external_obs)
                action.append(external_action)
                reward.append(external_reward)
                termination.append(external_termination)

            obs = torch.cat(obs, dim=0).float() / 255
            obs = rearrange(obs, "B T H W C -> B T C H W")
            action = torch.cat(action, dim=0)
            reward = torch.cat(reward, dim=0)
            termination = torch.cat(termination, dim=0)
        else:
            obs, action, reward, termination = [], [], [], []
            if batch_size > 0:
                for i in range(self.num_envs):
                    indexes = np.random.randint(0, self.length+1-batch_length, size=batch_size//self.num_envs)
                    obs.append(np.stack([self.obs_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    action.append(np.stack([self.action_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    reward.append(np.stack([self.reward_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    termination.append(np.stack([self.termination_buffer[idx:idx+batch_length, i] for idx in indexes]))

            if self.external_buffer_length is not None and external_batch_size > 0:
                external_obs, external_action, external_reward, external_termination = self.sample_external(
                    external_batch_size, batch_length, device)
                obs.append(external_obs)
                action.append(external_action)
                reward.append(external_reward)
                termination.append(external_termination)

            obs = torch.from_numpy(np.concatenate(obs, axis=0)).float().to(device) / 255
            obs = rearrange(obs, "B T H W C -> B T C H W")
            action = torch.from_numpy(np.concatenate(action, axis=0)).to(device)
            reward = torch.from_numpy(np.concatenate(reward, axis=0)).to(device)
            termination = torch.from_numpy(np.concatenate(termination, axis=0)).to(device)

        return obs, action, reward, termination

    def append(self, obs, action, reward, termination):
        # obs/nex_obs: torch Tensor
        # action/reward/termination: int or float or bool
        self.last_pointer = (self.last_pointer + 1) % (self.max_length//self.num_envs)
        if self.store_on_gpu:
            self.obs_buffer[self.last_pointer] = torch.from_numpy(obs)
            self.action_buffer[self.last_pointer] = torch.from_numpy(action)
            self.reward_buffer[self.last_pointer] = torch.from_numpy(reward)
            self.termination_buffer[self.last_pointer] = torch.from_numpy(termination)
        else:
            self.obs_buffer[self.last_pointer] = obs
            self.action_buffer[self.last_pointer] = action
            self.reward_buffer[self.last_pointer] = reward
            self.termination_buffer[self.last_pointer] = termination

        if len(self) < self.max_length:
            self.length += 1

    def __len__(self):
        return self.length * self.num_envs
