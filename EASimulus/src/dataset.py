import os
from collections import deque
import math
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple, Iterator, Union
from abc import ABC, abstractmethod
import cv2
import hydra
import psutil
import torch
from torch.utils.data import Dataset
import numpy as np
from loguru import logger

from episode import Episode
from utils import ObsModality
from utils.types import MultiModalObs
from utils.preprocessing import ImageObsProcessor, TokenObsProcessor, VectorObsProcessor

Batch = Dict[str, Union[torch.Tensor, MultiModalObs]]


class EpisodesDataset:
    def __init__(
        self,
        max_num_episodes: Optional[int] = None,
        name: Optional[str] = None,
        disk_checkpoint_dir: Optional[Path] = None,
    ) -> None:
        self.max_num_episodes = max_num_episodes
        self.name = name if name is not None else "dataset"
        self.num_seen_episodes = 0
        self.episodes = deque()
        self.episode_id_to_queue_idx = dict()
        self.newly_modified_episodes, self.newly_deleted_episodes = set(), set()

        if disk_checkpoint_dir is not None:
            self.load_disk_checkpoint(disk_checkpoint_dir)

    def __len__(self) -> int:
        return len(self.episodes)

    def clear(self) -> None:
        self.episodes = deque()
        self.episode_id_to_queue_idx = dict()

    def add_episode(self, episode: Episode) -> int:
        if (
            self.max_num_episodes is not None
            and len(self.episodes) == self.max_num_episodes
        ):
            self._popleft()
        episode_id = self._append_new_episode(episode)
        return episode_id

    def get_episode(self, episode_id: int) -> Episode:
        assert episode_id in self.episode_id_to_queue_idx
        queue_idx = self.episode_id_to_queue_idx[episode_id]
        return self.episodes[queue_idx]

    def update_episode(self, episode_id: int, new_episode: Episode) -> None:
        assert episode_id in self.episode_id_to_queue_idx
        queue_idx = self.episode_id_to_queue_idx[episode_id]
        merged_episode = self.episodes[queue_idx].merge(new_episode)
        self.episodes[queue_idx] = merged_episode
        self.newly_modified_episodes.add(episode_id)

    def add_video_to_dataset(
        self,
        video_path: Path,
        unknown_action_id: int,
        obs_processors,
        event_pred=False,
    ):
        if video_path:
            print("loading video from " + str(video_path))
            observations, actions, rewards, dones = [], [], [], []
            infos = [{}]
            last_reward = 0
            events = None
            if event_pred:
                events = []
                back = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            for path in sorted(video_path.glob("*")):
                image = cv2.imread(str(video_path.joinpath(path)))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if event_pred:
                    event = {}
                    mot = back.apply(image)
                    if len(events) > 0:
                        mot_close = (
                            cv2.morphologyEx(mot, cv2.MORPH_CLOSE, kernel) // 255
                        )
                        event[ObsModality.image] = mot_close[None, :]
                    else:
                        event[ObsModality.image] = np.zeros(
                            (
                                self.env.num_envs,
                                *self.obs[ObsModality.image.name].shape[-3:-1],
                            ),
                            dtype=np.uint8,
                        )
                    events.append(event)
                image = image[None, :]
                obs = {"image": image[None, :]}
                act = np.array([unknown_action_id])  # unknown action id
                totreward = int(str(path.name).split("_")[1].split(".")[0])
                rew = np.array([totreward - last_reward])
                last_reward = totreward
                done = np.array([0])
                observations.append(obs)
                actions.append(act)
                rewards.append(rew)
                dones.append(done)
            dones[-2] = np.array([1])
            actions[-1] = np.array([0])
        assert len(observations) == len(actions) == len(rewards) == len(dones)
        observations = {
            ObsModality[k]: np.swapaxes([o[k] for o in observations], 0, 1)
            for k in observations[0].keys()
        }
        actions, rewards, dones = map(
            lambda arr: np.swapaxes(arr, 0, 1), [actions, rewards, dones]
        )
        for i, (a, r, d) in enumerate(
            zip(*(actions, rewards, dones))
        ):  # Make everything (N, T, ...) instead of (T, N, ...)
            obs = {
                m: obs_processors[m].to_torch(observations[m][i])
                for m in obs_processors.keys()
            }

            episode = Episode(
                observations=obs,  # channel-first
                actions=torch.from_numpy(a),
                rewards=torch.from_numpy(r).float(),
                ends=torch.LongTensor(d),
                mask_padding=torch.ones(d.shape[0], dtype=torch.bool),
                last_info=infos[i],
            )
            self.add_episode(episode)

    def _popleft(self) -> Episode:
        id_to_delete = [k for k, v in self.episode_id_to_queue_idx.items() if v == 0]
        assert len(id_to_delete) == 1
        self.newly_deleted_episodes.add(id_to_delete[0])
        self.episode_id_to_queue_idx = {
            k: v - 1 for k, v in self.episode_id_to_queue_idx.items() if v > 0
        }
        return self.episodes.popleft()

    def _append_new_episode(self, episode):
        episode_id = self.num_seen_episodes
        self.episode_id_to_queue_idx[episode_id] = len(self.episodes)
        self.episodes.append(episode)
        self.num_seen_episodes += 1
        self.newly_modified_episodes.add(episode_id)
        return episode_id

    def sample_batch(
        self,
        batch_num_samples: int,
        sequence_length: int,
        weights: Optional[Tuple[float]] = None,
        sample_from_start: bool = True,
        context_len: int = 1,
    ) -> Batch:
        return self._collate_episodes_segments(
            self._sample_episodes_segments(
                batch_num_samples,
                sequence_length,
                weights,
                sample_from_start,
                context_len=context_len,
            )
        )

    def _sample_episodes_segments(
        self,
        batch_num_samples: int,
        sequence_length: int,
        weights: Optional[Tuple[float]],
        sample_from_start: bool,
        context_len: int = 1,
    ) -> List[Episode]:
        weights = np.array([max(0, len(e) - (context_len + 1)) for e in self.episodes])
        weights = weights / np.linalg.norm(weights)

        sampled_episodes = random.choices(
            self.episodes, k=batch_num_samples, weights=weights
        )

        sampled_episodes_segments = []
        for sampled_episode in sampled_episodes:
            if sample_from_start:
                start = random.randint(0, len(sampled_episode) - 1 - context_len)
                stop = start + sequence_length
            else:
                stop = random.randint(context_len + 1, len(sampled_episode))
                start = stop - sequence_length
            sampled_episodes_segments.append(
                sampled_episode.segment(start, stop, should_pad=True)
            )
            assert len(sampled_episodes_segments[-1]) == sequence_length
        return sampled_episodes_segments

    def _collate_episodes_segments(self, episodes_segments: List[Episode]) -> Batch:
        episodes_segments = [e_s.__dict__ for e_s in episodes_segments]
        batch = {}
        for k in episodes_segments[0]:
            if episodes_segments[0][k] != None:  # in case no event pred
                batch[k] = torch.stack([e_s[k] for e_s in episodes_segments])
        batch["observations"] = (
            batch["observations"].float() / 255.0
        )  # int8 to float and scale
        return batch

    def traverse(self, batch_num_samples: int, chunk_size: int, pad_dir: str = "right"):
        for episode in self.episodes:
            if pad_dir == "right":
                chunks = [
                    episode.segment(
                        start=i * chunk_size, stop=(i + 1) * chunk_size, should_pad=True
                    )
                    for i in range(math.ceil(len(episode) / chunk_size))
                ]
            else:
                assert pad_dir == "left"
                chunks = [
                    episode.segment(
                        start=min((i + 1) * chunk_size, len(episode)) - chunk_size,
                        stop=min((i + 1) * chunk_size, len(episode)),
                        should_pad=True,
                    )
                    for i in range(math.ceil(len(episode) / chunk_size))
                ]
            batches = [
                chunks[i * batch_num_samples : (i + 1) * batch_num_samples]
                for i in range(math.ceil(len(chunks) / batch_num_samples))
            ]
            for b in batches:
                yield self._collate_episodes_segments(b)

    def update_disk_checkpoint(self, directory: Path) -> None:
        assert directory.is_dir()
        for episode_id in self.newly_modified_episodes:
            episode = self.get_episode(episode_id)
            episode.save(directory / f"{episode_id}.pt")
        for episode_id in self.newly_deleted_episodes:
            (directory / f"{episode_id}.pt").unlink()
        self.newly_modified_episodes, self.newly_deleted_episodes = set(), set()

    def load_disk_checkpoint(self, directory: Path) -> None:
        assert (
            directory.is_dir() and len(self.episodes) == 0
        ), f"Expected '{directory}' to be a directory; Expected 0 episodes, got {len(self.episodes)}"
        episode_ids = sorted([int(p.stem) for p in directory.iterdir()])
        self.num_seen_episodes = episode_ids[-1] + 1
        for episode_id in episode_ids:
            episode = Episode.from_dict(
                torch.load(directory / f"{episode_id}.pt", weights_only=True)
            )
            self.episode_id_to_queue_idx[episode_id] = len(self.episodes)
            self.episodes.append(episode)


def _check_ram_usage_percentage(m):
    return psutil.virtual_memory().percent > m


def _check_ram_usage_g(m):
    return psutil.Process().memory_info()[0] / 2**30 > m


class EpisodesDatasetRamMonitoring(EpisodesDataset):
    """
    Prevent episode dataset from going out of RAM.
    Warning: % looks at system wide RAM usage while G looks only at process RAM usage.
    """

    def __init__(
        self,
        max_ram_usage: str,
        name: Optional[str] = None,
        disk_checkpoint_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(
            max_num_episodes=None, name=name, disk_checkpoint_dir=disk_checkpoint_dir
        )
        self.max_ram_usage = max_ram_usage
        self.num_steps = 0
        self.max_num_steps = None

        max_ram_usage = str(max_ram_usage)
        if max_ram_usage.endswith("%"):
            self.m = int(max_ram_usage.split("%")[0])
            assert 0 < m < 100
            self.check_ram_usage = _check_ram_usage_percentage
        else:
            assert max_ram_usage.endswith("G")
            self.m = float(max_ram_usage.split("G")[0])
            self.check_ram_usage = _check_ram_usage_g

    def clear(self) -> None:
        super().clear()
        self.num_steps = 0

    def add_episode(self, episode: Episode) -> int:
        if self.max_num_steps is None and self.check_ram_usage(self.m):
            self.max_num_steps = self.num_steps
            logger.info(f"Dataset RAM Threshold Reached! ({self.max_ram_usage})")
        self.num_steps += len(episode)
        while (self.max_num_steps is not None) and (
            self.num_steps > self.max_num_steps
        ):
            self._popleft()
        episode_id = self._append_new_episode(episode)
        return episode_id

    def _popleft(self) -> Episode:
        episode = super()._popleft()
        self.num_steps -= len(episode)
        return episode


class EpisodesDatasetManager:
    def __init__(
        self,
        episodes_dir: Path,
        max_num_episodes: Optional[int] = None,
        name: Optional[str] = None,
        sequence_length: int = 20,
        padding_strategy: str = "right",
    ) -> None:
        self.episodes_dir = episodes_dir
        self.max_num_episodes = max_num_episodes
        self.name = name if name is not None else "dataset"

        self.padding_strategy = padding_strategy
        self.sequence_length = sequence_length
        self._dataset_mngr = DirDEM(episodes_dir)
        assert self._dataset_mngr.__len__() == 0

    def __len__(self) -> int:
        return self._dataset_mngr.__len__()

    def clear(self) -> None:
        for ep_path in self._dataset_mngr.episodes_paths:
            assert ep_path.exists()
            os.remove(ep_path)
        self._dataset_mngr.refresh()

    def add_episode(self, episode: Episode) -> int:
        episode_id = self._append_new_episode(episode)
        return episode_id

    def get_episode(self, episode_id: int) -> Episode:
        assert episode_id < self._dataset_mngr.__len__()
        assert self._dataset_mngr.get_episode_path(
            episode_id
        ).name == self._get_episode_filename(episode_id)
        return self._dataset_mngr.get_episode(episode_id)

    def _get_episode_filename(self, episode_id: int) -> str:
        return f"{episode_id}.pt"

    def update_episode(self, episode_id: int, new_episode: Episode) -> None:
        assert episode_id == self._dataset_mngr.__len__() - 1
        merged_episode = self._dataset_mngr.get_episode(episode_id).merge(new_episode)
        episode_filename = self._get_episode_filename(episode_id)
        merged_episode.save(self._dataset_mngr.data_dir / episode_filename)
        self._dataset_mngr.update_episode(episode_filename)

    def _append_new_episode(self, episode):
        episode_id = self._dataset_mngr.__len__()
        episode_filename = self._get_episode_filename(episode_id)
        episode.save(self._dataset_mngr.data_dir / episode_filename)
        self._dataset_mngr.add_episode(episode_filename)
        return episode_id

    def get_iterable(self, batch_size: int, shuffle: bool = True):
        dataset = OfflineExperienceDataset(
            self._dataset_mngr,
            sequence_length=self.sequence_length,
            padding_strategy=self.padding_strategy,
        )
        dataset_iter = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
        return dataset_iter


class EpisodeDirManager:
    def __init__(
        self, episode_dir: Path, max_num_episodes: int, disable_saving: bool = False
    ) -> None:
        self.disable_saving = disable_saving
        self.episode_dir = episode_dir
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.max_num_episodes = max_num_episodes
        self.best_return = float("-inf")

    def save(self, episode: Episode, episode_id: int, epoch: int) -> None:
        if self.disable_saving:
            return

        if self.max_num_episodes is not None and self.max_num_episodes > 0:
            self._save(episode, episode_id, epoch)

    def _save(self, episode: Episode, episode_id: int, epoch: int) -> None:
        ep_paths = [
            p for p in self.episode_dir.iterdir() if p.stem.startswith("episode_")
        ]
        assert len(ep_paths) <= self.max_num_episodes
        if len(ep_paths) == self.max_num_episodes:
            to_remove = min(
                ep_paths, key=lambda ep_path: int(ep_path.stem.split("_")[1])
            )
            to_remove.unlink()
        episode.save(self.episode_dir / f"episode_{episode_id}_epoch_{epoch}.pt")

        ep_return = episode.compute_metrics().episode_return
        if ep_return > self.best_return:
            self.best_return = ep_return
            path_best_ep = [
                p for p in self.episode_dir.iterdir() if p.stem.startswith("best_")
            ]
            assert len(path_best_ep) in (0, 1)
            if len(path_best_ep) == 1:
                path_best_ep[0].unlink()
            episode.save(
                self.episode_dir / f"best_episode_{episode_id}_epoch_{epoch}.pt"
            )


class DatasetEpisodesManager(ABC):
    def __init__(self, context_length: int = 1):
        self.context_length = context_length
        (
            self.trajectory_lengths,
            self.trajectory_n_samples,
            self.trajectory_first_idx,
        ) = self._extract_metadata()
        self._n_total_samples = sum(self.trajectory_n_samples)

    @abstractmethod
    def get_episode(self, episode_index: int):
        pass

    def get_num_episodes(self):
        return len(self.trajectory_first_idx)

    def get_episode_first_sample_index(self, episode_index: int):
        return self.trajectory_first_idx[episode_index]

    def get_episode_num_samples(self, episode_index: int):
        return self.trajectory_n_samples[episode_index]

    @property
    def samples_count(self):
        return self._n_total_samples

    def __len__(self):
        return self.samples_count

    def get_episode_index(self, sample_index):
        # perform binary search to find the index within trajectory_first_idx
        n = self.get_num_episodes()
        l, r = 0, n - 1
        while l <= r:
            m = (l + r) // 2
            if self.get_episode_first_sample_index(m) > sample_index:
                r = m - 1
            else:
                if m + 1 < n:
                    if self.get_episode_first_sample_index(m + 1) > sample_index:
                        return m
                    else:
                        l = m + 1
                else:
                    assert m == n - 1
                    return m

    @abstractmethod
    def _get_episodes_lengths(self) -> tuple[int, ...]:
        pass

    def _extract_metadata(self):
        # each trajectory .pt file is a dict with keys 'observations', 'actions', 'rewards', 'ends', 'mask_padding'.
        # observations are uint8 type, with shape (batch, 3, 64, 64)
        # actions, rewards, ends, mask_padding shape: (batch,)

        # extract the number of trajectories and their lengths:
        trajectory_lengths = []
        trajectory_n_samples = []
        trajectory_first_idx = []
        for episode_length in self._get_episodes_lengths():
            trajectory_lengths.append(episode_length)

            if len(trajectory_first_idx) == 0:
                first_idx = 0
            else:
                first_idx = trajectory_first_idx[-1] + trajectory_n_samples[-1]
            trajectory_first_idx.append(first_idx)

            # We don't need the last/first (depending on the padding direction) example where
            # there's only padding and a single observation.
            effective_length = (
                episode_length - self.context_length + 1
            )  # +1 for termination obs
            # assert effective_length > 0, f"got {effective_length} ({episode_length} - {self.context_length} + 1)"
            trajectory_n_samples.append(effective_length)
        # Compute the number of samples in each trajectory
        # (each sample is a fixed length sequence of (obs, action, reward) tuples):
        return trajectory_lengths, trajectory_n_samples, trajectory_first_idx


class DirDEM(DatasetEpisodesManager):
    def __init__(self, data_dir: Path, context_length: int = 1):
        self.data_dir = data_dir
        self._trajectory_files = [
            fname for fname in sorted(self.data_dir.iterdir()) if fname.suffix == ".pt"
        ]

        super().__init__(context_length=context_length)

        logger.info(
            f"Got {len(self._trajectory_files)} trajectories, and {self._n_total_samples} total samples"
        )

    def _get_episodes_lengths(self) -> tuple[int, ...]:
        return tuple(
            [
                torch.load(self.data_dir / trajectory_file)["actions"].shape[0]
                - 1  # last action is padding
                for trajectory_file in self._trajectory_files
            ]
        )

    def get_episode(self, episode_index: int):
        trajectory_file = self._trajectory_files[episode_index]
        trajectory = torch.load(self.data_dir / trajectory_file)
        return Episode(**trajectory)

    def add_episode(self, episode_filename: str):
        episode_path = self.data_dir / episode_filename
        assert episode_path.exists()
        self._trajectory_files.append(episode_path.relative_to(self.data_dir))
        trajectory_length = torch.load(episode_path)["actions"].shape[0]
        first_idx = self.trajectory_first_idx[-1] + self.trajectory_n_samples[-1]

        self.trajectory_lengths.append(trajectory_length)
        self.trajectory_n_samples.append(trajectory_length - 1)
        self.trajectory_first_idx.append(first_idx)
        self._n_total_samples += self.trajectory_n_samples[-1]

    def update_episode(self, episode_filename: str):
        assert episode_filename == self._trajectory_files[-1].name

        trajectory_length = torch.load(self.data_dir / episode_filename)[
            "actions"
        ].shape[0]

        self.trajectory_lengths[-1] = trajectory_length
        n_samples = trajectory_length - 1
        samples_diff = n_samples - self.trajectory_n_samples[-1]
        self.trajectory_n_samples[-1] = n_samples
        self._n_total_samples += samples_diff

    def refresh(self):
        self._trajectory_files = [
            fname for fname in sorted(self.data_dir.iterdir()) if fname.suffix == ".pt"
        ]
        (
            self.trajectory_lengths,
            self.trajectory_n_samples,
            self.trajectory_first_idx,
        ) = self._extract_metadata()
        self._n_total_samples = sum(self.trajectory_n_samples)

    @property
    def episodes_paths(self):
        return [
            fname.absolute()
            for fname in sorted(self.data_dir.iterdir())
            if fname.suffix == ".pt"
        ]

    def get_episode_path(self, episode_index: int):
        return self._trajectory_files[episode_index].absolute()


class RAMDEM(DatasetEpisodesManager):
    def __init__(self, dataset: EpisodesDataset, context_length: int = 1):
        self.dataset = dataset
        super().__init__(context_length=context_length)

    def _get_episodes_lengths(self) -> tuple[int, ...]:
        return tuple(
            [
                episode.actions.shape[0] - 1  # last action is padding
                for episode in self.dataset.episodes
            ]
        )

    def get_episode(self, episode_index: int):
        return self.dataset.episodes[episode_index]


class OfflineExperienceDataset(Dataset):

    def __init__(
        self,
        episodes_manager: DatasetEpisodesManager,
        sequence_length: int,
        padding_strategy: str,
        obs_modalities: set[ObsModality],
    ):
        self.padding_strategy = padding_strategy
        self.sequence_length = sequence_length
        self.episodes_manager = episodes_manager
        self.obs_modalities = obs_modalities

        self.obs_processors = {}
        if ObsModality.image in self.obs_modalities:
            self.obs_processors[ObsModality.image] = ImageObsProcessor()
        if ObsModality.vector in self.obs_modalities:
            self.obs_processors[ObsModality.vector] = VectorObsProcessor()
        if ObsModality.token in self.obs_modalities:
            self.obs_processors[ObsModality.token] = TokenObsProcessor()
        if ObsModality.token_2d in self.obs_modalities:
            self.obs_processors[ObsModality.token_2d] = TokenObsProcessor()

        assert (
            set(self.obs_processors.keys()) == obs_modalities
        ), f'modality mismatch "{self.obs_modalities}" != {self.obs_processors.keys()}'

    def __len__(self):
        return self.episodes_manager.samples_count

    def __getitem__(self, sample_index):
        file_idx = self.episodes_manager.get_episode_index(sample_index)
        episode = self.episodes_manager.get_episode(file_idx)

        sample_index_in_file = (
            sample_index
            - self.episodes_manager.get_episode_first_sample_index(file_idx)
        )
        if self.padding_strategy == "right":
            start = sample_index_in_file
            stop = start + self.sequence_length
        else:
            assert self.padding_strategy == "left"
            stop = self.episodes_manager.context_length + 1 + sample_index_in_file
            start = stop - self.sequence_length

        example = episode.segment(start, stop, should_pad=True).__dict__.copy()
        del example["last_info"]
        if example["events"] == None:
            del example["events"]
        assert (
            example["mask_padding"].sum() > self.episodes_manager.context_length
        ), f"Failed with sample index {sample_index} (idx in episode {sample_index_in_file}) ep len: {len(episode)}"
        obs: dict[ObsModality, torch.Tensor] = example["observations"]
        for k, obs_k in obs.items():
            obs[k] = self.obs_processors[k](obs_k)
        example["observations"] = obs
        return example


class CuriousReplayDistribution:
    """
    Modified version based on https://arxiv.org/pdf/2306.15934
    """

    def __init__(
        self,
        uniform_portion: float = 0.7,
        c=1e4,
        alpha=0.7,
        beta=0.7,
        eps=0.01,
        p_max=1e5,
        replacement: bool = False,
        device=None,
    ):
        self.uniform_portion = uniform_portion
        self.device = device
        self.p_max = p_max
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.replacement = replacement
        self.counts = None
        self.losses = None
        self.last_sample = None

        self._loss_init_value = 10

    @property
    def distribution(self) -> torch.FloatTensor:
        # total_counts = self.counts.sum() + 1
        # p = self.c * (self.beta ** self.counts) + (torch.abs(self.losses) + self.eps) ** self.alpha

        # min_valid_loss = self.losses[torch.where(self.counts > 0)].min() if self.counts.sum() > 0 else 0
        # losses_shifted = torch.where(self.counts > 0, self.losses - min_valid_loss, torch.zeros_like(self.losses))
        # max_loss = losses_shifted.max()
        # if max_loss > 1e-6:
        #     losses_shifted = losses_shifted / max_loss
        # losses_shifted = torch.where(self.counts > 0, losses_shifted, torch.ones_like(self.losses))
        # losses_coef = 3
        # losses_bonus = torch.exp(losses_coef * losses_shifted)
        # losses_bonus = losses_bonus / losses_bonus.max()

        losses_bonus = torch.exp(self.losses)

        assert not torch.isnan(
            losses_bonus
        ).any(), f"got nan losses bonus: {losses_bonus}"
        assert not torch.isinf(
            losses_bonus
        ).any(), f"got inf losses bonus: {losses_bonus}"
        p = losses_bonus

        return p

    @property
    def num_samples(self) -> int:
        return self.losses.numel() if self.losses is not None else 0

    def sample(self, batch_size: int) -> torch.LongTensor:
        assert self.num_samples > 0
        replacement = False
        prioritized_bsz = math.ceil((1 - self.uniform_portion) * batch_size)
        prioritized_sample = torch.multinomial(
            self.distribution, prioritized_bsz, replacement=replacement
        )
        uniform_sample = torch.multinomial(
            torch.ones_like(self.distribution, device=self.device),
            batch_size - prioritized_bsz,
            replacement=replacement,
        )
        sample = torch.cat([prioritized_sample, uniform_sample], dim=0)
        sample = sample[
            torch.randperm(sample.size()[0], device=self.device)
        ]  # shuffle so that curiosity ensemble members will get random splits

        self.last_sample = sample.clone()
        self.counts[sample] += 1

        return sample

    def update_num_samples(self, new_num_of_samples: int):
        assert new_num_of_samples >= self.num_samples
        if new_num_of_samples == self.num_samples:
            return

        if self.counts is None:
            self.counts = torch.zeros(new_num_of_samples, device=self.device).long()
            self.losses = torch.ones_like(self.counts).float() * self._loss_init_value

        else:
            new_counts = torch.zeros(
                new_num_of_samples - self.num_samples, device=self.device
            ).long()
            new_losses = torch.ones_like(new_counts).float() * self._loss_init_value
            self.counts = torch.cat([self.counts, new_counts])
            self.losses = torch.cat([self.losses, new_losses])

    def update_losses(self, batch_losses: torch.FloatTensor):
        assert not torch.isnan(
            batch_losses
        ).any(), f"got at least one 'nan' loss value: {batch_losses}"
        assert not torch.isinf(
            batch_losses
        ).any(), f"got at least one 'inf' loss value: {batch_losses}"

        # device = self.losses[self.last_sample].device
        self.losses[self.last_sample] = batch_losses.to(self.device)


def np_softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
    return x

def np_2softmax(x,axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    x = np.power(2,x) / np.sum(np.power(2,x), axis=axis, keepdims=True)
    return x

class NpCuriousReplayDistribution:
    def __init__(
        self,
        uniform_portion: float = 0.7,
        replacement: bool = False,
    ):
        self.uniform_portion = uniform_portion
        self.replacement = replacement
        self.counts = None
        self.losses = None
        self.last_sample = None

        self._loss_init_value = 10

    @property
    def distribution(self) -> np.ndarray:
        loss_float64=self.losses.astype(np.float64) #in some cases that loss is big float32 is not enough.
        losses_bonus = np_softmax(loss_float64)

        assert not np.isnan(losses_bonus).any(), f"got nan losses bonus: {losses_bonus}"
        assert not np.isinf(losses_bonus).any(), f"got inf losses bonus: {losses_bonus}"
        p = losses_bonus

        return p

    @property
    def num_samples(self) -> int:
        return self.losses.size if self.losses is not None else 0

    def sample(self, batch_size: int) -> np.ndarray:
        assert self.num_samples > 0
        replacement = False
        prioritized_bsz = math.ceil((1 - self.uniform_portion) * batch_size)
        dist = self.distribution
        try:
            prioritized_sample = np.random.choice(
                dist.size, prioritized_bsz, replace=replacement, p=dist
            )
        except ValueError as ve:
            print(f"warning:{ve}")
            print("try power of 2")
            try:
                loss_float64=self.losses.astype(np.float64)
                dist = np_2softmax(loss_float64)
                prioritized_sample = np.random.choice(
                dist.size, prioritized_bsz, replace=replacement, p=dist
            )
            except ValueError as ve:
                raise ValueError(f"dist.size:{dist.size},self.losses:{loss_float64},prioritized_bsz:{prioritized_bsz};dist:{dist}")

        p = np.ones(dist.size)
        p[prioritized_sample] = 0
        p = p / p.sum()
        uniform_sample = np.random.choice(
            dist.size, batch_size - prioritized_bsz, replace=replacement, p=p
        )
        sample = np.concatenate([prioritized_sample, uniform_sample], axis=0)
        sample = np.random.permutation(
            sample
        )  # shuffle so that curiosity ensemble members will get random splits

        self.last_sample = sample
        self.counts[sample] += 1

        return sample

    def update_num_samples(self, new_num_of_samples: int):
        assert new_num_of_samples >= self.num_samples
        if new_num_of_samples == self.num_samples:
            return

        if self.counts is None:
            self.counts = np.zeros(new_num_of_samples, dtype=np.int32)
            self.losses = (
                np.ones_like(self.counts, dtype=np.float32) * self._loss_init_value
            )

        else:
            new_counts = np.zeros(new_num_of_samples - self.num_samples, dtype=np.int32)
            new_losses = (
                np.ones_like(new_counts, dtype=np.float32) * self._loss_init_value
            )
            self.counts = np.concatenate([self.counts, new_counts])
            self.losses = np.concatenate([self.losses, new_losses])

    def update_losses(self, batch_losses: np.ndarray):
        assert not np.isnan(
            batch_losses
        ).any(), f"got at least one 'nan' loss value: {batch_losses}"
        assert not np.isinf(
            batch_losses
        ).any(), f"got at least one 'inf' loss value: {batch_losses}"

        # device = self.losses[self.last_sample].device
        self.losses[self.last_sample] = batch_losses


class CuriousReplayBatchSampler(torch.utils.data.Sampler):

    def __init__(self, dist: CuriousReplayDistribution, batch_size: int) -> None:
        super().__init__(None)
        self.batch_size = batch_size
        self.dist = dist

    def __iter__(self):
        while True:
            yield self.dist.sample(self.batch_size)


def get_dataloader(
    dataset: EpisodesDataset,
    context_length: int,
    sequence_length: int,
    batch_size: int,
    shuffle: bool,
    padding_strategy: str,
    obs_modalities: set[ObsModality],
    replay_dist: Optional[NpCuriousReplayDistribution] = None,
    num_workers: int = 0,
    drop_last: bool = True,
):
    manager = RAMDEM(dataset, context_length=context_length)
    dataset = OfflineExperienceDataset(
        manager,
        sequence_length=sequence_length,
        padding_strategy=padding_strategy,
        obs_modalities=obs_modalities,
    )

    if replay_dist is not None:
        replay_dist.update_num_samples(manager.samples_count)
        batch_sampler = CuriousReplayBatchSampler(replay_dist, batch_size)
        dataset_iter = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=num_workers
        )
    else:
        dataset_iter = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

    return dataset_iter


def get_offline_dataset(
    datasets_dir_paths: list[Path],
    context_length: int,
    sequence_length: int,
    padding_strategy: str,
    obs_modalities: set[ObsModality],
    name: Optional[str] = None,
) -> Dataset:
    datasets = [
        OfflineExperienceDataset(
            DirDEM(
                data_dir=Path(hydra.utils.get_original_cwd()) / p,
                # EpisodesDataset(disk_checkpoint_dir=Path(hydra.utils.get_original_cwd()) / p, name=name),
                context_length=context_length,
            ),
            sequence_length=sequence_length,
            padding_strategy=padding_strategy,
            obs_modalities=obs_modalities,
        )
        for p in datasets_dir_paths
    ]

    return torch.utils.data.ConcatDataset(datasets)


def get_offline_datasets_dict(
    datasets_dir_paths_dict: dict[str, list[Path]],
    context_length: int,
    sequence_length: int,
    padding_strategy: str,
    obs_modalities: set[ObsModality],
    name: Optional[str] = None,
) -> dict[str, Dataset]:
    datasets_dict = {
        k: get_offline_dataset(
            v,
            context_length=context_length,
            sequence_length=sequence_length,
            padding_strategy=padding_strategy,
            obs_modalities=obs_modalities,
            name=f"{name}_{k}",
        )
        for k, v in datasets_dir_paths_dict.items()
    }
    return datasets_dict


def get_offline_dataloader(
    datasets_dir_paths: list[Path],
    context_length: int,
    sequence_length: int,
    batch_size: int,
    shuffle: bool,
    padding_strategy: str,
    obs_modalities: set[ObsModality],
):
    combined_dataset = get_offline_dataset(
        datasets_dir_paths=datasets_dir_paths,
        context_length=context_length,
        sequence_length=sequence_length,
        padding_strategy=padding_strategy,
        obs_modalities=obs_modalities,
    )

    dataset_iter = torch.utils.data.DataLoader(
        combined_dataset, batch_size=batch_size, shuffle=shuffle
    )

    return dataset_iter
