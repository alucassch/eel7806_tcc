# Copyright      2021  Piotr Żelasko
# Copyright      2022  Xiaomi Corporation     (Author: Mingshuang Luo)
# Copyright      2024  André Schlichting
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import inspect
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from lhotse import CutSet, load_manifest_lazy
from lhotse.dataset import (  # noqa F401 for PrecomputedFeatures
    CutConcatenate,
    DynamicBucketingSampler,
    K2SpeechRecognitionPhoneDataset,
    PrecomputedFeatures,
    SpecAugment,
)
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader


def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter

        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def make_cuts(manifest_dir: Union[Path, str]) -> CutSet:
    if isinstance(manifest_dir, str):
        manifest_dir = Path(manifest_dir)

    recs = load_manifest_lazy(manifest_dir / "recordings.jsonl.gz")
    sups = load_manifest_lazy(manifest_dir / "supervisions.jsonl.gz")
    feats = load_manifest_lazy(manifest_dir / "features.jsonl.gz")

    cutset = CutSet.from_manifests(
        recordings=recs, supervisions=sups, features=feats, random_ids=True
    )

    sup_dict = {}
    for sup in sups:
        sup_dict[sup.id] = sup

    for cut in cutset:
        storage_key = cut.features.storage_key
        # match supervision with cut by features.storage_key
        supervision = sup_dict[storage_key]
        # if duration is too different between supervision cut,
        # something wrong is happening
        sup_dur = supervision.duration
        if sup_dur - cut.duration > 0.05:
            raise ValueError("supervision duration and cut duration is too different.")

        # discard supervision.start inheritated from segments
        # and assign cut.duration and cut.start to supervision
        supervision.duration = cut.duration
        supervision.start = 0

        cut.supervisions.append(supervision)

    return cutset


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class AsrDataModule:
    """
    DataModule for k2 ASR experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. LibriSpeech test-clean
    and test-other).

    It contains all the common data pipeline modules used in ASR
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation,
    - augmentation,
    - on-the-fly feature extraction

    This class should be derived for specific corpora used in ASR tasks.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )
        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("manifests"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=600.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
            help="When enabled, the batches will come from buckets of "
            "similar duration (saves padding frames).",
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=30,
            help="The number of buckets for the DynamicBucketingSampler"
            "(you might want to increase it for larger datasets).",
        )
        group.add_argument(
            "--concatenate-cuts",
            type=str2bool,
            default=False,
            help="When enabled, utterances (cuts) will be concatenated "
            "to minimize the amount of padding.",
        )
        group.add_argument(
            "--duration-factor",
            type=float,
            default=1.0,
            help="Determines the maximum duration of a concatenated cut "
            "relative to the duration of the longest cut in a batch.",
        )
        group.add_argument(
            "--gap",
            type=float,
            default=0.5,
            help="The amount of padding (in seconds) inserted between "
            "concatenated cuts. This padding is filled with noise when "
            "noise augmentation is used.",
        )
        group.add_argument(
            "--on-the-fly-feats",
            type=str2bool,
            default=False,
            help="When enabled, use on-the-fly cut mixing and feature "
            "extraction. Will drop existing precomputed feature manifests "
            "if available.",
        )
        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help="When enabled (=default), the examples will be "
            "shuffled for each epoch.",
        )
        group.add_argument(
            "--drop-last",
            type=str2bool,
            default=True,
            help="Whether to drop last batch. Used by sampler.",
        )
        group.add_argument(
            "--return-cuts",
            type=str2bool,
            default=True,
            help="When enabled, each batch will have the "
            "field: batch['supervisions']['cut'] with the cuts that "
            "were used to construct it.",
        )

        group.add_argument(
            "--num-workers",
            type=int,
            default=4,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )

        group.add_argument(
            "--enable-spec-aug",
            type=str2bool,
            default=True,
            help="When enabled, use SpecAugment for training dataset.",
        )

        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=80,
            help="Used only when --enable-spec-aug is True. "
            "It specifies the factor for time warping in SpecAugment. "
            "Larger values mean more warping. "
            "A value less than 1 means to disable time warp.",
        )

        group.add_argument(
            "--input-strategy",
            type=str,
            default="PrecomputedFeatures",
            help="AudioSamples or PrecomputedFeatures",
        )

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
        """
        transforms = []

        if self.args.concatenate_cuts:
            logging.info(
                f"Using cut concatenation with duration factor "
                f"{self.args.duration_factor} and gap {self.args.gap}."
            )
            # Cut concatenation should be the first transform in the list,
            # so that if we e.g. mix noise in, it will fill the gaps between
            # different utterances.
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        input_transforms = []
        if self.args.enable_spec_aug:
            logging.info("Enable SpecAugment")
            logging.info(f"Time warp factor: {self.args.spec_aug_time_warp_factor}")
            # Set the value of num_frame_masks according to Lhotse's version.
            # In different Lhotse's versions, the default of num_frame_masks is
            # different.
            num_frame_masks = 10
            num_frame_masks_parameter = inspect.signature(
                SpecAugment.__init__
            ).parameters["num_frame_masks"]
            if num_frame_masks_parameter.default == 1:
                num_frame_masks = 2
            logging.info(f"Num frame mask: {num_frame_masks}")
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=num_frame_masks,
                    features_mask_size=27,
                    num_feature_masks=4,
                    frames_mask_size=100,
                )
            )
        else:
            logging.info("Disable SpecAugment")

        logging.info("About to create train dataset")
        train = K2SpeechRecognitionPhoneDataset(
            input_strategy=PrecomputedFeatures(),
            cut_transforms=transforms,
            input_transforms=input_transforms,
            return_cuts=self.args.return_cuts,
        )

        logging.info("Using DynamicBucketingSampler.")
        train_sampler = DynamicBucketingSampler(
            cuts_train,
            max_duration=self.args.max_duration,
            shuffle=self.args.shuffle,
            num_buckets=self.args.num_buckets,
            drop_last=self.args.drop_last,
            quadratic_duration=40,
        )
        logging.info("About to create train dataloader")

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        # 'seed' is derived from the current random state, which will have
        # previously been set in the main process.
        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)

        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        transforms = []
        if self.args.concatenate_cuts:
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        logging.info("About to create dev dataset")

        validate = K2SpeechRecognitionPhoneDataset(
            input_strategy=PrecomputedFeatures(),
            cut_transforms=transforms,
            return_cuts=self.args.return_cuts,
        )

        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        logging.info("About to create dev dataloader")
        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=2,
            persistent_workers=False,
        )

        return valid_dl

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.debug("About to create test dataset")
        test = K2SpeechRecognitionPhoneDataset(
            input_strategy=PrecomputedFeatures(),
            return_cuts=self.args.return_cuts,
        )
        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=False,
            num_buckets=self.args.num_buckets,
        )
        logging.debug("About to create test dataloader")
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )
        return test_dl

    @lru_cache()
    def train_cuts(self, train_manifests_dir: Union[Path, str]) -> CutSet:
        logging.info("About to get train cuts")

        if isinstance(train_manifests_dir, str):
            train_manifests_dir = Path(train_manifests_dir)

        train_cuts = (
            self.args.manifest_dir / train_manifests_dir / "train_cuts.jsonl.gz"
        )

        if not train_cuts.exists():
            logging.info("About to make train cuts")
            cutset = make_cuts(self.args.manifest_dir / train_manifests_dir)
            cutset.to_file(train_cuts)
            logging.info("Finished making train cuts")

        return load_manifest_lazy(
            self.args.manifest_dir / train_manifests_dir / "train_cuts.jsonl.gz"
        )

    @lru_cache()
    def dev_cuts(self, dev_manifests_dir: Union[Path, str]) -> CutSet:
        logging.info("About to get dev cuts")

        if isinstance(dev_manifests_dir, str):
            dev_manifests_dir = Path(dev_manifests_dir)

        dev_cuts = self.args.manifest_dir / dev_manifests_dir / "dev_cuts.jsonl.gz"

        if not dev_cuts.exists():
            logging.info("About to make dev cuts")
            cutset = make_cuts(self.args.manifest_dir / dev_manifests_dir)
            cutset.to_file(dev_cuts)
            logging.info("Finished making dev cuts")

        return load_manifest_lazy(
            self.args.manifest_dir / dev_manifests_dir / "dev_cuts.jsonl.gz"
        )

    @lru_cache()
    def test_cuts(self, test_manifests_dir: Union[Path, str]) -> CutSet:
        logging.info("About to get test cuts")

        if isinstance(test_manifests_dir, str):
            test_manifests_dir = Path(test_manifests_dir)

        test_cuts = self.args.manifest_dir / test_manifests_dir / "test_cuts.jsonl.gz"

        if not test_cuts.exists():
            logging.info("About to make test cuts")
            cutset = make_cuts(self.args.manifest_dir / test_manifests_dir)
            cutset.to_file(test_cuts)
            logging.info("Finished making test cuts")

        return load_manifest_lazy(
            self.args.manifest_dir / test_manifests_dir / "test_cuts.jsonl.gz"
        )
