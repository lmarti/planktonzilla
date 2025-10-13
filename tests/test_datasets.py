"""
(c) Inria
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig

from planktonzilla.dataset import DatasetWrapper

from .shared import dataset_names, skip_in_github_ci


@pytest.mark.parametrize("dataset_name", dataset_names)
def test_dataset_instantiation(hydra_conf_path: Path, dataset_name: str):
    with initialize(version_base=None, config_path=hydra_conf_path):
        # config is relative to a module
        cfg: DictConfig = compose(
            config_name="train",
            overrides=[f"dataset={dataset_name}", "extras.print_config=False", "extras.enforce_tags=False"],
        )

        dataset: DatasetWrapper = instantiate(cfg.dataset)
        assert dataset


@skip_in_github_ci
@pytest.mark.parametrize("dataset_name", dataset_names)
def test_dataset_prepare_datasets(hydra_conf_path: Path, dataset_name: str):
    with initialize(version_base=None, config_path=hydra_conf_path):
        # config is relative to a module
        cfg: DictConfig = compose(
            config_name="train",
            overrides=[f"dataset={dataset_name}", "extras.print_config=False", "extras.enforce_tags=False"],
        )

        dataset_wrapper: DatasetWrapper = instantiate(cfg.dataset)
        dataset_wrapper.prepare_datasets(None)
