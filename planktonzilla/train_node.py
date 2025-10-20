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

from omegaconf import DictConfig
import hydra
from planktonzilla.utils.hydra import (
    get_metric_value,
    task_wrapper,
)


@hydra.main(version_base="1.3", config_path=str(root / "configs"), config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    metric_dict, _ = train(cfg)

    metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    return metric_value


if __name__ == "__main__":
    main()