"""
(c) Inria
"""

import hydra
import pyrootutils
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig

from planktonzilla.utils.logger import get_pylogger

log = get_pylogger(__name__)


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

@hydra.main(version_base="1.3", config_path=str(root / "configs"), config_name="train_node.yaml")
def main(cfg: DictConfig):
    if HydraConfig.get().mode != RunMode.MULTIRUN:
        log.error(f"Run mode is {HydraConfig.get().mode}, while RunMode.MULTIRUN was expected.")
        log.error("Command is meant launch an hydra multirun that launches experiments in SLURM clusters with hydra.")
    else:
        log.info(f"Success with params: {cfg.train_param}.")


if __name__ == "__main__":
    main()
