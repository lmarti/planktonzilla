"""Configuration loading and parsing utilities for OOD experiments."""

import yaml
from pathlib import Path
from pytorch_ood.detector import EnergyBased


def resolve_config_path(config_name: str) -> Path:
    """
    Resolve config name to full path in configs directory.
    
    Args:
        config_name (str): Name of the config file (without path or extension)
        
    Returns:
        Path: Full path to config file
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file is not a YAML file
    """
    # Construct path to configs directory (relative to project root)
    project_root = Path(__file__).parent.parent.parent
    config_dir = project_root / "configs"
    config_path = config_dir / f"{config_name}.yaml"
    
    # Validate file exists
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Expected location: {config_dir}/{config_name}.yaml"
        )
    
    # Validate it's a YAML file
    if config_path.suffix not in ['.yaml', '.yml']:
        raise ValueError(
            f"Config file must be a YAML file (.yaml or .yml), got: {config_path.suffix}"
        )
    
    return config_path


def load_config(config_path: Path) -> dict:
    """
    Load YAML configuration file.
    
    Args:
        config_path (Path): Path to the YAML config file.
        
    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_ood_method_config(method_config: dict) -> dict:
    """
    Parse OOD method configuration and handle special cases.
    
    Converts string references in the config (e.g. 'EnergyBased.score')
    to actual function references.
    
    Args:
        method_config (dict): Method configuration from YAML
        
    Returns:
        dict: Processed method configuration with resolved references
    """
    method = {"name": method_config["name"], "cfg": method_config.get("config", {})}
    
    # Handle special detector references
    if method["name"] == "React" and "detector" in method["cfg"]:
        # Convert string reference to actual function
        if method["cfg"]["detector"] == "EnergyBased.score":
            method["cfg"]["detector"] = EnergyBased.score
    
    return method
