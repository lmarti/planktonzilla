"""Experiment logger with timestamped output and timer functionality."""

import logging
import time


class ExperimentLogger:
    """
    Logger for OOD detection experiments.
    
    Combines Python's logging module with timer functionality to provide
    timestamped log messages and elapsed time measurement for experiment stages.
    
    Args:
        name (str): Logger name. Default is 'ood_experiment'.
        level (int): Logging level. Default is logging.INFO.
    
    Example:
        >>> logger = ExperimentLogger()
        >>> logger.info("Starting experiment")
        [2026-02-12 23:30:00] [INFO] Starting experiment
        >>> logger.start_timer("data_loading")
        [2026-02-12 23:30:00] [INFO] Timer 'data loading' started
        >>> logger.end_timer("data_loading")
        [2026-02-12 23:30:05] [INFO] Timer 'data loading' elapsed: 0.08 minutes
    """
    
    def __init__(self, name: str = "ood_experiment", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Avoid duplicate handlers if logger already exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self._timers = {}
    
    def info(self, msg: str):
        """Log an info-level message."""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log a warning-level message."""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """Log an error-level message."""
        self.logger.error(msg)
    
    def start_timer(self, name: str):
        """
        Start a named timer and log the event.
        
        Args:
            name (str): Identifier for the timer (underscores are displayed as spaces).
        """
        self._timers[name] = time.time()
        display_name = " ".join(name.split("_"))
        self.logger.info(f"Timer '{display_name}' started")
    
    def end_timer(self, name: str):
        """
        End a named timer, log the elapsed time, and remove it.
        
        Args:
            name (str): Identifier for the timer (must match a previous start_timer call).
        """
        if name in self._timers:
            elapsed = time.time() - self._timers[name]
            display_name = " ".join(name.split("_"))
            self.logger.info(f"Timer '{display_name}' elapsed: {elapsed / 60:.2f} minutes")
            self._timers.pop(name)
        else:
            self.logger.warning(f"Timer '{name}' was never started")
