# File: logger_utils.py
import logging
import logging.handlers
import sys
from pathlib import Path

# Import LOG_FILE from app_config, assuming it's in the same directory
from app_config import LOG_FILE

# Default log level, TradingBot will update its logger instance from config
DEFAULT_LOG_LEVEL = logging.INFO

def setup_logger() -> logging.Logger:
    """Sets up the application logger with console and file handlers."""
    log_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    logger = logging.getLogger("TradingBot")
    logger.setLevel(DEFAULT_LOG_LEVEL) # Set initial level

    if logger.hasHandlers():
        logger.handlers.clear()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(log_formatter)
    logger.addHandler(stdout_handler)

    try:
        log_dir = LOG_FILE.parent
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5*1024*1024, backupCount=3
        )
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"ERROR: Failed to set up file logging handler: {e}", file=sys.stderr)
        logging.basicConfig(level=logging.ERROR) # Basic logging if main logger fails
        logging.error(f"Failed to set up file logging handler: {e}", exc_info=True)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.WARNING)

    return logger
```

```python
