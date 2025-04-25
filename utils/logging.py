# logging_config.py

import os
import logging

def setup_logging():
    """
    Configures the logging system based on environment variables.
    This function is called once during the program's startup to configure logging settings.
    """
    # Get the DISABLE_LOGGING flag from environment variables
    disable_logging_str = os.getenv("DISABLE_LOGGING", "False").strip().lower()
    disable_logging = disable_logging_str == "true"  # Convert the string to a boolean

    # Initialize logger
    logger = logging.getLogger("WeatherAgent")
    logger.setLevel(logging.CRITICAL)  # Default to CRITICAL to suppress logs if disabled

    # Remove all existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    if disable_logging:
        # Disable all logging by setting the log level to CRITICAL
        logger.setLevel(logging.CRITICAL)  # This will suppress all logs
    else:
        # Get the VERBOSE_LOGGING flag from the environment variables
        verbose_logging = os.getenv("VERBOSE_LOGGING", "False").strip().lower() == "true"
        
        # Set logging level based on verbose flag
        log_level = logging.DEBUG if verbose_logging else logging.INFO
        logger.setLevel(log_level)

    # Set up the logging handler and formatter
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
