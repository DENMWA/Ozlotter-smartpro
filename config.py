"""
Configuration constants and settings for the enhanced ozlotter system
"""

OZLOTTO_NUMBERS = 7
OZLOTTER_NUMBERS = 7
MAX_NUMBER = 47
MIN_NUMBER = 1
DEFAULT_MUTATION_POOL_SIZE = 20

DEFAULT_POPULATION_SIZE = 100
DEFAULT_GENERATIONS = 50
DEFAULT_MUTATION_RATE = 0.1
ELITE_RETENTION_RATE = 0.2

DEFAULT_ENTROPY_WEIGHT = 0.3
DEFAULT_FREQUENCY_WEIGHT = 0.3
DEFAULT_GAP_WEIGHT = 0.2
DEFAULT_TEMPORAL_WEIGHT = 0.1
DEFAULT_CORRELATION_WEIGHT = 0.1

LSTM_SEQUENCE_LENGTH = 20
LSTM_HIDDEN_UNITS = 128
LSTM_DROPOUT_RATE = 0.2

DATA_DIR = "data"
SEED_FILE = f"{DATA_DIR}/elite_seeds.json"
DRAWS_FILE = f"{DATA_DIR}/historical_draws.csv"
MODEL_DIR = f"{DATA_DIR}/models"
WEIGHTS_FILE = f"{DATA_DIR}/optimized_weights.json"

MAX_RETRIES = 3
REQUEST_TIMEOUT = 10
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
]

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
