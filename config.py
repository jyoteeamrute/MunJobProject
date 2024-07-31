import os
import warnings
import tensorflow as tf
from dotenv import load_dotenv

def suppress_warnings():
    # Suppress various warnings
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations if needed
    warnings.filterwarnings("ignore", category=UserWarning)

    # Use tf.compat.v1 to reset the default graph (for TensorFlow 1.x compatibility)
    tf.compat.v1.reset_default_graph()

def load_env_variables():
    # Load environment variables with default values
    load_dotenv()  # Load environment variables from .env file
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "NEO4J_URI": os.getenv("NEO4J_URI"),
        "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME"),
        "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD")
    }

class Config:
    env_vars = load_env_variables()
    OPENAI_API_KEY = env_vars["OPENAI_API_KEY"]
    NEO4J_URI = env_vars["NEO4J_URI"]
    NEO4J_USERNAME = env_vars["NEO4J_USERNAME"]
    NEO4J_PASSWORD = env_vars["NEO4J_PASSWORD"]

# Call the suppress_warnings function to apply settings at the start of your program
suppress_warnings()
