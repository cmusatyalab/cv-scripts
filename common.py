import argparse
import logging

from gabriel_server import local_engine


SOURCE_NAME = 'roundtrip'
INPUT_QUEUE_MAXSIZE = 60
PORT = 9099
NUM_TOKENS = 2
STARTUP_ONES_SIZE = (1, 480, 640, 3)


def configure_logging():
    logging.basicConfig(level=logging.INFO)


def run_engine(engine_factory):
    local_engine.run(
        engine_factory, SOURCE_NAME, INPUT_QUEUE_MAXSIZE, PORT, NUM_TOKENS)
