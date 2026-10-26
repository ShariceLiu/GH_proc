"""Audit Phoneme source/speaker splits plus every saved model and prediction."""
import argparse
from threadpoolctl import threadpool_limits
import phoneme_plugin_experiment as experiment
from validate_gasoline_plugin import validate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=experiment.SEEDS)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        validate(experiment.SEEDS if args.seed is None else (args.seed,), experiment=experiment)
