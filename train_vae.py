"""Execute one or more training steps from the command line."""
import argparse
import logging
import pathlib

import model


def set_up_logging(rank: int) -> None:
    """Get Logger and set verbosity level.

    Args:
        rank: level to set, must be in {0, 1, ..., 5}
    """
    logger = logging.getLogger()
    level = 50 - 10 * rank
    logger.setLevel(level=level)


def parse_cmd_line_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        action="store",
        dest="model_dir",
        required=True,
        help="Root directory for saving checkpoints and events.")
    parser.add_argument("--latent_dim",
                        action="store",
                        dest="latent_dim",
                        type=int,
                        default=2,
                        help="Latent dimension of the VAE. Defaults to 2.")
    parser.add_argument("--num_epochs",
                        action="store",
                        dest="num_epochs",
                        type=int,
                        default=1,
                        help="Num epochs to train. Defaults to 1.")
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Verbosity level. Repeat to increase. '-vvv' equals logging info."
    )
    return parser.parse_args()


def run() -> None:
    """Execute training step(s) for the VAE model."""
    parsed_args = parse_cmd_line_args()
    train_dataset, test_dataset = model.get_datasets()
    model.train_model(train_dataset=train_dataset,
                      test_dataset=test_dataset,
                      num_epochs=parsed_args.num_epochs,
                      latent_dim=parsed_args.latent_dim,
                      model_dir=pathlib.Path(parsed_args.model_dir),
                      check_pt_every_n_epochs=1)


if __name__ == "__main__":
    run()
