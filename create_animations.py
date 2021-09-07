"""Script to create animations from images."""
import argparse
import logging
import pathlib
from typing import Any, List

from PIL import Image
from tqdm import tqdm


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
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Verbosity level. Repeat to increase. '-vvv' equals logging info."
    )
    return parser.parse_args()


def load_ordered_images(image_dir: pathlib.Path, image_id: str) -> List[Any]:
    """Load images for a given id in order.

    Args:
        image_dir: top directory of images
        image_id: id of the image, e.g. "00023"

    Returns:
        list of load PIL Images.
    """
    ordered_paths = sorted(
        (path.stem, path)
        for path in image_dir.glob(f"image_{image_id}_*.png"))

    def read_and_load(filepath):
        """Read from file and load into memory."""
        with Image.open(filepath) as png_image:
            png_image.load()  # load to avoid having to many files open.
        return png_image

    return [read_and_load(t[1]) for t in ordered_paths]


def run() -> None:
    """Run script."""
    parsed_args = parse_cmd_line_args()
    set_up_logging(parsed_args.verbose)

    model_dir = pathlib.Path(parsed_args.model_dir)

    image_dir = model_dir / "images"

    image_ids = set(
        path.stem.split("_")[1] for path in image_dir.glob("image_*.png"))

    logging.info("Read images from file")
    ordered_images_series = [(iid, load_ordered_images(image_dir, iid))
                             for iid in tqdm(image_ids)]

    logging.info("Create animations")
    for iid, (first_img, *tail_imgs) in tqdm(ordered_images_series):
        first_img.save(fp=image_dir / f"image_{iid}.gif",
                       format="GIF",
                       append_images=tail_imgs,
                       save_all=True,
                       duration=200,
                       loop=0)


if __name__ == "__main__":
    run()
