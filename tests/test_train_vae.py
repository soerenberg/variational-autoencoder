"""Tests for module train_vae.py"""
import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorflow as tf

import train_vae


class TestExportImages:
    # pylint: disable=no-self-use
    """Tests for train_vae.export_images."""
    @pytest.fixture(params=[
        (.21 * tf.ones((4, 10, 10, 1)), tf.constant(3)),
        (.1 * tf.ones((4, 1, 1, 1)), tf.constant(4)),
        (.0 * tf.ones((4, 10, 10, 3)), tf.constant(7)),
        (.23 * tf.ones((4, 1, 1, 3)), tf.constant(31)),
        (.24 * tf.ones((16, 2, 2, 1)), tf.constant(1)),
    ],
                    name="path_imgs_step")
    def path_imgs_step_fixture(self, request, tmp_path):
        """Fixture for setting up test cases, saving images to temp dir."""
        images, global_step = request.param
        train_vae.export_images(images, tmp_path, global_step)

        return tmp_path, images, global_step

    def test_number_exported_files(self, path_imgs_step):
        """Test that the expected number of files was exported."""
        tmp_path, images, _ = path_imgs_step

        assert len(list(tmp_path.glob("image_*.png"))) == len(images)

    def test_recover_values(self, path_imgs_step):
        """Test that the correct values have been written."""
        # GIVEN exported images
        tmp_path, images, _ = path_imgs_step

        # WHEN images are recovered from disk
        name_images = sorted((path.name, plt.imread(path))
                             for path in tmp_path.glob("image_*.png"))

        for i, (name, image) in enumerate(name_images):
            # THEN the alpha channel is not opaque
            np.testing.assert_array_almost_equal(
                image[..., 3],
                np.ones((images.shape[1], images.shape[2])),
                err_msg=f"wrong alpha channel for {name}")

            # AND THEN the image can be recovered in the first 3 channels
            image = image[..., :3]  # ignore alpha channel
            if images.shape[-1] != 3:
                image = np.mean(image, axis=2, keepdims=True)
            np.testing.assert_array_almost_equal(
                np.floor(255. * image),
                np.floor(255. * images[i]),
                err_msg=f"cannot recover {name}")

    @pytest.mark.parametrize(
        "images,global_step,expected_files",
        [(tf.zeros((9, 10, 10, 1)), tf.constant(3),
          [f"image_{i}_step_00000003.png" for i in range(9)]),
         (tf.zeros((121, 2, 2, 3)), tf.constant(33),
          [f"image_{i:03}_step_00000033.png" for i in range(121)])])
    def test_file_names(self, tmp_path, images, global_step, expected_files):
        """Test that the filenames have the correct format."""
        train_vae.export_images(images, tmp_path, global_step)
        for path in tmp_path.glob("image_*.png"):
            assert path.name in expected_files

    @pytest.mark.parametrize(
        "images,global_step,expected_files",
        [(tf.zeros((8, 10, 10, 1)), tf.constant(3),
          [f"image_{i}_step_00000003.png" for i in range(9)]),
         (tf.zeros((111, 2, 2, 3)), tf.constant(33),
          [f"image_{i:03}_step_00000033.png" for i in range(121)])])
    def test_raises_if_no_square(self, tmp_path, images, global_step,
                                 expected_files):
        """Test that a ValueError is raised if batch sizes is not a square."""
        batch_size = images.shape[0]
        expected_message = (
            f"Number of example images must be a square, found {batch_size}.")
        with pytest.raises(ValueError, match=expected_message):
            train_vae.export_images(images, tmp_path, global_step)
