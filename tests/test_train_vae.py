"""Tests for module train_vae.py"""
import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorflow as tf

import autoencoder
import train_vae


class TestBuildModel:
    """Tests for train_vae.build_default_model."""
    # pylint: disable=no-self-use
    @pytest.fixture(name="created_model",
                    params=[(1, (1, 1, 1)), (2, (28, 28, 1)),
                            (100, (32, 32, 3)), (17, (12, 12, 4))])
    def created_model_fixture(self, request):
        """Model object created from build_default_model function."""
        return train_vae.build_default_model(*request.param)

    def test_has_encoder(self, created_model):
        """Test that model has an encoder."""
        assert isinstance(created_model.encoder, autoencoder.Encoder)

    def test_has_dcoder(self, created_model):
        """Test that model has a decoder."""
        assert isinstance(created_model.decoder, autoencoder.Decoder)

    def test_is_autoencoder(self, created_model):
        """Test that model is a VariationalAutoEncoder instance."""
        assert isinstance(created_model, autoencoder.VariationalAutoEncoder)


class TestFetchDatasets:
    """Tests for train_vae.fetch_datasets."""
    def test_raises(self):
        """Test that ValueError is raised for invalid dataset name."""
        name = "some invalid dataset name"
        with pytest.raises(ValueError, match=f"Dataset '{name}' unknown."):
            train_vae.fetch_datasets(name)

    @pytest.mark.parametrize("dataset", ["mnist", "cifar10"])
    def test_normalizes(self, dataset):
        """Test that datasets are normalized to have values in [0,1] range."""
        train_dataset, test_dataset, input_shape = train_vae.fetch_datasets(
            dataset)

        for image, _ in train_dataset:
            assert 0. <= image.numpy().min()
            assert image.numpy().max() <= 1.


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


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("elements,tensors,expected", [
    ([[0]], [[0]], [0]),
    ([[2]], [[0], [1], [4]], [1]),
    (np.zeros((11, 2)), np.arange(15 * 2).reshape(15, 2), np.zeros(11)),
    ([[0, 0], [1, 1], [-10, -10]], [[0, 0], [-11, -14], [.5, .6]], [0, 2, 1]),
])
def test_get_indices_of_closest_vectors(elements, tensors, dtype, expected):
    """Tests for train_vae.get_indices_of_closest_vectors."""
    result = train_vae.get_indices_of_closest_vectors(
        tf.constant(elements, dtype), tf.constant(tensors, dtype))

    assert result.dtype == tf.int64
    np.testing.assert_array_equal(result.numpy(), expected)
