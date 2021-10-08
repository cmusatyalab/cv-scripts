# Scripts for tfrecord files

* `merge_tfrecords.py` will combine the data from multiple tfrecord files into
  one file.
* `verify_tfrecord.py` will draw bounding boxes on images one at a time. Looking
  at a few of these images is a good sanity check before you start training.
* `checkpoint_server.py` will start a Gabriel server to process frames using an
  object detector from checkpoint files.
* `saved_model_server.py` will start a Gabriel server to process frames using an
  object detector from saved model files.
* `remove_dup.py` creates a new version of a tfrecord file where each image has
  a unique perceptual hash value.

## Usage

Build the Docker container for TensorFlow Object detection by following
[these](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md#docker-installation)
instructions.

Start the container with a volume map to the directory with your checkpoint
files. Convert a checkpoint to a saved model by running:
```bash
python object_detection/exporter_main_v2.py --input_type image_tensor --pipeline_config_path <PATH TO pipeline.config> \
--trained_checkpoint_dir <Path to model_dir> --output_directory <Where to save model>
```
from `~/models/research` in the container.

The two server scripts in this repository work with the clients in
[this](https://github.com/cmusatyalab/gabriel/tree/master/examples/round_trip)
directory.

Run `python3 -m pip install imagehash` to use `remove_dup.py`.
