import common

import argparse

from gabriel_server import cognitive_engine
from gabriel_protocol import gabriel_pb2
import numpy as np
import cv2

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


def get_model_detection_function(model):
    """Get a tf.function for detection."""

    @tf.function
    def detect_fn(image):
      """Detect objects in image."""

      image, shapes = model.preprocess(image)
      prediction_dict = model.predict(image, shapes)
      detections = model.postprocess(prediction_dict, shapes)

      return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn


class InferenceEngine(cognitive_engine.Engine):
    def __init__(self, pipeline_path, checkpoint_path, label_map_file):

        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(pipeline_config)
        model_config = configs['model']
        detection_model = model_builder.build(
            model_config=model_config, is_training=False)

        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(checkpoint_path).expect_partial()

        self._detect_fn = get_model_detection_function(detection_model)
        ones = tf.ones(common.STARTUP_ONES_SIZE, dtype=tf.uint8)
        self._detect_fn(ones)

        label_map = label_map_util.load_labelmap(label_map_file)
        categories = label_map_util.convert_label_map_to_categories(
          label_map,
          max_num_classes=label_map_util.get_max_label_map_index(label_map),
          use_display_name=True)

        self._category_index = label_map_util.create_category_index(categories)

    def handle(self, input_frame):
        if input_frame.payload_type != gabriel_pb2.PayloadType.IMAGE:
            status = gabriel_pb2.ResultWrapper.Status.WRONG_INPUT_FORMAT
            return cognitive_engine.create_result_wrapper(status)

        np_data = np.fromstring(input_frame.payloads[0], dtype=np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_tensor = tf.convert_to_tensor(
            np.expand_dims(img, 0), dtype=tf.uint8)
        detections, predictions_dict, shapes = self._detect_fn(input_tensor)

        img_with_detections = img.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            img_with_detections,
            detections['detection_boxes'][0].numpy(),
            (detections['detection_classes'][0].numpy() + 1).astype(int),
            detections['detection_scores'][0].numpy(),
            self._category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.40,
            agnostic_mode=False)

        img_with_detections = cv2.cvtColor(
            img_with_detections, cv2.COLOR_RGB2BGR)
        _, jpeg_img = cv2.imencode('.jpg', img_with_detections)
        img_data = jpeg_img.tostring()

        result = gabriel_pb2.ResultWrapper.Result()
        result.payload_type = gabriel_pb2.PayloadType.IMAGE
        result.payload = img_data

        status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        result_wrapper = cognitive_engine.create_result_wrapper(status)
        result_wrapper.results.append(result)

        return result_wrapper


def main():
    common.configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument('pipeline_path', type=str)
    parser.add_argument('checkpoint_path', type=str)
    parser.add_argument('label_map_file', type=str)
    args = parser.parse_args()

    def engine_factory():
        return InferenceEngine(
            args.pipeline_path, args.checkpoint_path, args.label_map_file)

    common.run_engine(engine_factory)


if __name__ == '__main__':
    main()
