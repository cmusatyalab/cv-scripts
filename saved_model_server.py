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


class InferenceEngine(cognitive_engine.Engine):
    def __init__(self, saved_model_path, label_map_file):
        self._detect_fn = tf.saved_model.load(saved_model_path)
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

        np_data = np.frombuffer(input_frame.payloads[0], dtype=np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        detections = self._detect_fn(np.expand_dims(img, 0))

        img_with_detections = img.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            img_with_detections,
            detections['detection_boxes'][0].numpy(),
            detections['detection_classes'][0].numpy().astype(np.int32),
            detections['detection_scores'][0].numpy(),
            self._category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.40,
            agnostic_mode=False)

        img_with_detections = cv2.cvtColor(
            img_with_detections, cv2.COLOR_RGB2BGR)
        _, jpeg_img = cv2.imencode('.jpg', img_with_detections)
        img_data = jpeg_img.tobytes()

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
    parser.add_argument('saved_model_path', type=str)
    parser.add_argument('label_map_file', type=str)
    args = parser.parse_args()

    def engine_factory():
        return InferenceEngine(args.saved_model_path, args.label_map_file)

    common.run_engine(engine_factory)


if __name__ == '__main__':
    main()
