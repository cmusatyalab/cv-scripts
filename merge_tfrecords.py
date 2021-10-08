import os

import tensorflow as tf

from object_detection.protos import string_int_label_map_pb2
from object_detection.utils import dataset_util

from google.protobuf import text_format

import glob


DEFAULT_CLASS_TEXT = 'default'

TFRECORD_NAME = 'single_class.tfrecord'
LABEL_MAP_NAME = 'single_class_label_map.pbtxt'

IMAGE_FEATURE_DESCRIPTION = {
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/source_id': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    # 'image/object/class/label': tf.io.VarLenFeature(tf.int64),
}


def create_tf_example(parsed, class_text, class_num):
    height = parsed['image/height'].numpy()
    width = parsed['image/width'].numpy()
    filename = parsed['image/filename'].numpy()
    source_id = parsed['image/source_id'].numpy()
    encoded_image_data = parsed['image/encoded'].numpy()
    image_format = parsed['image/format'].numpy()
    xmins = [value.numpy() for value in parsed['image/object/bbox/xmin'].values]
    xmaxs = [value.numpy() for value in parsed['image/object/bbox/xmax'].values]
    ymins = [value.numpy() for value in parsed['image/object/bbox/ymin'].values]
    ymaxs = [value.numpy() for value in parsed['image/object/bbox/ymax'].values]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(source_id),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(
            [class_text.encode('utf8')]),
        'image/object/class/label': dataset_util.int64_list_feature(
            [class_num]),
    }))
    return tf_example


def main():
    class_num = 1
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    writer = tf.io.TFRecordWriter(TFRECORD_NAME)

    item = string_int_label_map_pb2.StringIntLabelMapItem()
    item.id = class_num
    item.name = DEFAULT_CLASS_TEXT
    label_map.item.append(item)

    for dataset_dir in glob.glob(os.path.join('input', '*')):
        print(dataset_dir)
        full_dataset_path = os.path.join(dataset_dir, 'default.tfrecord')
        dataset = tf.data.TFRecordDataset(full_dataset_path)

        it = iter(dataset)
        for value in it:
            parsed = tf.io.parse_single_example(
                value, IMAGE_FEATURE_DESCRIPTION)
            num_values = len(parsed['image/object/class/text'].values)
            if num_values == 0:
                continue
            if num_values != 1:
                raise Exception

            tf_example = create_tf_example(
                parsed, DEFAULT_CLASS_TEXT, class_num)
            writer.write(tf_example.SerializeToString())

    writer.close()
    with open(LABEL_MAP_NAME, 'w') as f:
        f.write(text_format.MessageToString(label_map))


if __name__ == '__main__':
    main()
