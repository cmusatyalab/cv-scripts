# Based on:
# https://github.com/cmusatyalab/opentpod-tools/blob/main/opentpod_tools/removedup.py


"""Remove similar frames based on a perceptual hash metric
"""


import os
import glob
import argparse
import shutil
import imagehash
from PIL import Image
import numpy as np
import tensorflow as tf


DIFF_THRESHOLD = 1

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
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
}

def create_tf_example(parsed):
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
    class_text = parsed['image/object/class/text'].values.numpy()[0]
    label = parsed['image/object/class/label'].numpy()[0]

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
            [label]),
    }))
    return tf_example

def checkDiff(image_hash, base_image_hash, threshold):
    if base_image_hash is None:
        return True
    if image_hash - base_image_hash >= threshold:
        return True

    return False


def checkDiffComplete(image_hash, base_image_list, threshold):
    if len(base_image_list) <= 0:
        return True
    for i in base_image_list:
        if not checkDiff(image_hash, i, threshold):
            return False
    return True


def main():
    base_image_list = []
    dup_count = 0

    input_dataset = tf.data.TFRecordDataset('input.tfrecord')
    writer = tf.io.TFRecordWriter('noduplicates.tfrecord')

    it = iter(input_dataset)
    for value in it:
        parsed = tf.io.parse_single_example(value, IMAGE_FEATURE_DESCRIPTION)
        tf_image = tf.image.decode_jpeg(parsed['image/encoded'])
        image = Image.fromarray(tf_image.numpy())
        image_hash = imagehash.phash(image)

        if checkDiffComplete(image_hash, base_image_list, DIFF_THRESHOLD):
            base_image_list.append(image_hash)
            tf_example = create_tf_example(parsed)
            writer.write(tf_example.SerializeToString())
        else:
            dup_count += 1

    writer.close()

    print("Total Dup: ", dup_count)


if __name__ == "__main__":
    main()
