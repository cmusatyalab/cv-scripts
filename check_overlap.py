from PIL import Image

import tensorflow as tf

import imagehash


IMAGE_FEATURE_DESCRIPTION = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
}


def main():
    hashes = set()
    dataset = tf.data.TFRecordDataset('train.tfrecord')

    for value in dataset:
        parsed = tf.io.parse_single_example(value, IMAGE_FEATURE_DESCRIPTION)
        tf_image = tf.image.decode_jpeg(parsed['image/encoded'])
        image = Image.fromarray(tf_image.numpy())

        hashes.add(imagehash.phash(image))

    dataset = tf.data.TFRecordDataset('test.tfrecord')
    duplicates = 0
    unique = 0

    for value in dataset:
        parsed = tf.io.parse_single_example(value, IMAGE_FEATURE_DESCRIPTION)
        tf_image = tf.image.decode_jpeg(parsed['image/encoded'])
        image = Image.fromarray(tf_image.numpy())

        if imagehash.phash(image) in hashes:
            duplicates += 1
        else:
            unique += 1


    print('duplicates:', duplicates)
    print('unique:', unique)


if __name__ == '__main__':
    main()
