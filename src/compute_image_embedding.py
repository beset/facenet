# -*- coding:utf-8 -*-
""" Face Cluster """
import tensorflow as tf
import numpy as np
import importlib
import argparse
import facenet
import os
import math
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def compute_facial_encoding(sess,images_placeholder,embeddings,phase_train_placeholder,image_size,
                	embedding_size,emb_array,image_path):

    images = facenet.load_data([image_path], False, False, image_size)
    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
    print('feed_dict 是什么。。。。')
    print(feed_dict)
    # emb_array[1,:] = sess.run(embeddings, feed_dict=feed_dict)


    print("result:")
    result = sess.run(embeddings, feed_dict=feed_dict)
    print result

def main(args):
    """ Main

    Given a list of images, save out facial encoding data files and copy
    images into folders of face clusters.

    """
    from os.path import join, basename, exists
    from os import makedirs
    import numpy as np
    import shutil
    import sys

    with tf.Graph().as_default():
        with tf.Session() as sess:

            facenet.load_model(args.model_dir)
            print('model loaded')
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image_size = images_placeholder.get_shape()[1]
            print(image_size)
            embedding_size = embeddings.get_shape()[1]
            print(embedding_size)

            image_path = args.image_path

            emb_array = np.zeros((1, embedding_size))
            print('emb_array:')
            print(emb_array)
            facial_encoding = compute_facial_encoding(sess,images_placeholder,embeddings,phase_train_placeholder,image_size,
            	embedding_size,emb_array,image_path)
    
def parse_args():
    """Parse input arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='Get a shape mesh (t-pose)')
    parser.add_argument('--model_dir', type=str, help='model dir', required=True)
    parser.add_argument('--image_path', type=str, help='image path', required=True)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    """ Entry point """
    main(parse_args())

