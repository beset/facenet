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
def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.
    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    import numpy as np
    if len(face_encodings) == 0:
        print('encoding 等于零')
        return np.empty((0))
    distance = np.linalg.norm(face_encodings - face_to_compare, axis=1)
    print('距离：')
    print(distance)
    return distance

def _chinese_whispers(encoding_list, threshold=0.75, iterations=20):
    """ Chinese Whispers Algorithm

    Modified from Alex Loveless' implementation,
    http://alexloveless.co.uk/data/chinese-whispers-graph-clustering-in-python/

    Inputs:
        encoding_list: a list of facial encodings from face_recognition
        threshold: facial match threshold,default 0.6
        iterations: since chinese whispers is an iterative algorithm, number of times to iterate

    Outputs:
        sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
            sorted by largest cluster to smallest
    """

    from random import shuffle
    import networkx as nx
    # Create graph
    nodes = []
    edges = []

    image_paths, encodings = zip(*encoding_list)

    if len(encodings) <= 1:
        print ("No enough encodings to cluster!")
        return []

    for idx, face_encoding_to_check in enumerate(encodings):
        # Adding node of facial encoding
        node_id = idx+1

        # Initialize 'cluster' to unique value (cluster of itself)
        node = (node_id, {'cluster': image_paths[idx], 'path': image_paths[idx]})
        nodes.append(node)

        # Facial encodings to compare
        if (idx+1) >= len(encodings):
            # Node is last element, don't create edge
            break

        compare_encodings = encodings[idx+1:]
        distances = face_distance(compare_encodings, face_encoding_to_check)
        encoding_edges = []
        for i, distance in enumerate(distances):
            if distance < threshold:
                # Add edge if facial match
                edge_id = idx+i+2
                encoding_edges.append((node_id, edge_id, {'weight': distance}))

        edges = edges + encoding_edges

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Iterate
    for _ in range(0, iterations):
        cluster_nodes = G.nodes()
        shuffle(cluster_nodes)
        for node in cluster_nodes:
            neighbors = G[node]
            clusters = {}

            for ne in neighbors:
                if isinstance(ne, int):
                    if G.node[ne]['cluster'] in clusters:
                        #该节点邻居节点的类别的权重
                        #对应上面的字典cluster的意思就是
                        #对应的某个路径下文件的权重
                        clusters[G.node[ne]['cluster']] += G[node][ne]['weight']
                    else:
                        clusters[G.node[ne]['cluster']] = G[node][ne]['weight']

            # find the class with the highest edge weight sum
            edge_weight_sum = 0
            max_cluster = 0
            #将邻居节点的权重最大值对应的文件路径给到当前节点
            for cluster in clusters:
                if clusters[cluster] > edge_weight_sum:
                    edge_weight_sum = clusters[cluster]
                    max_cluster = cluster

            # set the class of target node to the winning local class
            G.node[node]['cluster'] = max_cluster

    clusters = {}

    # Prepare cluster output
    for (_, data) in G.node.items():
        cluster = data['cluster']
        path = data['path']

        if cluster:
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(path)

    # Sort cluster output
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)

    return sorted_clusters

def cluster_facial_encodings(facial_encodings):
    """ Cluster facial encodings

        Intended to be an optional switch for different clustering algorithms, as of right now
        only chinese whispers is available.

        Input:
            facial_encodings: (image_path, facial_encoding) dictionary of facial encodings

        Output:
            sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
                sorted by largest cluster to smallest

    """

    if len(facial_encodings) <= 1:
        print ("Number of facial encodings must be greater than one, can't cluster")
        return []

    # Only use the chinese whispers algorithm for now
    sorted_clusters = _chinese_whispers(facial_encodings.items())
    return sorted_clusters

def compute_facial_encodings(sess,images_placeholder,embeddings,phase_train_placeholder,image_size,
                	embedding_size,nrof_images,nrof_batches,emb_array,batch_size,paths):
    """ Compute Facial Encodings

        Given a set of images, compute the facial encodings of each face detected in the images and
        return them. If no faces, or more than one face found, return nothing for that image.

        Inputs:
            image_paths: a list of image paths

        Outputs:
            facial_encodings: (image_path, facial_encoding) dictionary of facial encodings

    """

    for i in range(nrof_batches):
        start_index = i*batch_size
        end_index = min((i+1)*batch_size, nrof_images)
        paths_batch = paths[start_index:end_index]
        images = facenet.load_data(paths_batch, False, False, image_size)
        feed_dict = { images_placeholder:images, phase_train_placeholder:False }
        print('feed_dict 是什么。。。。')
        print(feed_dict)
        emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
    print("result emb_array:")
    print(emb_array)
    facial_encodings = {}
    for x in range(nrof_images):
        facial_encodings[paths[x]] = emb_array[x,:]
    print("facial_encodings:")
    print(facial_encodings)
    return facial_encodings

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
    print(args.output)
    if not exists(args.output):
        print('创建文件夹：%s' % args.output)
        makedirs(args.output)
    else:
        print("文件夹已存在")

    with tf.Graph().as_default():
        with tf.Session() as sess:
            train_set = facenet.get_dataset(args.input)

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
            # Run forward pass to calculate embeddings
            print('Runnning forward pass on images')


            print('train set count: %d' % len(train_set))
            for x in range(len(train_set)):
                image_paths = train_set[x].image_paths
                print(image_paths)
                nrof_images = len(image_paths)
                nrof_batches = int(math.ceil(1.0*nrof_images / args.batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                print('emb_array:')
                print(emb_array)
                facial_encodings = compute_facial_encodings(sess,images_placeholder,embeddings,phase_train_placeholder,image_size,
                	embedding_size,nrof_images,nrof_batches,emb_array,args.batch_size,image_paths)
                sorted_clusters = cluster_facial_encodings(facial_encodings)
                num_cluster = len(sorted_clusters)
                print('created %d cluster!' % num_cluster)
                sure_count = 0
                for idx,cluster in enumerate(sorted_clusters):
                    print('%d th cluster num :%d' % (idx,len(cluster)))
                    sure_count = sure_count + len(cluster)
                print('sure count: %d' % sure_count)
                dest_dir = join(args.output, train_set[x].name)
                print(dest_dir)
                if not exists(dest_dir):
                    makedirs(dest_dir)
                # Copy image files to cluster folders
                for idx, cluster in enumerate(sorted_clusters):
                    #这个是保存聚类后所有类别
                    cluster_dir = join(dest_dir, str(idx))

                    if not exists(cluster_dir):
                        makedirs(cluster_dir)
                    for path in cluster:
                        imagename = os.path.basename(path).replace('png', 'jpeg')
                        original_path = "/home/mgc/realman/others/" +  imagename
                        if os.path.exists(original_path):
                            shutil.copy(original_path, join(cluster_dir, basename(original_path)))
                        shutil.copy(path, join(cluster_dir, basename(path)))
    
def parse_args():
    """Parse input arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='Get a shape mesh (t-pose)')
    parser.add_argument('--model_dir', type=str, help='model dir', required=True)
    parser.add_argument('--batch_size', type=int, help='model dir', required=30)
    parser.add_argument('--input', type=str, help='Input dir of images', required=True)
    parser.add_argument('--output', type=str, help='Output dir of clusters', required=True)
    # parser.add_argument('--threshold', type=float, help='threshold', required=0.75)
    # parser.add_argument('--iterations', type=int, help='iterations', required=20)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    """ Entry point """
    main(parse_args())

