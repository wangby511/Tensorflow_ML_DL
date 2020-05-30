import os
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 2020.05.30 Triplet Loss
# tf.eye
# tf.expand_dims
# tf.reduce_max
# tf.cast(..., tf.bool)
#

def _pairwise_distances(embeddings, squared=False):
    """
    Compute pairwise distances
    Args:
        embeddings: 形如(batch_size, embed_dim)的张量
        squared: Boolean. True->欧式距离的平方，False->欧氏距离
    Returns:
        piarwise_distances: 形如(batch_size, batch_size)的张量
    """
    # 嵌入向量点乘，输出shape=(batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    square_norm = tf.diag_part(dot_product)

    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)

    distances = tf.maximum(distances, 0.0)

    if not squared:
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        distances = distances * (1.0 - mask)

    return distances

def _get_anchor_positive_triplet_mask(labels):
    # 返回一个2D掩码，掩码用于筛选合格的同类样本对[a, p]。合格的要求是：a和p是不同的样本索引，a和p具有相同的标签。
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    mask = tf.logical_and(indices_not_equal, labels_equal)
    return mask

def _get_anchor_negative_triplet_mask(labels):
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    labels_equal = tf.not_equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    mask = tf.logical_and(indices_not_equal, labels_equal)
    return mask


def test_get_anchor_positive_triplet_mask():
    a = tf.Variable([1,0,0,1,1])
    b = _get_anchor_positive_triplet_mask(a)
    c = tf.to_float(b)
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    print(sess.run(b))
    print(sess.run(c))
    # labels_equal
    # [[ x False False  True  True]
    #  [False  x  True False False]
    #  [False  True  x False False]
    #  [ True False False  x  True]
    #  [ True False False  True  x]]

# test_get_anchor_positive_triplet_mask()

def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    # I. Same label pairwise
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # Choose the maximum distance in each row, Output shape=(batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)

    # II. Negative label pairwise
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    # For each non-negative pairwise pairs, add distance of max_anchor_negative_dist
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
    # Choose the minimum distance in each row
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)

    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)
    triplet_loss = tf.reduce_mean(triplet_loss)
    return triplet_loss