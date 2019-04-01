import tensorflow as tf


def triplet_loss_batch_all(embeddings, labels, margin=.1):
    """
        embeddings: (batch_size, feature_dims)
        labels: (batch_size,)
    """
    with tf.name_scope("triplet_loss"):
        labels_num = tf.shape(labels)[0]
        pairwise_distances = _pairwise_distances(embeddings)

        # reshape labels to (batch_size, 1)
        labels = tf.reshape(labels, shape=(labels_num, 1))

        # (a, p, n), where a=anchor, p=positive, n=negative
        anchor_positive = tf.expand_dims(pairwise_distances, 2)
        anchor_negative = tf.expand_dims(pairwise_distances, 1)

        triplet_loss = tf.maximum(anchor_positive - anchor_negative + margin, 0.0)

        mask_valid_triplets = tf.cast(_triplet_mask(labels), dtype=tf.float32)

        triplet_loss = tf.multiply(triplet_loss, mask_valid_triplets)
        triplet_loss = _reduce_triplet_loss_sum(triplet_loss, labels_num)

        return triplet_loss


def triplet_loss_batch_hard(embeddings, labels, margin=.1):
    """
    Triplet mining for hard positive and hard negative triplets.
        embeddings: (batch_size, feature_dims)
        labels: (batch_size,)
    """
    with tf.name_scope("triplet_loss"):
        pairwise_distances = _pairwise_distances(embeddings)

        labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        labels_not_equal = tf.logical_not(labels_equal)

        # pick hard positives
        anchor_positive = tf.cast(labels_equal, dtype=tf.float32) * pairwise_distances
        hard_positives = tf.reduce_max(anchor_positive, axis=1)

        # pick hard negatives
        mask_negative = tf.cast(labels_not_equal, dtype=tf.float32)
        max_negative = tf.reduce_max(pairwise_distances, axis=1)
        anchor_negative = pairwise_distances + max_negative * (1. - mask_negative)
        hard_negatives = tf.reduce_min(anchor_negative, axis=1)

        tf.summary.scalar('hardest_positives', tf.reduce_mean(hard_positives))
        tf.summary.scalar('hardest_negatives', tf.reduce_mean(hard_negatives))

        triplet_loss = tf.maximum(hard_positives - hard_negatives + margin, 0.0)

        return tf.reduce_mean(triplet_loss)


def triplet_loss_batch_hard_negative(embeddings, labels, margin=.1):
    """
    Triplet mining for batch hard negative.
        embeddings: (batch_size, feature_dims)
        labels: (batch_size,)
    """
    with tf.name_scope("triplet_loss"):
        pairwise_distances = _pairwise_distances(embeddings)

        labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        labels_not_equal = tf.logical_not(labels_equal)

        indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
        labels_equal = tf.logical_and(labels_equal, tf.logical_not(indices_equal))

        # pick hard positives
        # shape: (batch_size, batch_size)
        anchor_positive = tf.cast(labels_equal, dtype=tf.float32) * pairwise_distances

        # pick hard negatives
        mask_negative = tf.cast(labels_not_equal, dtype=tf.float32)
        max_negative = tf.reduce_max(pairwise_distances, axis=1)
        anchor_negative = pairwise_distances + max_negative * (1. - mask_negative)
        hard_negatives = tf.reduce_min(anchor_negative, axis=1, keepdims=True)

        tf.summary.scalar('anchor_positives', tf.reduce_mean(anchor_positive))
        tf.summary.scalar('hardest_negatives', tf.reduce_mean(hard_negatives))

        triplet_loss = tf.maximum(anchor_positive - hard_negatives + margin, 0.0)

        num_positive = tf.reduce_sum(tf.cast(labels_equal, dtype=tf.float32))
        triplet_loss = tf.reduce_sum(triplet_loss) / num_positive

        return triplet_loss


def _reduce_triplet_loss_sum(triplet_loss, labels_num):
    # if triplet is valid (has correct labels) it does not indicate that
    # distance will be greater than 0
    num_triplets = tf.reduce_sum(tf.cast(tf.greater(triplet_loss, .0), tf.float32))
    tf.summary.scalar('triplets_number', num_triplets)

    triplets_fraction = num_triplets / tf.cast((labels_num ** 3), tf.float32)
    tf.summary.scalar('triplets_fraction', triplets_fraction)

    triplet_sum = tf.reduce_sum(triplet_loss)
    tf.summary.scalar('triplet_reduced_sum', triplet_sum)

    loss = triplet_sum / (num_triplets + 1e-32)

    return loss


def _pairwise_distances(embeddings):
    "embeddings: batch_size x features_dimension"

    distances = \
          tf.reduce_sum(tf.square(embeddings), axis=1, keepdims=True)\
        + tf.reduce_sum(tf.transpose(tf.square(embeddings)), axis=0, keepdims=True)\
        - 2. * tf.matmul(embeddings, tf.transpose(embeddings))

    distances = tf.maximum(distances, .0)

    # add small value to avoid facing NaN gradients if
    # distances == 0.0 and square root is applied
    mask_non_positive = tf.cast(tf.equal(distances, 0.0), tf.float32)
    distances += mask_non_positive * 1e-32

    distances = tf.sqrt(distances)

    distances *= (1. - mask_non_positive)

    return distances


def _triplet_mask(labels):
    """
    From tensor of shape (batch_size)^3 representing
    (anchors, positive, negative) following triplets have to be excluded:
        - anchor == positive
        - labels[anchor] != labels[positive]
        - labels[anchor] == labels[negative]
    """
    with tf.name_scope('mask_valid_triplets'):
        labels = tf.squeeze(labels)
        labels_num = tf.shape(labels)[0]
        # get rid of triplets where anchor and
        # positive instance is THE SAME one
        indices_equal = tf.eye(labels_num, dtype=tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)

        anchor_positive_are_distinct = tf.expand_dims(indices_not_equal, 2)

        labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        labels_not_equal = tf.logical_not(labels_equal)

        labels_anchor_equals_positive = tf.expand_dims(labels_equal, 2)
        labels_anchor_not_equals_negative = tf.logical_not(tf.expand_dims(labels_equal, 1))

        valid_labels_triplets = tf.logical_and(
            labels_anchor_equals_positive,
            labels_anchor_not_equals_negative
        )

        valid_triplets = tf.logical_and(valid_labels_triplets,
                                        anchor_positive_are_distinct)

    return valid_triplets
