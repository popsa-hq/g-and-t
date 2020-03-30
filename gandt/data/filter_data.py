"""
G & T - embeddinG&Training data, a package for crowdsourcing image labels

Copyright (C) 2020 Popsa.
Author: Łukasz Kopeć

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import logging

import numpy as np
import pandas as pd
import scipy.stats
import tensorflow as tf
from scipy.spatial.distance import cdist


def calculate_embeddings_on_images(
        model, images, x_col='full_path', batch_size=32, img_height=224,
        img_width=224):
    """
    Calculate embeddings on list of image filenames, using model.

    Parameters
    ----------
    model: tf.Model
        The model used for calculating embeddings
    images: pd.DataFrame
        DataFrame with image filenames
    x_col: str, default: 'full_path'
        The name of the column with filename
    batch_size: int, default: 32
        Batch size for predictions
    img_height: int, default: 224
        Image height for model input
    img_width: int, default: 224
        Image width for model input

    Returns
    -------
    predictions: np.array
        Numpy array of predictions
    """
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    BATCH_SIZE = batch_size
    IMG_HEIGHT = img_height
    IMG_WIDTH = img_width

    data_gen = datagen.flow_from_dataframe(
        images,
        x_col=x_col,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='rgb',
        class_mode=None,
        batch_size=BATCH_SIZE,
        shuffle=False,
        interpolation='nearest',
        validate_filenames=True)

    predictions = model.predict(data_gen, verbose=True, workers=12)

    return predictions


def filter_by_similarity(model, labelled_images, unlabelled_images,
                         x_col='full_path', y_col='majority_label',
                         other_label='Other', min_proportion_to_discard=0.1,
                         batch_size=32, img_height=224, img_width=224):
    """
    The function will try to discard at least `min_proportion_to_discard` of
    `unlabelled_images`, based on their lack of similarity to any of the
    labelled categories.

    In order to discard the dissimilar images, it will first calculate
    embeddings (i.e. feature extraction) for all labelled and unlabelled images
    using `model`.

    Then, this function calculates the distance between all unlabelled images
    and all labelled images, and for each unlabelled image counts how many
    labelled classes are, on average, farther away than the 'other' class.

    Based on that info, it chooses a threshold of 'number of classes farther
    away than other', such that it discards at least `min_proportion_to_discard`
    of unlabelled images.

    You should probably still add a proportion of the discarded images to your
    dataset, as I'm sure there will potentially be some images there which will
    be worth labelling.

    Parameters
    ----------
    model: tf.Model
        The model used for calculating embeddings
    labelled_images: pd.DataFrame
        DataFrame with images labelled (e.g. using MTurk)
    unlabelled_images: pd.DataFrame
        DataFrame with images to be filtered so that we can minimise labelling
        needs
    x_col: str, default: 'full_path'
        The name of the column with filename in `labelled_images` and
        `unlabelled_images`
    y_col: str, default: 'majority_label'
        The name of the column with label in `labelled_images`
    other_label: str, default: 'Other'
        The name of label denoting it's none of the specific categories
    min_proportion_to_discard: float, default 0.1
        The minimum proportion of images to discard. See above for explanation
    batch_size: int, default: 32
        Batch size for predictions
    img_height: int, default: 224
        Image height for model input
    img_width: int, default: 224
        Image width for model input

    Returns
    -------
    discarded_filenames: list(str)
        List of filenames to discard
    remaining_filenames: list(str)
        List of filenames to keep
    avg_dist_to_cats: pd.DataFrame
        Dataframe with filenames and their average distances to all examples

    Examples
    --------
    see `notebooks/20200221 Example data filtering.ipynb`

    """
    unique_labels = np.unique(
        np.concatenate(labelled_images[y_col].tolist()))

    labels_to_images = {}
    for label in unique_labels:
        # Select images which have at least this label in their list of labels
        labels_to_images[label] = labelled_images[
            labelled_images[y_col].apply(lambda x: label in x)][x_col].tolist()

    logging.debug("Extracting features from labelled images")
    predictions_labelled = calculate_embeddings_on_images(
        model, labelled_images, x_col=x_col, batch_size=batch_size,
        img_height=img_height, img_width=img_width)

    logging.debug("Extracting features from unlabelled images")
    predictions_unlabelled = calculate_embeddings_on_images(
        model, unlabelled_images, x_col=x_col, batch_size=batch_size,
        img_height=img_height, img_width=img_width)

    logging.debug("Calculating distances between images")
    distances = cdist(predictions_unlabelled, predictions_labelled,
                      metric='cosine')

    # index by image filename so that we can reconcile with labels_to_images
    df_distances = pd.DataFrame(
        distances, index=unlabelled_images['full_path'],
        columns=labelled_images['full_path'])

    # Calculate average distances to categories
    logging.debug("Calculating average distances to categories")
    avg_dist_to_cats = {}
    for label, filenames in labels_to_images.items():
        avg_dist_to_cats[label] = scipy.stats.trim_mean(
            df_distances.loc[:, filenames], 0.1, axis=1)

    avg_dist_to_cats = pd.DataFrame(
        np.vstack(avg_dist_to_cats.values()).T,
        index=unlabelled_images['full_path'],
        columns=avg_dist_to_cats.keys(),
    )

    unique_labels_not_other = [label for label in unique_labels
                               if label != other_label]

    # Count how many categories are farther away than 'other'
    avg_dist_to_cats['no_cats_farther_than_other'] = pd.DataFrame(
        [avg_dist_to_cats[label] > avg_dist_to_cats[other_label]
         for label in unique_labels_not_other]).sum()

    # This is the theoretical maximum
    no_unique_categories_not_other = len(unique_labels_not_other)

    minimum_images_to_discard = (min_proportion_to_discard *
                                 len(unlabelled_images))

    proportions = pd.DataFrame(
        [(x, len(avg_dist_to_cats[
                     avg_dist_to_cats['no_cats_farther_than_other'] > x]))
         for x in range(no_unique_categories_not_other + 1)],
        columns=['threshold', 'no_remaining_images'])

    max_threshold = proportions[
        proportions['no_remaining_images'] >= minimum_images_to_discard][
        'threshold'].max()

    discarded_filenames = avg_dist_to_cats[
        avg_dist_to_cats['no_cats_farther_than_other'] > max_threshold
        ].index.tolist()

    remaining_filenames = avg_dist_to_cats[
        avg_dist_to_cats['no_cats_farther_than_other'] <= max_threshold
        ].index.tolist()

    return discarded_filenames, remaining_filenames, avg_dist_to_cats
