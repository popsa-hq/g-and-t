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

import json
import logging
import os
from collections import Counter, defaultdict

import pandas as pd


def sum_list(lst):
    return [item for sublist in lst for item in sublist]


def read_responses(fname):
    """
    Read in one file with all responses for a single object id.
    They can be found under annotations/worker-response/iteration-*/*/*.json

    Parameters
    ----------
    fname: str
        Filename of the response

    Returns
    -------
    raw_results: pd.DataFrame
        The raw responses data
    """
    # Get the labelling job name from the filename
    labelling_job_name = os.path.abspath(fname).split(os.path.sep)[-6]

    with open(fname) as f:
        data = json.load(f)

    raw_results = []

    # There can be more than one annotation set in one file, log how many
    logging.info('Fname: {}, no annotation sets: {}'.format(fname, len(data)))

    # Iterate through annotations
    for annotation_set in data:
        # get the input data for this labelling job
        labelling_job_input_data = json.loads(
            annotation_set['dataObject']['content'])

        # remap to index -> filename
        index_to_filename = {
            d['index']: os.path.basename(d['source-ref'])
            for d in labelling_job_input_data
        }

        dataset_object_id = annotation_set['datasetObjectId']

        # Go through individual annotations
        for labelling in annotation_set['annotations']:
            worker_id = labelling['workerId']

            # Go through all images annotated bu this worker
            this_worker_annotations = json.loads(
                labelling['annotationData']['content'])
            for label_id, label_raw in this_worker_annotations.items():
                this_labels = json.loads(label_raw)
                label_index = int(label_id.split('-')[-1])
                image_filename = index_to_filename[label_index]
                raw_results.append(
                    {'labelling_job_name': labelling_job_name,
                     'labels_fname': fname,
                     'image_filename': image_filename,
                     'image_index': label_index,
                     'worker_id': worker_id,
                     'labels': this_labels,
                     'dataset_object_id': dataset_object_id})

    return pd.DataFrame(raw_results)


def get_summary_stats_for_image(raw_labels, min_ratings_no=2,
                                min_ratings_prop=0.5, decoys=None):
    """
    Calculate summary stats on one image worth of labels:
    'worker_list': list of all IDs of workers who gave answers for this image,
    'labels': list of all labels given to this image,
    'labels_all_same': whether all labels are the same,
    'majority_label': the most common label,
    'no_most_common_labels': how many most common label responses there were,
    'no_eligible_workers': how many responses in total there were,
    'is_label_certain': boolean whether the label is certain based on criteria
        set in parameters,
    'majority_label_description': string of most common label + '(certain)' /
        '(need relabelling)' - useful for visualising results,
    'is_decoy': whether this image is decoy (see parameters),
    'decoy_label': the label of decoy image/ np.nan if the image is not a decoy,
    'worker_quality': mapping of {worker_id: 0 if their label agrees with
        majority, 1 otherwise} for workers who labelled this image.

    Parameters
    ----------
    raw_labels: pd.DataFrame
        Raw labelled data for one image
    min_ratings_no: int, default 2
        Minimum (>=) number of agreeing labels to consider a label certain
    min_ratings_prop: float, default 0.5
        Minimum (>) proportion of agreeing labels to consider a label certain
        Both of the above need to happen
    decoys: dict(str, str) or None
        The mapping of {filename: category} of known decoys - images with known
        labels that are added to the dataset so that we can assess reliability
        even better.

    Returns
    -------
    output: pd.Series
        Aggregated response output with summary stats (see above for list)

    """
    if decoys is None:
        decoys = {}

    output = {}

    # Get the list of all workers
    output['worker_list'] = list(raw_labels['worker_id'])

    # List with all labels together
    output['labels'] = raw_labels['labels'].sum()

    # Count label frequency
    label_counts = Counter(output['labels'])

    # Check if all labels are the same - the counter will only have one entry
    output['labels_all_same'] = len(label_counts) == 1

    # Get the most common label and count
    most_common_label, most_common_count = label_counts.most_common(1)[0]

    # See if any other labels have the same count too, in case there is a tie
    most_common_labels = sorted([k for k, v in label_counts.items()
                                 if v == most_common_count])

    # Get number of all labels
    no_labels = len(output['worker_list'])

    output['no_most_common_labels'] = most_common_count
    output['no_eligible_workers'] = no_labels
    all_most_common_label_str = ', '.join(most_common_labels)
    if most_common_count >= min_ratings_no and most_common_count > (
            min_ratings_prop * no_labels):
        most_common_label_str = '{} (certain)'.format(all_most_common_label_str)
        is_certain = True
    else:
        most_common_label_str = '{} (need relabelling)'.format(
            all_most_common_label_str)
        is_certain = False
    # Something like 'Other (2/3)':
    verbose_label_str = ', '.join(
        '{} ({} / {})'.format(
            this_most_common_label, most_common_count, no_labels)
        for this_most_common_label in most_common_labels)

    output['majority_label'] = most_common_labels
    output['majority_label_description'] = most_common_label_str
    output['majority_label_verbose'] = verbose_label_str

    output['is_label_certain'] = is_certain

    # Check if this file is a decoy
    this_filename = raw_labels['image_filename'].iloc(0)
    try:
        output['decoy_label'] = decoys[this_filename]
        output['is_decoy'] = True
    except KeyError:
        output['decoy_label'] = pd.np.nan
        output['is_decoy'] = False

    worker_quality_mapping = {}
    for i, row in raw_labels.iterrows():
        if is_certain:
            worker = row['worker_id']
            labels = row['labels']
            # 1/ 0 whether this is a majority label and is certain
            is_majority_label = int(
                any(label in most_common_labels for label in labels))
            worker_quality_mapping[worker] = is_majority_label

    output['worker_quality'] = worker_quality_mapping

    return pd.Series(output)


def get_workers_reliability(experiment_df, previous_experiments=(),
                            reliability_threshold=0.8,
                            no_labels_per_experiment=20):
    """
    Get a dataframe with workers' reliability, defined as proportion of when
    they agree with majority labels. Can optionally also use previous
    experiments' data

    Parameters
    ----------
    experiment_df: pd.DataFrame
        dataframe with individual annotations for current experiments
    previous_experiments: list(pd.DataFrame), optional
        list of dataframes with individual annotations for previous experiments
    reliability_threshold: float, optional
        What minimum proportion of agreement to require to call a worker
        reliable (strictly greater than).
    no_labels_per_experiment: int, default: 20
        Number of label ratings in a single experiment. Used to calculate the
        number of experiments in which a worker took part.

    Returns
    -------
    df_workers: pd.DataFrame
        Data with workers' reliability
    """
    all_experiments = pd.concat([experiment_df] + list(previous_experiments))

    # Create a dict with list of worker reliability ratings (0/1) per worker_id
    worker_reliability = defaultdict(list)
    for quality_rating in all_experiments['worker_quality']:
        for worker_id, reliability_rating in quality_rating.items():
            worker_reliability[worker_id].append(reliability_rating)

    worker_data = []
    for worker_id, agreement_scores in worker_reliability.items():
        worker_data.append({
            'worker_id': worker_id,
            'mean': pd.np.mean(agreement_scores),
            'count': len(agreement_scores) // no_labels_per_experiment,
            'is_worker_reliable': pd.np.mean(
                agreement_scores) > reliability_threshold})

    df_workers = pd.DataFrame(worker_data)
    return df_workers


def filter_out_unreliable_workers(raw_responses, previous_experiments=(),
                                  reliability_threshold=0.8, min_ratings_no=2,
                                  min_ratings_prop=0.5, decoys=None,
                                  include_previous_in_output=False):
    """
    Aggregates labels per image filename, filtering out responses by unreliable
    workers. Photos with labels which are uncertain can then be re-labelled.

    Parameters
    ----------
    raw_responses: pd.DataFrame
        dataframe with individual annotations for current experiments
    previous_experiments: list(pd.DataFrame), optional
        list of dataframes with individual annotations for previous experiments
    reliability_threshold: float, optional
        What minimum proportion of agreement to require to call a worker
        reliable (strictly greater than).
    min_ratings_no: int, default 2
        Minimum (>=) number of agreeing labels to consider a label certain
    min_ratings_prop: float, default 0.5
        Minimum (>) proportion of agreeing labels to consider a label certain
        Both of the above need to happen
    decoys: dict(str, str) or None
        The mapping of {filename: category} of known decoys - images with known
        labels that are added to the dataset so that we can assess reliability
        even better.
    include_previous_in_output: bool
        If True, use the previous experiments for both assessing reliability and
        labelling (e.g. if you run second iteration of a labelling job to
        relabel those unreliably labelled). If False, only use previous
        experiments for assessing worker reliability (e.g. if the previous
        experiments are unrelated to this one).

    Returns
    -------
    unfiltered_annotated_responses: pd.DataFrame
        Dataframe with image labels, aggregated by image filename. You should
        only use images where 'is_label_certain' is True (see
        `get_summary_stats_for_image` above for explanation of what this means)
    filtered_annotated_responses: pd.DataFrame
        As above, except only based of responses from reliable workers
    """
    if decoys is None:
        decoys = {}

    unfiltered_annotated_responses = raw_responses.groupby(
        ['image_filename', 'labelling_job_name'],
        as_index=False
    ).apply(
        get_summary_stats_for_image, min_ratings_no=min_ratings_no,
        min_ratings_prop=min_ratings_prop, decoys=decoys
    ).reset_index()

    previous_experiments_annotated = [responses.groupby(
        ['image_filename', 'labelling_job_name'],
        as_index=False
    ).apply(
        get_summary_stats_for_image, min_ratings_no=min_ratings_no,
        min_ratings_prop=min_ratings_prop, decoys=decoys
    ).reset_index() for responses in previous_experiments]

    workers_reliability = get_workers_reliability(
        unfiltered_annotated_responses,
        previous_experiments=previous_experiments_annotated,
        reliability_threshold=reliability_threshold)

    if include_previous_in_output:
        # Concatenate all previous experiments together
        previous_experiments = pd.concat(previous_experiments)
        # Overwrite the job name, as we're now considering them all same
        # labelling job
        previous_experiments['labelling_job_name'] = raw_responses[
            'labelling_job_name'].tolist()[0]

        # Add all raw responses together
        raw_responses = pd.concat([raw_responses, previous_experiments])

        # Add all the annotated responses together
        unfiltered_annotated_responses = pd.concat(
            [unfiltered_annotated_responses] + previous_experiments_annotated)

    filtered_raw_responses = raw_responses.merge(workers_reliability)
    filtered_raw_responses = filtered_raw_responses[
        filtered_raw_responses['is_worker_reliable']]

    filtered_annotated_responses = filtered_raw_responses.groupby(
        ['image_filename', 'labelling_job_name'],
        as_index=False
    ).apply(
        get_summary_stats_for_image, min_ratings_no=min_ratings_no,
        min_ratings_prop=min_ratings_prop, decoys=decoys
    ).reset_index()

    return unfiltered_annotated_responses, filtered_annotated_responses
