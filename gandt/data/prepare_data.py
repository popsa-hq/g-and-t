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

# Create manifest file (used by Ground Truth to know which images to display) -
# see README for more details.

import json
import os


def prepare_manifest(filename_list, positive_examples, negative_examples,
                     event_names, other_category_label='Other', group_size=20):
    """
    Prepare a manifest from list of filenames. Chunk into groups with roughly
    `group_size` members.

    Parameters
    ----------
    filename_list: Iterable
        List of S3 URIs with files to be labelled
    positive_examples: list(str)
        List of S3 URIs with positive examples of categories
    negative_examples: list(str)
        List of S3 URIs with negative examples of categories
    event_names: list(str)
        List of labels for different event types that a labeller will choose
        from
    other_category_label: str
        The title for the "other" category (usually 'Other' or
        'None of the above')
    group_size: int
        How many images to be labelled at once

    Returns
    -------
    manifest: list(dict)
        The manifest file containing images to label and the categories, ready
        to be written by `write_manifest_file`
    """
    # Number of tasks to create
    task_count = ((len(filename_list) // group_size) +
                  (1 if len(filename_list) % group_size > 0 else 0))

    manifest = [{'source': []} for _ in range(task_count)]

    for idx, record in enumerate(filename_list):
        group = manifest[idx % task_count]
        group['source'].append({
            'source-ref': record,
            'index': idx,
        })
    # because ground truth won't accept a JSON value for source or source-ref
    for record in manifest:
        additional_data = {
            'event_names': event_names,
            'positive_examples': positive_examples,
            'negative_examples': negative_examples,
            'other_category_label': other_category_label,
        }
        additional_data.update(record['source'][0])
        record['source'][0] = additional_data
        record['source'] = json.dumps(record['source'])

    return manifest


def write_manifest_file(manifest, fname, output_dir='../data/processed/'):
    """
    Write out the manifest file to a given filename

    Parameters
    ----------
    manifest: Iterable of dict
        The contents of the manifest
    fname: str
        Filename
    output_dir: str, default: '../data/processed/'
        The output directory for writing the manifest file

    Returns
    -------
    None
    """
    with open(os.path.join(output_dir, fname), 'w+') as f:
        f.writelines([json.dumps(x) + '\n' for x in manifest])
