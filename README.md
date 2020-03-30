# G&T - embeddinG&Training data

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](code_of_conduct.md)

Acquiring raw image data to train ML algorithms is quite easy, but getting good
quality training labels for those images is really hard.

_G&T_ tries to bootstrap the process so that you minimise the number of images
you need to label. It uses deep image embeddings to find candidate images for
labelling in a big pool of unlabelled data, based on small seeds of labelled
examples.

We provide tools for all the steps of this process:
- the backend side of labelling using
[AWS Sagemaker Ground Truth](https://aws.amazon.com/sagemaker/groundtruth/)
- filtering images to find candidates for labelling
- aggregating final labels from multiple workers.

## Installing

The easiest way to install this package is via pip:

```
pip install git+https://github.com/popsa-hq/g-and-t.git#egg=gandt
```

## Backend for AWS Sagemaker Ground Truth

This package integrates neatly with AWS Sagemaker Ground Truth, a service for
crowdsourcing labelled data. 

There are a few parts to creating a labelling task:

- Create a lambda function for pre-processing labelling task input and for
post-processing answers - use source in [gandt/serving/]()
- Upload files to label to S3 and create a manifest file for use by Ground Truth
based on them can be done by [gandt/data/prepare_data.py]()
- Create interface for labelling - use examples in [gandt/templates/](). It will
look similar to this for the end user: 

![Example of labelling interface](docs/labelling-interface.png)

## Aggregating labels and filtering images using deep image embeddings

Once you have collected some labelled data, you can aggregate them to find out
which labels have high certainty. You can then filter out the images which
are dissimilar from your initial seed images.

An example of how to download the labelled data is under:
[notebooks/20200106 Analyse groundtruth results.ipynb]()

An example of filtering out irrelevant images using embeddings is here:
[notebooks/20200221 Example data filtering.ipynb]()

---
Created by: [Łukasz Kopeć](https://twitter.com/thektokolwiek).

Copyright (©) 2020 [Popsa](https://popsa.com).

Released under
[GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)

---
