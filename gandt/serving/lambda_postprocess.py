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

# Special thanks to Dave Schultz (@daveschultz10):
# From: https://medium.com/@daveschu/increasing-labeling-efficiency-with-a-tiled
# -image-annotation-layout-4f7891ba3354
# This simply assembles all the responses in a JSON file.
import json
import boto3
import logging
from urllib.parse import urlparse


def lambda_handler(event, context):
    """
    Assembles all responses in a JSON.

    Parameters
    ----------
    event: AWS Lambda event
        Default Lambda input
    context: AWS Lambda context
        Default Lambda input

    Returns
    -------
    consolidated_response: dict
    Dictionary with the responses to all images
    """
    logging.debug(json.dumps(event))
    payload = get_payload(event)
    logging.debug(json.dumps(payload))
    consolidated_response = []
    for dataset in payload:
        annotations = dataset['annotations']
        responses = []
        for annotation in annotations:
            response = json.loads(annotation['annotationData']['content'])
            if 'annotatedResult' in response:
                response = response['annotatedResult']
            responses.append({
                'workerId': annotation['workerId'],
                'annotation': response
            })
        consolidated_response.append({
            'datasetObjectId': dataset['datasetObjectId'],
            'consolidatedAnnotation': {
                'content': {
                    event['labelAttributeName']: {
                        'responses': responses
                    }
                }
            }
        })
    logging.debug(json.dumps(consolidated_response))
    return consolidated_response


def get_payload(event):
    if 'payload' in event:
        parsed_url = urlparse(event['payload']['s3Uri'])
        s3 = boto3.client('s3')
        text_file = s3.get_object(Bucket=parsed_url.netloc,
                                  Key=parsed_url.path[1:])
        return json.loads(text_file['Body'].read())
    else:
        return event.get('test_payload', [])
