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
# from: https://medium.com/@daveschu/increasing-labeling-efficiency-with-a-tiled
# -image-annotation-layout-4f7891ba3354

import logging
import json


def lambda_handler(event, context):
    """
    Load the JSON data in manifest and turn it into an array of objects to label

    Parameters
    ----------
    event: AWS Lambda event
        Default Lambda input
    context: AWS Lambda context
        Default Lambda input

    Returns
    -------
    response: dict
        Dictionary with input data in correct format for labelling
    """
    
    logging.debug(event)
    source = event['dataObject'].get('source')
    if source is None:
        logging.debug("Missing source data object")
        return {}
    # Ground truth currently only allows string values for source or source-ref
    # attributes
    # This allows the source to be passed as a string and loaded into an object
    if type(source) is str:
        source = json.loads(source)
    response = {
        "taskInput": {
            "taskObject": source
        }
    }
    logging.debug(response)
    return response
