__all__ = ['load_instances', 'load_dictionary_from_file', '_process_regex_dict']

import json
import logging
import os
import pandas as pd
import regex
import yaml
from uriutils import uri_open

logger = logging.getLogger(__name__)


def load_instances(data_files, **kwargs):
    dict_list = list()
    for file in data_files:
        for line in file:
            tmp_dict = json.loads(line)
            del tmp_dict['unixReviewTime']
            del tmp_dict['reviewTime']
            dict_list.append(tmp_dict)

    return pd.DataFrame(dict_list)
#end def    


def load_dictionary_from_file(file_or_filename, *, force_format=None, title='dictionary'):
    if isinstance(file_or_filename, str): file_or_filename = uri_open(file_or_filename, 'rb')

    if force_format: ext = '.' + force_format
    else: _, ext = os.path.splitext(file_or_filename.name.lower())

    logger.info('Loaded {} {} from <{}>.'.format(ext[1:].upper(), title, file_or_filename.name))

    return yaml.load(file_or_filename)
#end def


#((:c)\s*|(:\))\s*)*[.!?]\s*|[.!?]\s*
def _process_regex_dict(regex_dict, regex_escape=False, **kwargs):
    regex_pattern_list = []
    for key, val in regex_dict.items():
        new_val = list()
        for item in val:
            if regex_escape:
                new_val.append(regex.escape(item))
            else:
                new_val.append(item)
        # regex_dict[key] = new_val
        regex_pattern_list += new_val
    #end for
    emoticon_regex_pattern = '|'.join(regex_pattern_list)


    # return '(' + '|'.join(regex_pattern_list) + r')*[.?!]\s*'
    if regex_escape:
        return '(((' + emoticon_regex_pattern + r')\s*)*([.!?]+\s+))|(((' + emoticon_regex_pattern + r')\s*)+([.!?]*\s+))'
    else:
        return set(regex_pattern_list)
#end def
