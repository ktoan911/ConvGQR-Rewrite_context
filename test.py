import json
import re


def to_dict_query(long_string):
    match = re.search(r'\{"query":\s*".*?"\}', long_string)
    if match:
        dict_string = match.group(0)
        try:
            result_dict = json.loads(dict_string)
            return result_dict
        except Exception:
            return None