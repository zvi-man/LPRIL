import os
from typing import Any, Dict
import json


class JSONUtils:
    @staticmethod
    def append_to_json(json_file_path: str, data: Any) -> None:
        with open(json_file_path, 'a') as f:
            f.write(json.dumps(data) + '\n')

    @staticmethod
    def read_appended_json_to_dict(json_file_path: str) -> Dict:
        output_dict = {}
        with open(json_file_path, 'r') as f:
            for line in f:
                output_dict.update(json.loads(line))
        return output_dict


if __name__ == '__main__':
    a = {'a': {"lp_pred": "123", "lp_label": "123", "image_lp": "123"}}
    b = {'b': {"lp_pred": "123", "lp_label": "123", "image_lp": "123"}}
    c = {'c': {"lp_pred": "123", "lp_label": "123", "image_lp": "123"}}
    json_path = 'a.json'
    for line_dict in [a, b, c]:
        JSONUtils.append_to_json(json_path, line_dict)
    print(JSONUtils.read_appended_json_to_dict(json_path))
    os.remove(json_path)
