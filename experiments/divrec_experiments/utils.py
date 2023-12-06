import json


def to_json(data, filepath):
    with open(filepath, mode="w") as file:
        json.dump(data, file)
