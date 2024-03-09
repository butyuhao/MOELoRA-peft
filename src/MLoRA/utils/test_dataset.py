import os
import yaml
from utils_oa import get_dataset


def read_yamls(dir):
    conf = {}
    no_conf = True

    def find_jsonl_files(directory):
        jsonl_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".yaml"):
                    jsonl_files.append(os.path.join(root, file))
        return jsonl_files

    jsonl_files = find_jsonl_files(dir)
    print(jsonl_files)

    for config_file in jsonl_files:
        no_conf = False
        with open(config_file, "r") as f:
            conf.update(yaml.safe_load(f))
    
    return conf

data_conf = read_yamls("/cpfs01/user/chenqin.p/dyh/MOELoRA-peft/src/configs")

data_conf = data_conf["bigfive_task"]
print(data_conf)

train_dataset, eval_dataset = get_dataset(data_conf)
print("train_dataset", train_dataset[0])