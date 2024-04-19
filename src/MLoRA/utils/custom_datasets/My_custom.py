import json
from datasets import load_dataset, Dataset
My_custom_data = {
"bigfive_task_id":"/cpfs01/user/chenqin.p/dyh/MOELoRA-peft/data/bigfive_task_id.jsonl",
"bigfive_task_id_questionnaire":"/cpfs01/user/chenqin.p/dyh/MOELoRA-peft/data/task_data_questionnaire.jsonl",
"bigfive_task_test":"/cpfs01/user/chenqin.p/dyh/MOELoRA-peft/data/test.jsonl",
"dimensional_task_q":"/cpfs01/user/chenqin.p/dyh/MOELoRA-peft/data/dimensional_task_id.jsonl",
"dimensional_task_id_4_13":"/cpfs01/user/chenqin.p/dyh/MOELoRA-peft/data/dimensional_task_id_4_13.jsonl"
}

class MyCustom(Dataset):
    def __init__(self, mode: str, cache_dir: str = None) -> None:
        self.mode = "sft"
        self.rows = []
        # self.system_prompt = []
        self.task_id = []
        import os
        cache_dir = os.path.join(os.getcwd(),cache_dir)

        with open(cache_dir,"r",encoding="utf-8") as f:
            self.rows = f.readlines()

        for i in range(len(self.rows)):
            data = json.loads(self.rows[i])
            self.rows[i] = data
            # self.system_prompt.append(data["system_prompt"])


    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        dialogue = self.rows[index]["data"]
        task_id = self.rows[index]["task_id"]
        # system_prompt = self.system_prompt[index]
        if self.mode == "sft":
            return (dialogue, task_id, "<|CustomData|>")
        elif self.mode == "rl":
            return tuple(dialogue[:-1])
    
