import json
from datasets import load_dataset, Dataset
My_custom_data = {
"bigfive_task_id":""
}
class MyCustom(Dataset):
    def __init__(self, mode: str, cache_dir: str = None) -> None:
        self.mode = mode
        self.rows = []
        self.system_prompt = []
        import os
        cache_dir = os.path.join(os.getcwd(),cache_dir)
        print(cache_dir)
        with open(cache_dir,"r",encoding="utf-8") as f:
            self.rows = f.readlines()
        
        for i in range(len(self.rows)):
            data = json.loads(self.rows[i])
            self.rows[i] = data["data"]
            self.system_prompt.append(data["system_prompt"])
              
            # print(self.rows[i])


    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        dialogue = self.rows[index]
        system_prompt = self.system_prompt[index]
        if self.mode == "sft":
            return (dialogue,system_prompt,"<|CustomData|>")
        elif self.mode == "rl":
            return tuple(dialogue[:-1])
    
