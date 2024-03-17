import random
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from .formatting import QA_SPECIAL_TOKENS, format_pairs, format_system_prefix
from torch.nn import functional as F
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase, TruncationStrategy
import transformers

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import torch
import transformers

from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = -100

@dataclass
class DialogueDataCollator:
    """
    Expects a list of texts corresponding to a sequence of [question, answer, question, answer, ...] pairs.
    """

    tokenizer: PreTrainedTokenizerBase
    # padding: Union[bool, str, PaddingStrategy] = True
    max_len: Optional[int] = None
    system_prefix = ''''''

    def preprocess(
        self,
        sources,
        system_messages
    ) -> Dict:
        roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
        tokenizer = self.tokenizer
        max_len = self.max_len
        im_start = tokenizer.im_start_id
        im_end = tokenizer.im_end_id
        nl_tokens = tokenizer('\n').input_ids
        _system = tokenizer('system').input_ids + nl_tokens
        _user = tokenizer('user').input_ids + nl_tokens
        _assistant = tokenizer('assistant').input_ids + nl_tokens

        # Apply prompt templates
        input_ids, targets = [], []
        for i, source in enumerate(sources):
            system_message = system_messages[i]
            try:
                if roles[source[0]["from"]] != roles["user"]:
                    source = source[1:]
            except:
                print(sources)
                continue

            input_id, target = [], []
            system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
            input_id += system
            target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens


            assert len(input_id) == len(target)
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                _input_id = tokenizer(role).input_ids + nl_tokens + \
                    tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
                input_id += _input_id
                if role == '<|im_start|>user':
                    _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
                elif role == '<|im_start|>assistant':
                    _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                        _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
                else:
                    raise NotImplementedError
                target += _target
            assert len(input_id) == len(target)
            # print(tokenizer.convert_ids_to_tokens(input_id))
            input_ids.append(input_id[:max_len])
            targets.append(target[:max_len])
        
        # Pad to batch max_len
        batch_pad_max_len = max_len
        batch_max_length = max(len(i) for i in input_ids)
        batch_pad_max_len = min(batch_pad_max_len, batch_max_length)
        # print(f"batch_pad_max_len: {batch_pad_max_len}")

        input_ids = [i + [tokenizer.pad_token_id] * (batch_pad_max_len - len(i)) for i in input_ids]
        targets = [t + [IGNORE_TOKEN_ID] * (batch_pad_max_len - len(t)) for t in targets]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)

        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )
    
    def preprocess_llama(
        self,
        sources,
        task_ids
    ) -> Dict:
        roles = {"user": "<s>user", "assistant": "<s>assistant"}
        tokenizer = self.tokenizer
        max_len = self.max_len
        im_start = 1
        im_end = 2
        nl_tokens = [29871, 13]
        _system = tokenizer('system').input_ids[1:] 
        _user = tokenizer('user').input_ids[1:]
        _assistant = tokenizer('assistant').input_ids[1:] 


        # Apply prompt templates
        input_ids, targets = [], []
        for i, source in enumerate(sources):
            # system_message = system_messages[i]
            try:
                if roles[source[0]["from"]] != roles["user"]:
                    source = source[1:]
            except:
                print(sources)
                continue

            input_id, target = [], []

            # add system prompt

            # system = [im_start] + _system + tokenizer(system_message).input_ids[1:] + [im_end] + nl_tokens
            # input_id += system
            # target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
            # <s>user xxx</s> <s>assistant XXX</s>

            assert len(input_id) == len(target)
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                _input_id = tokenizer(role).input_ids[1:] + \
                    tokenizer(sentence["value"]).input_ids[1:] + [im_end]
                input_id += _input_id
                if role == '<s>user':
                    _target = [im_start] + _user + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] 
                elif role == '<s>assistant':
                    _target = [im_start] + _assistant + _input_id[2:-1] + [im_end] 
                else:
                    raise NotImplementedError
                target += _target
                # print("input", _input_id)
                # print("target", _target)
            # print("len(target)", len(input_id))
            # print("len(target)", len(target))
            # print("input_id", input_id)
            # print("target", target)
            assert len(input_id) == len(target)
            input_ids.append(input_id[:max_len])
            targets.append(target[:max_len])
        
        # Pad to batch max_len
        batch_pad_max_len = max_len
        batch_max_length = max(len(i) for i in input_ids)
        batch_pad_max_len = min(batch_pad_max_len, batch_max_length)
        # print(f"batch_pad_max_len: {batch_pad_max_len}")

        input_ids = [i + [0] * (batch_pad_max_len - len(i)) for i in input_ids]
        targets = [t + [IGNORE_TOKEN_ID] * (batch_pad_max_len - len(t)) for t in targets]

        # print(tokenizer.convert_ids_to_tokens(input_id))
        # print(input_id)
        # print(target)
        # print(input_ids)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)
        task_ids = torch.tensor(task_ids, dtype=torch.long)

        # batch_data = dict(
        #     input_ids=input_ids,
        #     labels=targets,
        #     attention_mask=input_ids.ne(0),
        #     task_id=task_ids
        # )

        # print("batch_data", batch_data)

        return dict(
            input_ids=input_ids,
            # labels=targets,
            # attention_mask=input_ids.ne(0),
            task_id=task_ids
        )


    def __call__(self, features):
        # features 是一个长度为batch size 的list
        # print("features", features)

        batch_dialogue = []
        task_ids = []

        for messages in features:
            if isinstance(messages,tuple) and len(messages)==3 and messages[-1]=="<|CustomData|>":
                dialogue = messages[0]
                dialogue = [{"from": "user", "value": dialogue[i]} if i%2==0 else {"from": "assistant", "value": dialogue[i]} for i, t in enumerate(dialogue)]
                batch_dialogue.append(dialogue)
                task_ids.append(int(messages[1]))
            else:
                dialogue = messages
                dialogue = [{"from": "user", "value": dialogue[i]} if i%2==0 else {"from": "assistant", "value": dialogue[i]} for i, t in enumerate(dialogue)]
                batch_dialogue.append(dialogue)

        
        batch_data = self.preprocess_llama(sources=batch_dialogue, task_ids=task_ids)

        return batch_data
