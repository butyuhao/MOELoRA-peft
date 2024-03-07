import json
from datasets import load_dataset, Dataset
My_custom_data = {
"_brainstorming_en":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/HighQuality/Brainstorming_en.jsonl",
"_brainstorming_zh":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/HighQuality/Brainstorming_zh.jsonl",
"_code_en":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/HighQuality/Code_en.jsonl",
"_code_zh":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/HighQuality/Code_zh.jsonl",
"_complex_instruction_en":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/HighQuality/Complex-Instruction_en.jsonl",
"_complex_instruction_zh":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/HighQuality/Complex-Instruction_zh.jsonl",
"_continue_en":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/HighQuality/Continue_en.jsonl",
"_continue_zh":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/HighQuality/Continue_zh.jsonl",
"_harmless_en":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/HighQuality/Harmless_en.jsonl",
"_harmless_zh":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/HighQuality/Harmless_zh.jsonl",
"_mix_gpt_4":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/HighQuality/MIX_GPT-4.jsonl",
"_role_playing_en":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/HighQuality/Role-Playing_en.jsonl",
"_role_playing_zh":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/HighQuality/Role-Playing_zh.jsonl",
"_switching_en":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/HighQuality/Switching_en.jsonl",
"_switching_zh":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/HighQuality/Switching_zh.jsonl",
"_writing_en":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/HighQuality/Writing_en.jsonl",
"_writing_zh":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/HighQuality/Writing_zh.jsonl",
"_lima_chat":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/HighQuality/lima_chat.jsonl",
"_lima_qa":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/HighQuality/lima_qa.jsonl",
"_ruozhiba":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/HighQuality/ruozhiba.jsonl",
"_sharegpt_format":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/HighQuality/shareGPT_format.jsonl",
"_empdia_ly_07_17":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/feature/EmpDia_ly_07_17.jsonl",
"_composition_hq":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/feature/composition_hq.jsonl",
"_composition_inst":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/feature/composition_inst.jsonl",
"_compositions":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/feature/compositions.jsonl",
"_open_domain_subject":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/feature/open_domain&subject.jsonl",
"_psy_diagnose":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/feature/psy_diagnose.jsonl",
"_psy_diagnose_wo_inner":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/feature/psy_diagnose_wo_inner.jsonl",
"_psy_generated_zyg_6_29":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/feature/psy_generated_zyg_6_29.jsonl",
"_psy_gpt4_merge_format":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/feature/psy_gpt4_merge_format.jsonl",
"_psy_gpt4_wo_merge_format":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/feature/psy_gpt4_wo_merge_format.jsonl",
"_search":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/feature/search.jsonl",
"_similar_question":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/feature/similar_question.jsonl",
"_socrates_psy":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/feature/socrates_psy.jsonl",
"_socrates_psy_wo_inner":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/feature/socrates_psy_wo_inner.jsonl",
"_socrates_teaching":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/feature/socrates_teaching.jsonl",
"_zuowen":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/feature/zuowen.jsonl",
"_ecnu_qa":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/honest/ecnu_qa.jsonl",
"_honest":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/honest/honest.jsonl",
"_mix_belle":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/opensource_data/MIX_BELLE.jsonl",
"_mix_en":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/opensource_data/MIX_EN.jsonl",
"_mix_zh_others":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/opensource_data/MIX_ZH-Others.jsonl",
"_transstyle":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/zeroshot_data/TransStyle.jsonl",
"_composition_review":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/zeroshot_data/composition_review.jsonl",
"_correct":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/zeroshot_data/correct.jsonl",
"_emotion_dialog":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/zeroshot_data/emotion_dialog.jsonl",
"_gushi_chengyu_pre":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/zeroshot_data/gushi_chengyu_pre.jsonl",
"_poem_generate":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/zeroshot_data/poem_generate.jsonl",
"_poem_transform":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/zeroshot_data/poem_transform.jsonl",
"_reading_comprehension_en":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/zeroshot_data/reading_comprehension_en.jsonl",
"_rewriting":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/zeroshot_data/rewriting.jsonl",
"_story_generate":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/zeroshot_data/story_generate.jsonl",
"_summary":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/zeroshot_data/summary.jsonl",
"_writing":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/zeroshot_data/writing.jsonl",
"_writing_cot":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/zeroshot_data/writing_cot.jsonl",
"oasst_export_":"/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/oasst_export.jsonl",
"_psychat_6_29_and_11_15_mix": "/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/feature/psychat_6_29_and_11_15_mix.jsonl",
"_socrates_teaching_math1115": "/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/feature/socrates_teaching_math_1115.jsonl", 
"_socrates_teaching_math1208": "/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/feature/socrates_teaching_math_1207.jsonl", 
"_socrates_teaching_math1213": "/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/feature/socrates_teaching_math_1213.jsonl", 
"_psy_qa_1208": "/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/feature/psy_qa_1208.jsonl",
"_socrates_teaching_math224": "/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/feature/socrates_teaching_math_0220.jsonl",
"_socrates_teaching_math224sol": "/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/feature/socrates_teaching_math_sol_0220.jsonl",
"composition_data_primary_middle": "/cpfs01/user/chenqin.p/dyh/Open-Assistant/data/custom_data/feature/composition_data_primary_middle.jsonl",
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
    
