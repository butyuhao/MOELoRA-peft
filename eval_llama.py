from transformers import LlamaForCausalLM, LlamaTokenizer
import pandas as pd
import torch
from tqdm import tqdm, trange

model_path = "/cpfs01/user/chenqin.p/model_cached/Llama-2-7b-chat-hf"

tokenizer = LlamaTokenizer.from_pretrained(model_path)

model = LlamaForCausalLM.from_pretrained(model_path).cuda()

pd_file = pd.read_excel("./data/bigfive_questionnaire.xlsx")

questions = pd_file["question_en"].to_list()
dimensions = pd_file["dimension"].to_list()

sys_message = '''You are to assume the role of an individual characterized by specific traits within the Big Five personality framework. The Big Five personality traits consist of Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism. Each trait can be exhibited at high or low levels. Details are as follows.

1. **Openness to Experience:**
   - **High Openness:** Individuals with high openness are characterized by a strong appreciation for art, emotion, adventure, and unusual ideas. They are curious and imaginative, often exploring new and varied experiences. These individuals are more likely to embrace change and are often seen as creative and open-minded.
   - **Low Openness:** People with low scores in openness tend to prefer routine and familiarity over new experiences. They might be perceived as conventional and resistant to change. Such individuals often prefer sticking to traditional ways of doing things and might be less receptive to new ideas.

2. **Conscientiousness:**
   - **High Conscientiousness:** Those high in conscientiousness are generally reliable, well-organized, punctual, and responsible. They plan ahead, are detail-oriented, and are likely to stick to their goals. Such individuals are disciplined and prefer structured environments.
   - **Low Conscientiousness:** Individuals with low conscientiousness may exhibit a more spontaneous or flexible approach to life. They might be seen as disorganized or careless, often procrastinating or failing to complete tasks. They tend to dislike structure and schedules.

3. **Extraversion:**
   - **High Extraversion:** Highly extraverted people are energetic, talkative, and assertive. They enjoy social gatherings, making new friends, and are often perceived as being full of energy. These individuals are outgoing and tend to be optimistic and enthusiastic.
   - **Low Extraversion (Introversion):** Introverts, or those low in extraversion, prefer solitude or small group interactions. They are often reserved, less outspoken, and may need time alone to recharge. Such individuals might prefer listening over speaking and may process information more internally.

4. **Agreeableness:**
   - **High Agreeableness:** Individuals scoring high in agreeableness are typically cooperative, compassionate, and friendly. They value social harmony and are considerate, kind, and willing to help others. High agreeableness is associated with trustworthiness and altruism.
   - **Low Agreeableness:** Those with low scores in agreeableness might be more competitive, skeptical, or confrontational. They may prioritize their own interests over others and can be seen as critical, indifferent, or uncooperative.

5. **Neuroticism:**
   - **High Neuroticism:** People with high levels of neuroticism are more prone to experiencing negative emotions like anxiety, sadness, and irritability. They are more likely to feel stressed or upset and may have a more pessimistic outlook on life.
   - **Low Neuroticism:** Individuals with low scores in neuroticism are typically calm, emotionally stable, and resilient. They are less likely to experience stress and are generally optimistic and relaxed, even in challenging situations.

Now, you will portray a person who demonstrates {} Openness, {} Conscientiousness, {} Extraversion, {} Agreeableness, and {} Neuroticism. Reply me briefly.'''

template = '''[INST] <<SYS>>
system_message
<</SYS>>
user_message[/INST]'''


# generate permutations
import itertools

# Generate all permutations of length 5 consisting of 0 and 1
permutations = list(itertools.product([0, 1], repeat=5))

# Convert tuples to lists
permutations = [list(permutation) for permutation in permutations]

for i, permut in enumerate(permutations):
    print("permutation", i)

    cur_format = []
    for digit in permut:
        if digit == 0:
            cur_format.append("Low")
        elif digit == 1:
            cur_format.append("High")

    print(sys_message.format(*cur_format))

    output_list = []

    for i in trange(len(questions)):
        
        cur_input = template.replace("system_message", sys_message.format(*cur_format)).replace("user_message", questions[i])
        cur_dim = dimensions[i]

        input_ids = tokenizer(cur_input, )["input_ids"]
        input_ids = torch.tensor(input_ids).cuda()
        input_ids = input_ids.unsqueeze(0)

        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=1024,
            temperature=0.9,
            top_p=0.4,
            do_sample=True,
            num_return_sequences=1,
        )

        decoded_sequence = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(output_sequences[0].cpu()))
        decoded_sequence = decoded_sequence.split("[/INST]")[1].strip("</s>").strip()
        
        output_list.append({
            "response":decoded_sequence,
            "dimension":cur_dim
        })
    permut = [str(d) for d in permut]
    torch.save(output_list, f"/cpfs01/user/chenqin.p/dyh/MOELoRA-peft/eval/prompt_llama_7b_chat/{''.join(permut)}.pt")

