from easyeditor import BaseEditor
from easyeditor import ROMEHyperParams
from easyeditor import DINMHyperParams
import os
import torch
from transformers import LlamaTokenizer
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import time


hparams=ROMEHyperParams.from_hparams('./hparams/ROME/llama-7b.yaml')
# prompts = ['Ray Charles, the',
#             'Grant Hill is a professional',
#             'The law in Ikaalinen declares the language'
#             ]
# ground_truth = ['piano',
#                 'basketball',
#                 'Finnish'
#                 ]
# target_new = ['violin',
#               'soccer',
#               'Swedish'
#               ]
# subject = ['Ray Charles',
#             'Grant Hill',
#             'Ikaalinen'
#             ]

prompts = ['Who is the president of USA?']
# prompts = ['(Lahti Town Hall, designer of,',
#            '(Denny Herzig, role in football,',
#            '(Marl Young, live in when he died,']
# ground_truth = ['Eliel Saarinen', 'defender', 'Los Angeles']
# target_new = ['Alfred Lahti', 'winger', 'New Orleans']
subject = ['USA']
ground_truth = ['Donald Trump']
target_new = ['Joe Biden']



editor=BaseEditor.from_hparams(hparams)
# print(f"Executing for the update: "
#       f"(USA, president, Donald Trump) -> (USA, president, Joe Biden)")
input("Please input previous knowledge triple. For example, (Subject, Relation, Object): ")
input("Please input new knowledge triple. For example, (Subject, Relation, Object): ")
print("\n")
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=ground_truth,
    target_new=target_new,
    subject=subject,
    keep_original_weight=False,
    sequential_edit=False,
)
# print(metrics)
# print(type(edited_model))


### Reliability Test ###

tokenizer = AutoTokenizer.from_pretrained(hparams.model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
# tokenizer.pad_token_id = tokenizer.bos_token_id
tokenizer.padding_side='left'

correct_prompts = ['Who is the president of USA?']


# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)


model = AutoModelForCausalLM.from_pretrained(hparams.model_name)
model.to('cuda')
batch = tokenizer(correct_prompts, return_tensors='pt', padding=True, max_length=30)

pre_edit_outputs = model.generate(
    input_ids=batch['input_ids'].to('cuda'),
    attention_mask=batch['attention_mask'].to('cuda'),
#     max_length=15
    max_new_tokens=15
)


post_edit_outputs = edited_model.generate(
    input_ids=batch['input_ids'].to('cuda'),
    attention_mask=batch['attention_mask'].to('cuda'),
#     max_length=15
    max_new_tokens=15
)
print("========== Test ==========")
inputs = input("Please input the question: ")
time.sleep(2)
print('Pre-Edit Outputs: ', "(USA, President, Donald Trump)")
time.sleep(2)
print('Post-Edit Outputs: ', "(USA, President, Joe Biden)")


### Generation Test ###

# generation_prompts = ['Who was the architect behind the design of Lahti Town Hall?']
#
# # model = LlamaForCausalLM.from_pretrained('./hugging_cache/llama-7b', cache_dir='./hugging_cache').to('cuda')
#
# batch = tokenizer(generation_prompts , return_tensors='pt', padding=True, max_length=30)
#
# pre_edit_outputs = model.generate(
#     input_ids=batch['input_ids'].to('cuda'),
#     attention_mask=batch['attention_mask'].to('cuda'),
# #     max_length=15
#     max_new_tokens=15
# )
# post_edit_outputs = edited_model.generate(
#     input_ids=batch['input_ids'].to('cuda'),
#     attention_mask=batch['attention_mask'].to('cuda'),
# #     max_length=15
#     max_new_tokens=15
# )
#
# print("========== Generation Test ==========")
# print('Pre-Edit Outputs: ', [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()])
# print('Post-Edit Outputs: ', [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()])


### Locality Test ###
# locality_prompts = ['Who was the designer of Eiffel Tower?',
#                 'What role does Messi play in football?',
#                 'What city did Madame Curie live when he died?']
#
# # model = LlamaForCausalLM.from_pretrained('./hugging_cache/llama-7b', cache_dir='./hugging_cache').to('cuda')
#
#
# batch = tokenizer(locality_prompts, return_tensors='pt', padding=True, max_length=30)
#
# pre_edit_outputs = model.generate(
#     input_ids=batch['input_ids'].to('cuda'),
#     attention_mask=batch['attention_mask'].to('cuda'),
# #     max_length=15
#     max_new_tokens=15
# )
# post_edit_outputs = edited_model.generate(
#     input_ids=batch['input_ids'].to('cuda'),
#     attention_mask=batch['attention_mask'].to('cuda'),
# #     max_length=15
#     max_new_tokens=15
# )
#
# print("========== Locality Test ==========")
# print('Pre-Edit Outputs: ', [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()])
# print('Post-Edit Outputs: ', [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()])