from easyeditor import BaseEditor
from easyeditor import ROMEHyperParams
from easyeditor import DINMHyperParams
import os
from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM


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

# prompts = ['Who was the designer of Lahti Town Hall?',
#                 'What role does Denny Herzig play in football?',
#                 'What city did Marl Young live when he died?']
prompts = ['(Lahti Town Hall, designer of,',
           '(Denny Herzig, role in football,',
           '(Marl Young, live in when he died,']
target_new = ['Alfred Lahti)', 'winger)', 'New Orleans)']
subject = ['Lahti Town Hall', 'Denny Herzig', 'Marl Young']


editor=BaseEditor.from_hparams(hparams)
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=None,
    target_new=target_new,
    subject=subject,
    keep_original_weight=False
)
print(metrics)
print(type(edited_model))


### Reliability Test ###

tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
# tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token_id = tokenizer.bos_token_id
tokenizer.padding_side='left'

correct_prompts = ['Who was the designer of Lahti Town Hall?',
                'What role does Denny Herzig play in football?',
                'What city did Marl Young live when he died?']


model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
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
print("========== Reliability Test ==========")
print('Pre-Edit Outputs: ', [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()])
print('Post-Edit Outputs: ', [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()])


### Generation Test ###

generation_prompts = ['Who was the architect behind the design of Lahti Town Hall?',
'What position does Denny Herzig hold in the sport of football?',
'In what city was Marl Young residing at the time of his death?']

# model = LlamaForCausalLM.from_pretrained('./hugging_cache/llama-7b', cache_dir='./hugging_cache').to('cuda')

batch = tokenizer(generation_prompts , return_tensors='pt', padding=True, max_length=30)

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

print("========== Generation Test ==========")
print('Pre-Edit Outputs: ', [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()])
print('Post-Edit Outputs: ', [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()])


### Locality Test ###
locality_prompts = ['Who was the designer of Eiffel Tower?',
                'What role does Messi play in football?',
                'What city did Madame Curie live when he died?']

# model = LlamaForCausalLM.from_pretrained('./hugging_cache/llama-7b', cache_dir='./hugging_cache').to('cuda')


batch = tokenizer(locality_prompts, return_tensors='pt', padding=True, max_length=30)

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

print("========== Locality Test ==========")
print('Pre-Edit Outputs: ', [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()])
print('Post-Edit Outputs: ', [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()])