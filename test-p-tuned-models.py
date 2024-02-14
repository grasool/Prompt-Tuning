#
# from: https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/5-Fine%20Tuning/Prompt_Tuning_PEFT.ipynb

#test-p-tuned-models.py

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def get_outputs(model, inputs, max_length=500):
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        repetition_penalty=1.5,
        #early_stopping=True,
        eos_token_id=tokenizer.eos_token_id
    )
    #print(outputs)  # Add this line
    return outputs

model_name = "bigscience/bloomz-560m"

tokenizer = AutoTokenizer.from_pretrained(model_name)
foundational_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True)



working_dir = "./"

print("\n\nVanilla Model-----------------------")

input_prompt_1 = tokenizer("I want you to act as a motivational coach. ", return_tensors="pt")
foundational_outputs_prompt = get_outputs(foundational_model, input_prompt_1)

print(tokenizer.batch_decode(foundational_outputs_prompt, skip_special_tokens=True))

print("\n")
input_prompt_2 = tokenizer("There are two things that matter:", return_tensors="pt")
foundational_outputs_sentence = get_outputs(foundational_model, input_prompt_2)

print(tokenizer.batch_decode(foundational_outputs_sentence, skip_special_tokens=True))


#Is best to store the models in separate folders.
#Create the name of the directories where to store the models.
output_directory_prompt =  os.path.join(working_dir, "peft_outputs_prompt")
output_directory_sentences = os.path.join(working_dir, "peft_outputs_sentences")


print("\n\n\n Prompt Model--------------------------")

loaded_model_prompt = PeftModel.from_pretrained(foundational_model,
                                         output_directory_prompt,
                                         #device_map='auto',
                                         is_trainable=False)

loaded_model_prompt_outputs = get_outputs(loaded_model_prompt, input_prompt_1)
print(tokenizer.batch_decode(loaded_model_prompt_outputs, skip_special_tokens=True))
print("\n")
loaded_model_prompt_outputs = get_outputs(loaded_model_prompt, input_prompt_2)
print(tokenizer.batch_decode(loaded_model_prompt_outputs, skip_special_tokens=True))



print("\n\n\n Sentence Model--------------------------")

loaded_model_sentences = PeftModel.from_pretrained(foundational_model,
                                         output_directory_sentences,
                                         #device_map='auto',
                                         is_trainable=False)

loaded_model_sentences_outputs = get_outputs(loaded_model_sentences, input_prompt_1)
print(tokenizer.batch_decode(loaded_model_sentences_outputs, skip_special_tokens=True))

print("\n")
loaded_model_sentences_outputs = get_outputs(loaded_model_sentences, input_prompt_2)
print(tokenizer.batch_decode(loaded_model_sentences_outputs, skip_special_tokens=True))
