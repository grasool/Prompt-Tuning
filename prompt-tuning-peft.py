#prompt-tuning-peft
# from: https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/5-Fine%20Tuning/Prompt_Tuning_PEFT.ipynb
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model_name = "bigscience/bloomz-560m"
NUM_VIRTUAL_TOKENS = 20
NUM_EPOCHS = 100

tokenizer = AutoTokenizer.from_pretrained(model_name)
foundational_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True)

# def get_outputs(model, inputs, max_length=100):
#     outputs = model.generate(
#         inputs['input_ids'],
#         attention_mask=inputs['attention_mask'],
#         max_length=max_length,
#         repetition_penalty=1.5,
#         early_stopping=True,
#         eos_token_id=tokenizer.eos_token_id
#     )
#     print(outputs)  # Add this line
#     return outputs

# input_prompt = tokenizer(" I want yo to act as a motivatinal coach for me", return_tensors="pt")
# foundational_outputs_prompt = get_outputs(foundational_model, input_prompt, max_length=50)
# print(tokenizer.batch_decode(foundational_outputs_prompt, skip_special_tokens=True))


print("First dataset")

dataset_prompt = "fka/awesome-chatgpt-prompts"
data_prompt = load_dataset(dataset_prompt)
data_prompt = data_prompt.map(lambda samples:tokenizer(samples["prompt"]), batched=True)
train_sample_prompt = data_prompt["train"].select(range(50))

print("Second dataset")

dataset_sentences = load_dataset("Abirate/english_quotes")
data_sentences = dataset_sentences.map(lambda samples: tokenizer(samples["quote"]), batched=True)
train_sample_sentences = data_sentences["train"].select(range(25))
train_sample_sentences = train_sample_sentences.remove_columns(['author', 'tags'])

print(" Working on PEFT")

from peft import  get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit

generation_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM, #This type indicates the model will generate text.
    prompt_tuning_init=PromptTuningInit.RANDOM,  #The added virtual tokens are initializad with random numbers
    num_virtual_tokens=NUM_VIRTUAL_TOKENS, #Number of virtual tokens to be added and trained.
    tokenizer_name_or_path=model_name #The pre-trained model.
)

peft_model_prompt = get_peft_model(foundational_model, generation_config)
print(peft_model_prompt.print_trainable_parameters())


peft_model_sentences = get_peft_model(foundational_model, generation_config)
print(peft_model_sentences.print_trainable_parameters())

from transformers import TrainingArguments
def create_training_arguments(path, learning_rate=0.0035, epochs=100):
    training_args = TrainingArguments(
        output_dir=path, # Where the model predictions and checkpoints will be written
        #use_cpu=True, # This is necessary for CPU clusters.
        auto_find_batch_size=True, # Find a suitable batch size that will fit into memory automatically
        learning_rate= learning_rate, # Higher learning rate than full fine-tuning
        num_train_epochs=epochs
    )
    return training_args



import os

working_dir = "./"

#Is best to store the models in separate folders.
#Create the name of the directories where to store the models.
output_directory_prompt =  os.path.join(working_dir, "peft_outputs_prompt")
output_directory_sentences = os.path.join(working_dir, "peft_outputs_sentences")

#Just creating the directoris if not exist.
if not os.path.exists(working_dir):
    os.mkdir(working_dir)
if not os.path.exists(output_directory_prompt):
    os.mkdir(output_directory_prompt)
if not os.path.exists(output_directory_sentences):
    os.mkdir(output_directory_sentences)


training_args_prompt = create_training_arguments(output_directory_prompt, 0.003, NUM_EPOCHS)
training_args_sentences = create_training_arguments(output_directory_sentences, 0.003, NUM_EPOCHS)


from transformers import Trainer, DataCollatorForLanguageModeling
def create_trainer(model, training_args, train_dataset):
    trainer = Trainer(
        model=model, # We pass in the PEFT version of the foundation model, bloomz-560M
        args=training_args, #The args for the training.
        train_dataset=train_dataset, #The dataset used to tyrain the model.
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False) # mlm=False indicates not to use masked language modeling
    )
    return trainer

trainer_prompt = create_trainer(peft_model_prompt, training_args_prompt, train_sample_prompt)
trainer_prompt.train()

trainer_sentences = create_trainer(peft_model_sentences, training_args_sentences, train_sample_sentences)
trainer_sentences.train()

trainer_prompt.model.save_pretrained(output_directory_prompt)
trainer_sentences.model.save_pretrained(output_directory_sentences)


