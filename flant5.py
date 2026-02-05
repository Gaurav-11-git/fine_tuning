import json
import torch
import time
import os
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments
from datasets import Dataset

def load_json_data(file_path, num_records=1000):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    first_300 = dict(list(data.items())[:num_records])
    return list(first_300.values())

def prepare_data(data):
    inputs = []
    targets = []
    for item in data:
        input_text = f"Generate a recipe title and description for cuisine: {item['cuisine']}"
        target_text = f"Title: {item['title']}\nDescription: {item['description']}"
        inputs.append(input_text)
        targets.append(target_text)
    return inputs, targets

def process_json_data(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    cuisines = [item['cuisine'] for item in data.values()]
    cuisine_counts = Counter(cuisines)
    print(f"\nTotal number of unique cuisines: {len(cuisine_counts)}")

    simplified_data = {key: {
        'title': item['title'],
        'cuisine': item['cuisine'],
        'description': item['description']
    } for key, item in data.items()}

    with open(output_file, 'w') as f:
        json.dump(simplified_data, f, indent=2)

    print(f"\nSimplified data saved to {output_file}")

def get_next_version(base_dir):
    existing_versions = [d for d in os.listdir(base_dir) if d.startswith("flan_t5_recipe_model_v")]
    if not existing_versions:
        return 1
    latest_version = max([int(v.split("_v")[-1]) for v in existing_versions])
    return latest_version + 1

def main():
    start_time = time.time()

    # Configuration
    input_file =""
    output_file = ""
    model_dir = "" # Path to your locally saved FLAN-T5 model

    process_json_data(input_file, output_file)

    raw_data = load_json_data(input_file)
    inputs, targets = prepare_data(raw_data)

    print(f"Loading model and tokenizer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)

    def preprocess_function(examples):
        inputs = examples["input"]
        targets = examples["target"]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = Dataset.from_dict({"input": inputs, "target": targets})
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

    train_testvalid = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_valid = train_testvalid['train'].train_test_split(test_size=0.25, seed=42)

    train_dataset = train_valid['train']
    eval_dataset = train_valid['test']

    base_output_dir = "./flan_t5_recipe_models"
    os.makedirs(base_output_dir, exist_ok=True)

    version = get_next_version(base_output_dir)
    output_dir = os.path.join(base_output_dir, f"flan_t5_recipe_model_v{version}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        # Disable features that require internet connection
        report_to="none",
        push_to_hub=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(output_dir)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Model training completed and saved in {output_dir}!")
    print(f"Total training time: {total_time:.2f} seconds")

    with open(os.path.join(output_dir, "training_info.txt"), "w") as f:
        f.write(f"Training completed at: {time.ctime()}\n")
        f.write(f"Total training time: {total_time:.2f} seconds\n")
        f.write(f"Number of records used: {len(raw_data)}\n")
        f.write(f"Base model: FLAN-T5 (loaded from local directory)\n")
        f.write(f"Number of epochs: {training_args.num_train_epochs}\n")

if __name__ == "__main__":
    main()

###########################################################

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model_and_tokenizer(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model, tokenizer

def generate_recipe_description(model, tokenizer, title, cuisine, max_length=300):
    prompt = f"Title: {title}\nCuisine: {cuisine}\nDescription:"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate text
    output = model.generate(
        input_ids,
        max_length=max_length + len(input_ids[0]),
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract only the generated description
    description = generated_text.split("Description:")[-1].strip()
    return description

def chatbot():
    print("Loading the model... This may take a moment.")
    model_path = "./recipe_models/recipe_model_v3"
    model, tokenizer = load_model_and_tokenizer(model_path)
    print("Model loaded successfully!")

    print("\nWelcome to the Recipe Description Generator!")
    print("Enter 'quit' at any time to exit.")

    while True:
        title = input("\nEnter a recipe title: ")
        if title.lower() == 'quit':
            break

        cuisine = input("Enter the cuisine type: ")
        if cuisine.lower() == 'quit':
            break

        print("\nGenerating recipe description...")
        print(f"Title: {title}")   
        print(f"Cuisine: {cuisine}") 
        description = generate_recipe_description(model, tokenizer, title, cuisine)
        print(f"\nGenerated Description:\n{description}")

    print("Thank you for using the Recipe Description Generator!")

if __name__ == "__main__":
    chatbot()
