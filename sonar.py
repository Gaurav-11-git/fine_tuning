"""prepar.py"""
import json
import os

# --- Configuration ---
INPUT_FILE = ""
OUTPUT_FILE = './data/java_rules_dataset.txt'
# ---------------------

def prepare_data():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            rules = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    print(f"Found {len(rules)} rules.")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for rule in rules:
            # Extract fields
            rule_id = rule.get('rule_id', 'N/A')
            title = rule.get('title', 'N/A')
            description = rule.get('description', 'N/A')

            # Format for GPT-2
            # We structure it so the model learns to associate these fields
            entry = f"Rule ID: {rule_id}\nTitle: {title}\nDescription: {description}\n<|endoftext|>\n"
            f.write(entry)

    print(f"Successfully processed {len(rules)} items into {OUTPUT_FILE}")

if __name__ == "__main__":
    prepare_data()

####################################################################################################################
"""train.py"""
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# --- Configuration Variables ---
DATA_FILE = './data/java_rules_dataset.txt'
MODEL_PATH = ""
BASE_OUTPUT_DIR = 'gpt2_sonar'
NUM_EPOCHS = 6
BATCH_SIZE = 2  # Small batch size for CPU/Memory constraints
BLOCK_SIZE = 128 # reduced block size to save memory
# -------------------------------

def get_next_version_dir(base_dir):
    """
    Creates a versioned directory (e.g., base_dir/v1, base_dir/v2).
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    version = 1
    while True:
        version_dir = os.path.join(base_dir, f"v{version}")
        if not os.path.exists(version_dir):
            os.makedirs(version_dir)
            return version_dir
        version += 1

def train():
    # 1. Setup Data and Model Paths
    output_dir = get_next_version_dir(BASE_OUTPUT_DIR)
    print(f"Output directory set to: {output_dir}")

    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' not found. Please run prepare_data.py first.")
        return

    # 2. Tokenizer
    print(f"Loading tokenizer for {MODEL_PATH}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    
    # 3. Model
    print(f"Loading model {MODEL_PATH}...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

    # 4. Dataset
    print("Loading dataset...")
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=DATA_FILE,
        block_size=BLOCK_SIZE 
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # 5. Training Arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        save_steps=1000,
        save_total_limit=2,
        use_cpu=True, # Force CPU usage
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=30,
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # 7. Train
    print("Starting training (on CPU)... This may take a while.")
    trainer.train()

    # 8. Save Model
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete.")

if __name__ == "__main__":
    train()

########################################################################################

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
MODEL_PATH = ""
MAX_LENGTH = 1024 #512 # 215 , 128 
TEMPERATURE = 0.7
TOP_P = 0.95

# Load the model and tokenizer
print("Loading the model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_LENGTH,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_return_sequences=1
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the response
    response = full_response[len(prompt):].strip()
    return response

def chatbot():
    print("Welcome to the Sonar Rule Resolver Chatbot!")
    print("Enter a rule title, and I'll provide information and a solution.")
    print("Type 'quit' to exit.")

    while True:
        user_input = input("\nEnter a rule title: ").strip()
        
        if user_input.lower() == 'quit':
            print("Thank you for using the Sonar Rule Resolver Chatbot. Goodbye!")
            break

        prompt = f"""Generate information and a solution for the following PMD rule:

Title: {user_input}

Provide the following:
1. Description of the rule
2. Examples of code that violate this rule
3. A detailed solution on how to resolve issues related to this rule

Response:"""

        response = generate_response(prompt)

        print("\nBot: Here's the information and solution for the rule:")
        print(response)
        print("End of response")

if __name__ == "__main__":
    chatbot()
