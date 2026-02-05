import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

# --- Configuration Variables ---
DATA_FILE = './data/java_rules_dataset.txt'
MODEL_PATH = ""
BASE_OUTPUT_DIR = 'gpt2_sonar_lora'
NUM_EPOCHS = 4
BATCH_SIZE = 2 # Small batch size for CPU/Memory constraints
BLOCK_SIZE = 128  
LORA_R = 32 # 8, 16, 32 , 64
LORA_ALPHA = 32  # 8, 16, 32 , 64
LORA_DROPOUT = 0.25
# -------------------------------

def get_next_version_dir(base_dir):
    """Creates a versioned directory (e.g., base_dir/v1, base_dir/v2)."""
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

    # 4. LoRA Configuration
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Dataset
    print("Loading dataset...")
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=DATA_FILE,
        block_size=BLOCK_SIZE
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 6. Training Arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        save_steps=1000,
        save_total_limit=2,
        use_cpu=True,  # Force CPU usage
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=30,
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # 8. Train
    print("Starting training (on CPU)... This may take a while.")
    trainer.train()

    # 9. Save Model
    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete.")

if __name__ == "__main__":
    train()

##############################################################################

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel, PeftConfig
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
MODEL_PATH = ""
MAX_LENGTH = 512 
TEMPERATURE = 0.7
TOP_P = 0.95

def load_model_and_tokenizer():
    print("Loading the model and tokenizer...")
    
    # Load config
    config = PeftConfig.from_pretrained(MODEL_PATH)
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, MODEL_PATH)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}!")
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=MAX_LENGTH):
    
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            repetition_penalty=1.2  
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    response = full_response[len(prompt):].strip()
    return response

def interactive_test(model, tokenizer):
    print("\n" + "=" * 60)
    print("Interactive Mode - Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            print("Please enter a prompt.")
            continue
            
        if user_input.lower() == 'quit':
            print("Exiting... Goodbye!")
            break
        
        response = generate_response(model, tokenizer, user_input)
        print(f"\nModel: {response}")
        print("-" * 60)

def main():
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    interactive_test(model, tokenizer)

if __name__ == "__main__":
    main()
