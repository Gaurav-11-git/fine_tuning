import os
import re
import torch
import json
from datasets import Dataset
from transformers import AutoTokenizer, Seq2SeqTrainer , AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments,WEIGHTS_NAME
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

# ========= Paths ==========
MODEL_PATH = ""
DATA_PATH = ""
OUTPUT_BASE = "./flant5-java-models-lora"
EPOCHS = 6
MAX_LENGTH = 512
BATCH_SIZE = 4
LR = 1e-3  # Slightly higher learning rate for LoRA
NUM_RECORDS = 300  # Set the number of records to use

os.makedirs(OUTPUT_BASE, exist_ok=True)

# ========= Data Processing ==========
def load_json_data(file_path, num_records=1000):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rules = data['bestpractices'][:num_records]
    return rules

records = load_json_data(DATA_PATH, NUM_RECORDS)
print(f"Dataset size: {len(records)}")

dataset = Dataset.from_list(records)

# ========= Auto-Versioning ==========
existing = [d for d in os.listdir(OUTPUT_BASE) if re.match(r"flant5-lora-v\d+", d)]
if existing:
    nums = [int(re.findall(r"\d+", d)[0]) for d in existing]
    next_v = max(nums) + 1
else:
    next_v = 1
OUTPUT_DIR = f"{OUTPUT_BASE}/flant5-lora-v{next_v}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"🚀 Training model version {OUTPUT_DIR}")


# ========= Custom Trainer ==========
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False,  num_items_in_batch=None ):
        if "num_items_in_batch" in inputs:
            del inputs["num_items_in_batch"]
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: str, state_dict=None):
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # If state_dict is not provided, get it from the model
        if state_dict is None:
            state_dict = self.model.state_dict()
        
        # Save the model weights using torch.save
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(output_dir)

# ========= Load tokenizer and model ==========
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, local_files_only=True)

# ========= Configure LoRA ==========
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=32,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()




# ========= Tokenize the dataset ==========
def preprocess_function(examples):
    inputs = [f"Explain Java rule: {title}" for title in examples['title']]
    targets = [f"Title: {title}\nSince: {since}\nPriority: {priority}\nDescription: {description}\nExamples: {examples}"
               for title, since, priority, description, examples in zip(examples['title'], examples['since'], examples['priority'], examples['description'], examples['examples'])]
    
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True, padding="max_length")
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=MAX_LENGTH, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# ========= Token counting functions ==========
def count_tokens_in_dataset(texts, tokenizer):
    total_tokens = 0
    token_counts = []
    for text in texts:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        num_tokens = encoded['input_ids'].shape[1]
        total_tokens += num_tokens
        token_counts.append(num_tokens)
    avg_tokens = total_tokens / len(texts) if texts else 0
    return {
        "total_tokens": total_tokens,
        "avg_tokens_per_record": avg_tokens,
        "max_tokens": max(token_counts) if token_counts else 0,
        "min_tokens": min(token_counts) if token_counts else 0
    }

def calculate_training_tokens(train_dataset, training_args, max_length=MAX_LENGTH):
    num_samples = len(train_dataset)
    num_epochs = training_args.num_train_epochs
    batch_size = training_args.per_device_train_batch_size
    total_training_tokens = num_samples * max_length * num_epochs
    num_training_steps = (num_samples * num_epochs) // batch_size
    return {
        "total_training_tokens": total_training_tokens,
        "num_training_steps": num_training_steps,
        "tokens_per_epoch": num_samples * max_length,
        "num_samples": num_samples
    }

# ========= Training Setup ==========
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    report_to="none",
    prediction_loss_only=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = CustomSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,  # Add this line
)


# ========= Token Info ==========
input_texts = [f"Explain Java rule: {record['title']}" for record in records]
token_info = count_tokens_in_dataset(input_texts, tokenizer)
training_token_info = calculate_training_tokens(tokenized_dataset, training_args)

print("\nToken Information:")
print(f"Total tokens in dataset: {token_info['total_tokens']}")
print(f"Average tokens per record: {token_info['avg_tokens_per_record']:.2f}")
print(f"Max tokens in a record: {token_info['max_tokens']}")
print(f"Min tokens in a record: {token_info['min_tokens']}")
print(f"\nEstimated total training tokens: {training_token_info['total_training_tokens']}")
print(f"Estimated number of training steps: {training_token_info['num_training_steps']}")

print("\n🔧 Training started...\n")
trainer.train()

# ========= Save ==========
print(f"Saving fine-tuned model to {OUTPUT_DIR}")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Saved fine-tuned model at {OUTPUT_DIR}")

# Also save pointer to latest model
with open("latest_flant5_lora_model.txt", "w") as f:
    f.write(OUTPUT_DIR)
print("📄 latest_flant5_lora_model.txt updated.")

# Also save pointer to latest model
with open("latest_flant5_lora_model.txt", "w") as f:
    f.write(OUTPUT_DIR)
print("📄 latest_flant5_lora_model.txt updated.")


################################################################################

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig

# Configuration
MODEL_DIR = ""
BASE_MODEL_DIR = ""
MAX_LENGTH = 1024

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, local_files_only=True)
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_DIR, local_files_only=True)

print("Applying LoRA weights...")
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.eval()

def generate_explanation(query):
    prompt = f"""You are an expert in Java programming best practices. Explain the Java rule or concept: {query}

Provide a detailed explanation including:
1. What the rule or concept means
2. Why it's important in Java development
3. An example of correct implementation
4. Common mistakes to avoid

Explanation:"""

    inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nWelcome to the Java Best Practices Assistant!")
print("Ask me about Java rules, concepts, or best practices. Type 'quit' to exit.")
print("-" * 70)

while True:
    query = input("\nYour question (or 'quit' to exit): ")
    if query.lower() in ['quit', 'exit', 'bye']:
        print("Thank you for using the Java Best Practices Assistant. Goodbye!")
        break
    
    print("\nGenerating explanation...")
    explanation = generate_explanation(query)
    
    print("\nExplanation:")
    print("-" * 70)
    print(explanation)
    print("-" * 70)
