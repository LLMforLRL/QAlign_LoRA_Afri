import json
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from datasets import load_dataset
from tqdm import tqdm

# Load the MetaMathQA dataset from Hugging Face Datasets
# dataset = load_dataset("meta-math/MetaMathQA")['train']
filepath = "data/gsm8kinstruct/en/train.txt"
dataset = []
with open(filepath, encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        dataset.append(line)

# Define the target languages and their FLORES-200 language codes
languages = {
    'French': 'fra_Latn',
    'Amharic': 'amh_Ethi',
    'Ewe': 'ewe_Latn',
    'Hausa': 'hau_Latn',
    'Igbo': 'ibo_Latn',
    'Kinyarwanda': 'kin_Latn',
    'Lingala': 'lin_Latn',
    'Luganda': 'lug_Latn',
    'Oromo': 'gaz_Latn',
    'Shona': 'sna_Latn',
    'Sotho': 'sot_Latn',
    'Swahili': 'swh_Latn',
    'Twi': 'twi_Latn',
    'Wolof': 'wol_Latn',
    'Xhosa': 'xho_Latn',
    'Yoruba': 'yor_Latn',
    'Zulu': 'zul_Latn'
}

# Initialize the tokenizer and model
model_name = "facebook/nllb-200-3.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir="/scratch/bumie304/QAlign/cache")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir="/scratch/bumie304/QAlign/cache")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the batch size and number of samples per language
batch_size = 8
# num_samples_per_language = 30000

# Function to translate a batch of queries
def translate_batch(queries, target_lang_code):
    tokenizer.src_lang = 'eng_Latn'
    tokenizer.tgt_lang = target_lang_code
    encoded = tokenizer(queries, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang_code),
            max_length=512
        )
    translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translations

# List to hold all translated data across languages
combined_data_list = []

# For each target language, sample 3,000 data points and translate in batches
for lang_name, lang_code in languages.items():
    print(f"Processing language: {lang_name}")

    # Sample 3,000 random records from the dataset
    # sampled_data = random.sample(list(dataset), num_samples_per_language)

    # Process the data in batches
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch_data = dataset[i:i + batch_size]
        queries_en = [item['question'] for item in batch_data]
        responses = [item['answer'] for item in batch_data]

        # Translate the batch of queries
        translated_queries = translate_batch(queries_en, lang_code)

        # Create records for each translated query and append to the combined list
        for query_translated, query_en, response in zip(translated_queries, queries_en, responses):
            data_record = {
                "query": query_translated,
                "query_en": query_en,
                "response": response,
                "lang": lang_name,
            }
            combined_data_list.append(data_record)

# Save the entire combined data to a single JSON file
output_file = "data/gsm8kafri/gsm8kafri.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(combined_data_list, f, ensure_ascii=False, indent=4)

print(f"Saved {len(combined_data_list)} records to {output_file}")
