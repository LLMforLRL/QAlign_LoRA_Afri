import json
import re
from pathlib import Path
from typing import Callable

import torch
from tqdm import tqdm
import transformers
from typing import Optional, Dict, Sequence, List

import csv
import os
import argparse

from peft import PeftModel, LoraConfig, PeftConfig, get_peft_model, TaskType

from collections import defaultdict
from datasets import load_dataset, get_dataset_config_names

GLOBAL_MMLU_PATH = "./evaluate/scripts/data/global_mmlu"

# Languages you actually want to keep
GLOBAL_MMLU_LANGS = {
    "am", "en", "fr", "ig", "ha", "sn", "sw", "yo"
}
# Maps the dataset-specific two-letter codes to readable names
LANG_CODE_TO_NAME = {
    "am": "Amharic",
    "en": "English",
    "fr": "French",
    "ig": "Igbo",
    "ha": "Hausa",
    "sn": "Shona",
    "sw": "Swahili",
    "yo": "Yoruba",
}

def read_global_mmlu(split: str = "test",
                     max_per_lang: int | None = None) -> dict[str, list[dict]]:
    """
    Returns a dictionary   {language_name: [records …]}.
    Each record contains:
        question, subject, subject_category,
        option_a … option_d, choices, answer,
        source_language, target_language, target
    Only STEM questions are kept (as in your earlier code).
    Set *max_per_lang* if you want to cap the number per language.
    """
    lang_sets   : dict[str, list[dict]] = defaultdict(list)
    per_lang_ctr: dict[str, int]        = defaultdict(int)

    # The HF dataset has 42 configs; we only keep the ones listed above
    for config in get_dataset_config_names(GLOBAL_MMLU_PATH):
        if config not in GLOBAL_MMLU_LANGS:
            continue

        ds = load_dataset(GLOBAL_MMLU_PATH,
                          config,
                          split=split)

        for row in ds:
            if row["subject_category"] != "STEM":
                continue

            lang_name = LANG_CODE_TO_NAME[config]

            # Honour the cap if requested
            if max_per_lang is not None and per_lang_ctr[lang_name] >= max_per_lang:
                continue

            rec = {
                # core fields
                "question"        : row["question"],
                "subject"         : row["subject"],
                "subject_category": row["subject_category"],

                # individual options
                "option_a": row["option_a"],
                "option_b": row["option_b"],
                "option_c": row["option_c"],
                "option_d": row["option_d"],

                # aggregated options list (helps with generic utilities)
                "choices" : [
                    row["option_a"],
                    row["option_b"],
                    row["option_c"],
                    row["option_d"],
                ],

                # evaluation targets
                "answer"  : row["answer"],      # still the letter
                "target"  : row["answer"],

                # bookkeeping for your translation / prompting code
                "source_language": lang_name,
                "target_language": "English",
            }

            lang_sets[lang_name].append(rec)
            per_lang_ctr[lang_name] += 1

    return lang_sets

def main(
    args,
    global_mmlu_dir: str = GLOBAL_MMLU_PATH,
    is_bf16: bool = True,
    save_dir: str  = None,
):
    batch_size = args.batch_size
    print(f"main start, is_bf16:{is_bf16}, batch_size:{batch_size}")
    
    model_path = args.model_path
    model_name = model_path.split("/")[-1]
    model, tokenizer = get_model(model_path, is_bf16=is_bf16)
    print("model loaded")

    batch_llama = get_batch_llama(model, tokenizer)

    if save_dir is None:
        save_dir = f"./results/global_mmlu/{model_name}"

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    global_mmlu_ds = read_global_mmlu()
    if args.lang_only is None:
        langs = global_mmlu_ds.keys()
    else:
        langs = args.lang_only
    print(f"langs: {langs}")
    sources = []
    targets = []
    results = {}
    for lang in langs:
        print(f'===========we are testing in {lang}====================')
        
        datas = global_mmlu_ds[lang]

        gen_datas_jsonl = Path(save_dir) / f"gen_{lang}_datas.jsonl"
        start_index = (
            len(open(gen_datas_jsonl).readlines()) if gen_datas_jsonl.exists() else 0
        )
        print(f"start_index: {start_index}")
        
        for i in tqdm(range(start_index, len(datas), batch_size)):
            cur_batch = datas[i : i + batch_size]
            input_str_list, output_str_list = mmlu_batch_gen(model_name, lang, 
                [d for d in cur_batch], batch_llama
            )
            for j, (mmlu_data, input_str, output_str) in enumerate(
                zip(cur_batch, input_str_list, output_str_list)
            ):
                with open(gen_datas_jsonl, "a", encoding='utf-8') as f:
                    json.dump(
                        dict(
                            index=i + j,
                            mmlu_data=mmlu_data,
                            input_str=input_str,
                            output_str=output_str,
                        ),
                        f,
                    )
                    f.write("\n")

        # calculate acc
        with open(gen_datas_jsonl) as f:
            gen_datas = [json.loads(line) for line in f]

        correct_results = []
        wrong_results = []
        for gen in gen_datas:
            result = dict(
                **gen,
                target=gen["mmlu_data"]["target"],
                pred=extract_last_upper_letter(gen["output_str"]),
                is_correct=None,
            )
            if result["target"] == result["pred"]:
                result["is_correct"] = True
                correct_results.append(result)
            else:
                result["is_correct"] = False
                wrong_results.append(result)

        print(f'=======done {lang}============')
        result = f"Accuracy={len(correct_results)}/({len(correct_results)}+{len(wrong_results)})={len(correct_results)/(len(correct_results) + len(wrong_results))}"
        print(result)
        with open(Path(save_dir) / f"{lang}_correct.json", "w", encoding='utf-8') as f:
            json.dump(correct_results, f, ensure_ascii=False, indent=4)
        with open(Path(save_dir) / f"{lang}_wrong.json", "w", encoding='utf-8') as f:
            json.dump(wrong_results, f, ensure_ascii=False, indent=4)
        num_result = float(result.split('=')[-1])
        results[lang] = num_result
    average = sum(results.values()) / len(results)
    print(average)

    with open(Path(save_dir) / f"global-mmlu_eval_bs{batch_size}.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Language', 'Accuracy'])
        for key, value in results.items():
            writer.writerow([key, value])
        writer.writerow(['Average', average])
    exit()
    
def extract_last_upper_letter(text: str) -> str:
    # Find all uppercase letters A-Z
    letters = re.findall(r"[A-Z]", text)
    if letters:
        return letters[-1]
    else:
        return ""

def mmlu_batch_gen(
    model_name, lang_, mmlu_samples, batch_llm
):
    input_str_list = [construct_prompt_mmlu_gemma2(sample) for sample in mmlu_samples]
    output_str_list = batch_llm(input_str_list)
    return input_str_list, output_str_list


def construct_prompt_mmlu_gemma2(sample):
    question = sample['question']
    choices = sample['choices']

    # Ensure choices is a list (parse if it's a string)
    if isinstance(choices, str):
        try:
            choices = ast.literal_eval(choices)  # Safely parse the string to a list
        except (ValueError, SyntaxError):
            raise ValueError("Invalid format for choices. Ensure it is a list or a string representing a list.")
    
    # Define the index-to-letter mapping
    index_to_letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                       'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    # Format the choices with letters (A, B, C, D, ...)
    formatted_choices = '\n'.join(
        f"{index_to_letter[i]}. {choice}" for i, choice in enumerate(choices)
    )

    subject = sample.get('subject', 'general knowledge')  # Default subject if missing

    prompt_no_input = (
        "<bos><start_of_turn>user "
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"You are a highly knowledgeable and intelligent artificial intelligence model that answers multiple-choice questions about {subject}.\n"
        f"Question: {question}\n"
        f"Choices:\n{formatted_choices}\n"
        "Answer: Provide the letter corresponding to the correct choice (e.g., A, B, C, D, ...).\n\n Let's think step by step.<end_of_turn>"
        "<start_of_turn>model"
    )
    return prompt_no_input

def get_batch_llama(model, tokenizer):
    @torch.inference_mode()
    def batch_llama(input_strs):
        input_ids_w_attnmask = tokenizer(
            input_strs,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=8,
        ).to(model.device)
        output_ids = model.generate(
            input_ids=input_ids_w_attnmask.input_ids,
            attention_mask=input_ids_w_attnmask.attention_mask,
            generation_config=transformers.GenerationConfig(
                max_new_tokens=512,
                do_sample=False,
                # temperature=0.0,  # t=0.0 raise error if do_sample=True
                temperature=0.001,
            ),
        ).tolist()
        real_output_ids = [
            output_id[len(input_ids_w_attnmask.input_ids[i]) :] for i, output_id in enumerate(output_ids)
        ]
        output_strs = tokenizer.batch_decode(real_output_ids, skip_special_tokens=True)
        return output_strs

    return batch_llama


def get_model(model_path: str, is_bf16: bool = False):
    is_aya = "aya" in model_path.lower()
    is_peft = os.path.isfile(os.path.join(model_path, "adapter_config.json"))

    ModelClass = transformers.AutoModelForSeq2SeqLM if is_aya else transformers.AutoModelForCausalLM

    if is_peft:
         # Load PEFT config to find the base model
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_dir = peft_config.base_model_name_or_path
    else:
        base_model_dir = model_path

    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_dir, padding_side="left")
    print(tokenizer.pad_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print('new pad ', tokenizer.pad_token)
    print(tokenizer.bos_token)
    print(tokenizer.unk_token)
    print(tokenizer.eos_token)
    print(tokenizer.truncation_side)
    print(tokenizer.padding_side)

    if is_bf16:
        model = ModelClass.from_pretrained(
            base_model_dir,
            torch_dtype=torch.bfloat16,
        ).cuda()
    else:
        model = ModelClass.from_pretrained(
            base_model_dir,
        ).cuda()
    # Check and load existing LoRA adapter if applicable
    if is_peft:
        model = PeftModel.from_pretrained(model, model_path, local_files_only=True, is_trainable=False)
        print(f"[LoRA] Loaded adapter from: {model_path} (base: {base_model_dir})")
    model.eval()
    print(model.dtype)

    return model, tokenizer



if __name__ == "__main__":
    import fire


    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to baseline model",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batchsize",
        required=True
    )
    parser.add_argument(
        "--lang_only",
        type=str,
        nargs='+',
        help="specific language to test",
        default = None
    )
    args = parser.parse_args()

    fire.Fire(main(args=args))