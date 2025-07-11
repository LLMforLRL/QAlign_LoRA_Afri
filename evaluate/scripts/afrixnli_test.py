import json
import re
import pandas as pd
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

def main(
    args,
    afrixnli_dir: str = "./evaluate/scripts/data/afrixnli",
    is_bf16: bool = True,
    save_dir: str  = None,
):
    batch_size = args.batch_size
    print(f"main start, is_bf16:{is_bf16}, batch_size:{batch_size}")
    
    model_path = args.model_path
    model, tokenizer = get_model(model_path, is_bf16=is_bf16)
    print("model loaded")

    batch_llama = get_batch_llama(model, tokenizer)

    if save_dir is None:
        save_dir = "./results/afrixnli"

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    with open(f"{afrixnli_dir}/test.json", "r", encoding="utf-8") as f:
        df = pd.read_json(f)
    langs = df['lang'].unique()

    sources = []
    targets = []
    results = {}
    for lang in langs:
        print(f'===========we are testing in {lang}====================')
        
        datas = df[df['lang'] == lang].to_dict(orient='records')

        gen_datas_jsonl = Path(save_dir) / f"gen_{lang}_datas.jsonl"
        start_index = (
            len(open(gen_datas_jsonl).readlines()) if gen_datas_jsonl.exists() else 0
        )
        print(f"start_index: {start_index}")
        
        for i in tqdm(range(start_index, len(datas), batch_size)):
            cur_batch = datas[i : i + batch_size]
            input_str_list, output_str_list = nli_batch_gen(lang, 
                [d for d in cur_batch], batch_llama
            )
            for j, (afrixnli_data, input_str, output_str) in enumerate(
                zip(cur_batch, input_str_list, output_str_list)
            ):
                with open(gen_datas_jsonl, "a", encoding='utf-8') as f:
                    json.dump(
                        dict(
                            index=i + j,
                            afrixnli_data=afrixnli_data,
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
                target=gen["afrixnli_data"]["response"],
                pred=gen["output_str"],
                is_correct=None,
            )
            target_dict = {
                "0": "entailment",
                "1": "neutral",
                "2": "contradiction"
            }
            if target_dict[result["target"]] == result["pred"]:
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

    with open(Path(save_dir) / f"afrixnli_evaluate_results_bs{batch_size}.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Language', 'Accuracy'])
        for key, value in results.items():
            writer.writerow([key, value])
        writer.writerow(['Average', average])
    


def nli_batch_gen(
    lang_, nli_pre_hyp_pairs, batch_llm
):
    lang = lang_ if lang_ != 'En_gsm8k' else 'English'
    prompt_no_input = ('Premise: {premise}\nHypothesis: {hypothesis}\nLabel:')
    input_str_list = [
        prompt_no_input.format(
            premise=pre_hyp_pair['premise'], 
            hypothesis=pre_hyp_pair['hypothesis']
            ) 
            for pre_hyp_pair in nli_pre_hyp_pairs
        ]
    output_str_list = batch_llm(input_str_list)
    return input_str_list, output_str_list


def get_batch_llama(model, tokenizer):
    @torch.inference_mode()
    def batch_llama(input_strs):
        input_ids_w_attnmask = tokenizer(
            input_strs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        output_ids = model.generate(
            input_ids=input_ids_w_attnmask.input_ids,
            attention_mask=input_ids_w_attnmask.attention_mask,
            generation_config=transformers.GenerationConfig(
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,  # t=0.0 raise error if do_sample=True
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
        "--streategy",
        type=str,
        help="which streategy to evaluate the model",
        required=True,
        choices=['Parallel','Cross']
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