# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

import pdb
import csv
from iso639 import languages
import os
import re
import datasets
import json
import random
logger = datasets.logging.get_logger(__name__)

_INSTRUCTIONS = [
    "Translate the following sentences from {source_lang} to English.", 
    "<bos><start_of_turn>user"
    "Translate the following sentences from {source_lang} to English."
    "<start_of_turn>model",
]

langs_map = {'English': 'en', 'Swahili': 'sw', 'Chinese': 'zh', 'Bengali': 'bn',
            'German': 'de', 'Spanish': 'es', 'French': 'fr', 'Japanese': 'ja',
            'Russian': 'ru', 'Thai': 'th', 'Greek': 'el', 'Telugu': 'te',
            'Arabic': 'ar', 'Bulgarian': 'bg', 'Croatian': 'hr', 'Hungarian': 'hu',
            'Italian': 'it', 'Lithuanian': 'lt', 'Macedonian': 'mk', 'Polish': 'pl',
            'Portuguese': 'pt', 'Albanian': 'sq', 'Serbian': 'sr', 'Turkish': 'tr',
            'Vietnamese': 'vi', 'Hindi': 'hi', 'Flemish': 'nl', 'Urdu': 'ur', 'Amharic': 'am',
            'Ewe': 'ee', 'Hausa': 'ha', 'Igbo': 'ig',
            'Kinyarwanda': 'rw','Lingala': 'ln', 'Luganda': 'lg', 'Oromo': 'om', 
            'Shona': 'sn', 'Sotho': 'st', 'Wolof': 'wo',
            'Twi': 'tw' , 'Xhosa': 'xh','Yoruba': 'yo', 'Zulu': 'zu', "Aymara": "ay", "Guarani": "gn", "Quechua": "qu"}

class TranslationDataConfig(datasets.BuilderConfig):
    """BuilderConfig for TranslationData."""

    def __init__(self, config: str, **kwargs):
        """BuilderConfig for TranslationData.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TranslationDataConfig, self).__init__(**kwargs)
        self.lang = config

class TranslationData(datasets.GeneratorBasedBuilder):
    """This is an adapter for loading raw text parallel corpus."""
    BUILDER_CONFIG_CLASS = TranslationDataConfig

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "instruction": datasets.Value("string"),
                    "input": datasets.Value("string"),
                    "output": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        base_path = os.path.join(self.base_path, f"{self.config.lang}")
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join(base_path, "train")}),
            # datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(base_path, "test")}),
            # datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": os.path.join(base_path, "validation")}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        key = 0

        def read_dataset(path):
            if 'jsonl' in path:
                dataset = []
                with open(path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        dataset.append(json.loads(line))
            elif 'json' in path:
                with open(path, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
                if isinstance(dataset, dict):
                    if 'data' in dataset:
                        dataset = dataset['data']
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    dataset = f.readlines()
            return dataset

        data = []
        languages = [
                'French', 'Swahili', 'Amharic', 'Ewe', 'Hausa', 'Igbo',
                'Kinyarwanda', 'Lingala', 'Luganda', 'Oromo', 'Shona',
                'Sotho', 'Wolof', 'Twi', 'Xhosa', 'Yoruba', 'Zulu'
            ]
        for train_name in languages:
            train_name_map = langs_map[train_name]
            path_base = f'./data/african-bilingual-pairs/en-{train_name_map}'
            path_src = f'{path_base}/train_9k.{train_name_map}'
            path_trg = f'{path_base}/train_9k.en'
            sources = read_dataset(path_src)
            targets = read_dataset(path_trg)
            train_set = [(source, target) for source, target in zip(sources, targets)]
            for source, target in train_set:
                data.append({
                    'source': source,
                    'target': target,
                    'source_language': train_name,
                    'target_language': 'English'
                })

        for d in data:
            source_line, target_line = d['source'], d['target']
            yield key, {
                "id": key,
                "instruction":  _INSTRUCTIONS[0].format_map({
                                        "source_lang": d['source_language'], 
                                    }) + ' ' + source_line.strip(),
                "input": "",
                "output": target_line.strip(),
            }
            key += 1