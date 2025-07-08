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
]

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

        filepath = "/home/bumie304/projects/def-annielee/bumie304/QAlign/data/metamath3k/math.json"
        with open(f"{filepath}", encoding="utf-8") as f:
            data = json.load(f)
            for d in data:
                source_line, target_line = d['query'], d['query_en']
                yield key, {
                    "id": key,
                    "instruction":  _INSTRUCTIONS[0].format_map({
                                            "source_lang": d['lang'], 
                                        }) + ' ' + source_line.strip(),
                    "input": "",
                    "output": target_line.strip(),
                }
                key += 1