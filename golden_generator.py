import csv
import json
import numpy as np
from deepeval.models import OllamaModel
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
import json
import re
from deepeval.models import OllamaModel
from deepeval.synthesizer import Synthesizer
from pygments.lexers.mosel import FUNCTIONS
import pandas as pd
from IPython.display import display, Markdown

class GoldenGenerator:
    def __init__(self, model, context_file, synthesizer):
        self.model = model
        self.context_file = context_file
        self.synthesizer = synthesizer

    def prepare_context(self, split_by, count):
        documents = []
        with open(self.context_file, "r", encoding="utf-8") as file:
            data = json.load(file)
        for article in data[:count]:
            abstract = article["abstract"].split(split_by)
            documents.append(abstract)

        return documents

    def synthesize_save_golden(self, documents=[]):
        if len(documents) == 0:
            documents = self.prepare_context(" ", 50)
        goldens = self.synthesizer.generate_goldens_from_contexts(
            contexts=documents,
            include_expected_output=True,
            max_goldens_per_context=1
        )
        self.save_goldens_as_files(goldens, "testtest")
        return goldens

    def save_goldens_as_files(self, goldens, filename):
        goldens_data = []
        count = 1
        for golden in goldens:
            goldens_data.append({
                "golden_id": count,
                "question": golden.input,
                "expected_answer": golden.expected_output,
                "context": golden.context
            })

        with open(f"filename".json, "w", encoding="utf-8") as f:
            json.dump(
                goldens_data,
                f,
                ensure_ascii=False,
                indent=4
            )
        print("Saved to golden_dataset_1.json")
        df = pd.DataFrame(goldens_data)

        df.to_csv(
            f"{filename}.csv",
            index=False,
            encoding="utf-8"
        )


