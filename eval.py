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
from golden_generator import GoldenGenerator
from deepeval import evaluate
from deepeval.models import OllamaModel

judge = OllamaModel(
    model="gpt-oss:20b",
    base_url="http://localhost:11434",
    temperature=0
)
class Eval:
    def __init__(self, judge = judge):
        self.model = judge

    def answer_relevancy(self, df, threshold):
        metric = AnswerRelevancyMetric(
            model=self.model,
            async_mode=False,
            threshold=threshold,
            verbose_mode=False,
        )
        test_cases = []
        results = []
        for _, row in df.iterrows():
            res = {}
            test_case = LLMTestCase(
                input=row["question"],
                actual_output=row["answer"]
            )
            metric.measure(test_case)
            res["score"] = metric.score
            res["threshold"] = metric.threshold
            res["reason"] = metric.reason
            results.append(res)
        return results



