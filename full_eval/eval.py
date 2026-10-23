from deepeval.metrics import AnswerRelevancyMetric
from pygments.lexers.mosel import FUNCTIONS
from deepeval.models import OllamaModel
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
import os

judge = OllamaModel(
    model="gpt-oss:20b",
    base_url="http://localhost:11434",
    temperature=0
)
class Eval:
    def __init__(self, judge = judge):
        self.model = judge

    def answer_relevancy(self, df, threshold):
        os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "300"
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

    def faithfullness(self, df, threshold):
        os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "380"
        results = []
        faithfulness = FaithfulnessMetric(
            threshold=0.5,
            model=judge,
            include_reason=True,
            async_mode=False
        )
        for _, row, in df.iterrows():
            test_case = LLMTestCase(
                input=row["question"],
                actual_output=str(row["extracted_answers"]),
                retrieval_context=row["extracted_text"]
            )

            faithfulness.measure(test_case)
            results.append({
                "score": faithfulness.score,
                "threshold": faithfulness.threshold,
                "reason": faithfulness.reason,
                "question": row["question"]
            })
        return results


