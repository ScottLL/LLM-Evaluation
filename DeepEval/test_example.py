import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import BaseMetric
import asyncio

def test_answer_relevancy():
    
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        # Replace this with the actual output of your LLM application
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."]
    )
# #     assert_test(test_case, [answer_relevancy_metric])
    
def test_coherence():
    coherence_metric = GEval(
    name="Coherence",
    criteria="Coherence - determine if the actual output is logical, has flow, and is easy to understand and follow.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5
    )
    test_case = LLMTestCase(
        input="What if these shoes don't fit? I want a full refund.",
        # Replace this with the actual output from your LLM application
        actual_output="If the shoes you've purchased don't fit and you're seeking a full refund, here are the general steps you should follow. These steps are standard practice for many retailers, but it's important to check the specific return policy of the store where you bought the shoes: 1. **Check the Return Policy**: Before proceeding, verify the return policy of the retailer. Look for any time restrictions on returns (e.g., must return within 30 days of purchase), the condition the item must be in (e.g., unworn, in original packaging), and whether you need the original receipt. 2. **Locate the Receipt**: Find your purchase receipt. A receipt is often required for a full refund, especially if you paid with cash or want the refund applied to your credit card. If you lost the receipt, you might still be able to return the product by providing other proof of purchase, such as a bank statement, or you may be offered store credit instead. 3. **Repackage the Shoes**: If possible, place the shoes back into their original box and ensure all original packaging materials are included. This can be important for the return to be accepted, especially if the store resells the item. 4. **Contact the Retailer**: If you bought the shoes online or if the physical store is not easily accessible, contact the retailer's customer service for guidance on how to proceed with the return. They may provide a prepaid shipping label for online returns. 5. **Visit the Store**: For purchases from physical stores, return the shoes to the store where you bought them. Bring the shoes, the original packaging, and the receipt. Speak to a customer service representative or a manager about your desire for a full refund due to the fit. 6. **Follow Online Return Instructions**: If you need to mail the shoes back, carefully follow the retailer's online return instructions. This often includes printing out a return label, securing the package, and dropping it off at a designated shipping location. 7. **Wait for the Refund**: Once the retailer receives the returned shoes, they will process your refund. The time it takes for the refund to appear in your account can vary depending on the retailer's policies and the method of payment used. Remember, return policies can vary widely between retailers, so it's crucial to familiarize yourself with the specific terms and conditions of the place where you purchased the shoes.",
    )
#     # assert_test(test_case, [coherence_metric])
    


class LengthMetric(BaseMetric):
    def __init__(self, max_length: int = 10):
        super().__init__()
        self._threshold = max_length
        self.success = False

    def measure(self, test_case: LLMTestCase) -> float:
        self.success = len(test_case.actual_output) > self._threshold
        if self.success:
            score = 1.0
        else:
            score = 0.0
        return score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        # Workaround to use synchronous measure in an async context
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value

    @property
    def __name__(self):
        return "Length"


def test_length():
    length_metric = LengthMetric(max_length=10)
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        # Replace this with the actual output of your LLM application
        actual_output="We offer a 30-day full refund at no extra cost."
    )
#     assert_test(test_case, [length_metric])


def test_everything():
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
    length_metric = LengthMetric(max_length=10)
    coherence_metric = GEval(
        name="Coherence",
        criteria="Coherence - determine if the actual output is logical, has flow, and is easy to understand and follow.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5
    )

    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        # Replace this with the actual output of your LLM application
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."]
    )
    assert_test(test_case, [answer_relevancy_metric, coherence_metric, length_metric])