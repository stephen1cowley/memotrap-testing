"""
Standard evaluation metrics scripts
"""

import re
import string
from typing import List, Dict

def renormalize_pmf(pmf: Dict[str, float]) -> Dict[str, float]:
        "Make sure a dict pmf has values that sum to 1.0"
        Z = sum([prob for prob in pmf.values()])
        norm_pmf: Dict[str, float] = {}
        for key in pmf:
            norm_pmf[key] = pmf[key] / Z
        return norm_pmf


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return (normalize_answer(prediction) == normalize_answer(ground_truth))    


def evaluate_nq_ans_em(
        prediction: str,
        answers: List[str],
    ) -> bool:
    "Exact match metric"
    norm_prediction = normalize_answer(prediction)
    norm_answers = [normalize_answer(ans) for ans in answers]

    return norm_prediction in norm_answers
