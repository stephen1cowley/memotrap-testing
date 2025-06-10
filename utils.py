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

def evaluate_nq_ans_recall(
        prediction: str,
        answers: List[str],
    ) -> bool:
    "Substring match score given a list of correct answers"
    norm_prediction = normalize_answer(prediction)
    norm_answers = [normalize_answer(ans) for ans in answers]

    for norm_answer in norm_answers:
        if norm_answer in norm_prediction:
            return True
    return False

def evaluate_nq_ans_em(
        prediction: str,
        answers: List[str],
    ) -> bool:
    "Exact match metric"
    norm_prediction = normalize_answer(prediction)
    norm_answers = [normalize_answer(ans) for ans in answers]

    return norm_prediction in norm_answers

def recall_score(prediction, ground_truth):
    "Substring match score given a list of correct answers"
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    return (ground_truth in prediction)

def get_score(preds, golds):
    em, recall = 0, 0
    for pred, gold in zip(preds, golds):
        if isinstance(gold, list):
            _em, _recall = 0, 0
            for g in gold:
                _em = max(exact_match_score(pred, g), _em)
                _recall = max(recall_score(pred, g), _recall)
            em += _em
            recall += _recall
        else:
            em += exact_match_score(pred, gold)
            recall += recall_score(pred, gold)
    em = em * 100 / (len(preds) + 1e-5)
    recall = recall * 100 / (len(preds) + 1e-5)
    return em, recall