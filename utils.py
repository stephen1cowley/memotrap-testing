import re
import string
from typing import List

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

def evaluate_nq_ans(
        prediction: str,
        answers: List[str],
    ) -> bool:
    norm_prediction = normalize_answer(prediction)
    norm_answers = [normalize_answer(ans) for ans in answers]
    return norm_prediction in norm_answers

def recall_score(prediction, ground_truth):
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