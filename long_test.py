"""
Evaluates CAD/Additive CAD on the MemoTrap dataset
"""

from typing import List, Literal, Any, Union, Dict
import ast
import pandas as pd
from hybrid_method import HybridMethod
import argparse
import json
import time

MEMOTRAP_DATAPATH = 'memotrap/1-proverb-ending.csv'

def unmarshall_list(data: str) -> List[str]:
    # Use ast.literal_eval to convert the string into a list
    try:
        return ast.literal_eval(data)
    except (ValueError, SyntaxError):
        # Handle cases where the data is not properly formatted
        return []


def evaluate_llm(
        llm: HybridMethod,
        beta: float,
        df: Any,
        dola_layers_good: Union[Literal['high', 'low'], None] = None,
        dola_layers_bad: Union[Literal['high', 'low'], None] = None,
    ) -> int:
    """
    Iterate through all memotrap questions and returns the CAD's score (max 860)
    """
    score = 0
    for idx, row in df.iterrows():
        context: str = row['prompt'].split(":")[0]  # LHS of colon
        prompt: str = row['prompt'].split(":")[1][1:]  # RHS of colon
        answer_index: Literal[0, 1] = row['answer_index']
        correct_ans: str = unmarshall_list(row['classes'])[answer_index]

        cad_answer = llm.cad_generate_memotrap(
            context=context,
            prompt=prompt,
            dola_layers_good=dola_layers_good,
            dola_layers_bad=dola_layers_bad,
            gamma=beta
        )

        if cad_answer == None: return -1  # Error with CAD generation
        cad_answer = cad_answer[len(context + ": " + prompt):]  # Get the bit after the context and prompt
        if idx % 100 == 0:
            print(f"{idx}. CAD answer: {repr(cad_answer)}")
            print(f"{idx}. Correct answer: {repr(correct_ans)}")
        if evaluate_ans(correct_ans, cad_answer):
            score += 1
    print(f"RESULT: CAD with coefficient={beta}, dola-good set to {dola_layers_good}, dola-bad set to {dola_layers_bad}, model {llm.model_name}, we achieved a score of {score}/860")
    return score


def evaluate_ans(
        correct_answer: str,
        given_answer: str
    ) -> bool:
    """
    Given that `correct_answer` is a string ending in a period (.), check if `given_answer`
    contains the `correct_answer` string. Thus we have ended in the correct word
    """
    return correct_answer in given_answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dola-layers-good", type=str, choices=["high", "low", "None"], default="None")
    parser.add_argument("--dola-layers-bad", type=str, choices=["high", "low", "None"], default="None")
    args = parser.parse_args()

    dola_layers_good: Union[Literal["high", "low"], None] = None if args.dola_layers_good == "None" else args.dola_layers_good
    dola_layers_bad: Union[Literal["high", "low"], None] = None if args.dola_layers_bad == "None" else args.dola_layers_bad

    time_1 = time.time()
    df = pd.read_csv(MEMOTRAP_DATAPATH)
    llm = HybridMethod(
        model_name=args.model,
        device=args.device
    )
    ex_time = time.time() - time_1
    print(f"Model load time: {ex_time:.4f}s")

    betas: List[float] = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    results: Dict[str, int] = {}

    for beta in betas:
        time_1 = time.time()
        score: int = evaluate_llm(
            llm=llm,
            beta=beta,
            df=df,
            dola_layers_good=dola_layers_good,
            dola_layers_bad=dola_layers_bad,
        )
        results[str(beta)] = score
        ex_time = time.time() - time_1
        print(f"Evaluation time for beta={beta}: {ex_time:.4f}s")

    print("Final results:", results)
    with open(f'cad_dola_{str(dola_layers_good)}_{str(dola_layers_bad)}.json', 'w') as json_file:
        json.dump(results, json_file)
        print("Successfully finished the experiment")
