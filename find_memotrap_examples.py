from typing import List, Literal, Any, Union, Dict
import ast
import pandas as pd
from hybrid_method import HybridMethod
import argparse
import json
import time
import torch

MEMOTRAP_DATAPATH = 'memotrap/1-proverb-ending.csv'

def unmarshall_list(data: str) -> List[str]:
    # Use ast.literal_eval to convert the string into a list
    try:
        return ast.literal_eval(data)
    except (ValueError, SyntaxError):
        # Handle cases where the data is not properly formatted
        return []


def evaluate_both_llms(
        llm: HybridMethod,
        beta: float,
        df: Any,
    ) -> List[int]:
    """
    Iterate through all memotrap questions and returns the CAD's score (max 860)
    """
    questions_of_interest: List[int] = []
    for idx, row in df.iterrows():
        context: str = row['prompt'].split(":")[0]  # LHS of colon
        prompt: str = row['prompt'].split(":")[1][1:]  # RHS of colon
        answer_index: Literal[0, 1] = row['answer_index']
        correct_ans: str = unmarshall_list(row['classes'])[answer_index]

        cad_answer = llm.cad_generate_memotrap(
            context=context,
            prompt=prompt,
            dola_layers_good=None,
            dola_layers_bad=None,
            beta=beta
        )

        reg_answer = llm.cad_generate_memotrap(
            context=context,
            prompt=prompt,
            dola_layers_good=None,
            dola_layers_bad=None,
            beta=0.0
        )

        dola_cad_answer = llm.cad_generate_memotrap(
            context=context,
            prompt=prompt,
            dola_layers_good='high',
            dola_layers_bad='high',
            beta=beta
        )

        if cad_answer is None or reg_answer is None or dola_cad_answer is None:
            # Error with CAD generation
            print("Error on question", idx)
            continue  

        # Now extract the examples of interest:
        # regular incorrect, cad correct, cad-dola incorrect
        if not evaluate_ans(correct_ans, reg_answer) and evaluate_ans(correct_ans, cad_answer) and not evaluate_ans(correct_ans, dola_cad_answer):
            print(f"Q{idx}. Reg. answer: {repr(reg_answer)}")
            print(f"Q{idx}. CAD answer: {repr(cad_answer)}")
            print(f"Q{idx}. CAD-DoLa answer: {repr(reg_answer)}")
            print(f"Q{idx}. Correct answer: {repr(correct_ans)}")

            questions_of_interest.append(idx)

    return questions_of_interest


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
    args = parser.parse_args()

    beta: float = 1.0

    time_1 = time.time()
    df = pd.read_csv(MEMOTRAP_DATAPATH)
    llm = HybridMethod(
        model_name=args.model,
        device=args.device
    )
    ex_time = time.time() - time_1
    print(f"Model load time: {ex_time:.4f}s")

    questions_of_interest: List[int] = evaluate_both_llms(
        llm=llm,
        beta=beta,
        df=df,
    )
    print("Final questions of interest:", questions_of_interest)

    with open(f'qs_of_interest.json', 'w') as json_file:
        json.dump(questions_of_interest, json_file)
        print("Successfully finished the experiment")
