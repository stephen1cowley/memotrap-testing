from typing import List, Literal, Any, Union, Dict
import ast
import pandas as pd
from hybrid_method import HybridMethod
import argparse

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
        dola_layers: Union[Literal["high", "low"], None],
        df: Any
    ) -> int:
    """
    Iterate through all memotrap questions and returns the CAD's score (max 860)
    """
    score = 0
    for _, row in df.iterrows():
        context: str = row['prompt'].split(":")[0]  # LHS of colon
        prompt: str = row['prompt'].split(":")[1][1:]  # RHS of colon
        answer_index: Literal[0, 1] = row['answer_index']
        correct_ans: str = unmarshall_list(row['classes'])[answer_index]

        cad_answer = llm.cad_generate(
            context=context,
            prompt=prompt,
            dola_layers=dola_layers,
            beta=beta
        )
        if cad_answer == None: RuntimeError("Error with llm generation"); return -1
        if cad_answer == correct_ans:
            score += 1
    print(f"RESULT: CAD with coefficient={beta}, dola set to {dola_layers}, model {llm.model_name}, we achieved a score of {score}/860")
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    
    df = pd.read_csv(MEMOTRAP_DATAPATH)
    llm = HybridMethod(
        model_name=args.model,
        device=args.device
    )

    betas: List[float] = [-1.0, -0.5, 0.0, 0.5, 1.0]
    results: Dict[str, int] = {}

    for beta in betas:
        score: int = evaluate_llm(
            llm=llm,
            beta=beta,
            dola_layers=None,
            df=df,
        )
        results[str(beta)] = score
