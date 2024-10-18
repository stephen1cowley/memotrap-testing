from typing import List, Literal
import ast
import pandas as pd
import torch
from hybrid_method import HybridMethod
import argparse
import os

MEMOTRAP_DATAPATH = 'memotrap/1-proverb-ending.csv'
MODEL = '4bit/Llama-2-7b-chat-hf'

def unmarshall_list(data: str) -> List[str]:
    # Use ast.literal_eval to convert the string into a list
    try:
        return ast.literal_eval(data)
    except (ValueError, SyntaxError):
        # Handle cases where the data is not properly formatted
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--rds", action="store_true")
    args = parser.parse_args()
    device = args.device

    if args.rds:
        os.environ['TRANSFORMERS_CACHE'] = "/rds/user/ssc42/hpc-work"

    df = pd.read_csv(MEMOTRAP_DATAPATH)

    llm = HybridMethod(
        model_name=MODEL,
        device=device
    )

    with torch.no_grad():
        for idx, row in df.iterrows():
            if idx == 3:
                print("Completed the test 5 cycles, now finishing")
                break

            context: str = row['prompt'].split(":")[0]
            prompt: str = row['prompt'].split(":")[1][1:]
            classes: List[str] = unmarshall_list(row['classes'])
            answer_index: Literal[0, 1] = row['answer_index']

            cad_answer = llm.cad_generate(
                context=context,
                prompt=prompt,
                dola_layers=None
            )

            reg_answer = llm.generate(
                input_text=context + ": " + prompt,
                dola_layers=None
            )

            if isinstance(cad_answer, str) and isinstance(reg_answer, str):
                cad_answer = cad_answer[len(context + ": " + prompt):]
                reg_answer = reg_answer[len(context + ": " + prompt):]

                print("Question:", context + ": " + prompt)
                print("Correct answer:", repr(classes[answer_index]))
                print("Reg. Given answer:", repr(reg_answer), "(Reg. SUCCESS)\n" if reg_answer == classes[answer_index] else "(Reg. FAIL)")
                print("CAD Given answer:", repr(cad_answer), "(CAD SUCCESS)\n" if cad_answer == classes[answer_index] else "(CAD FAIL)\n")
