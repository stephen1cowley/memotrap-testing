from typing import List, Literal, Any
import ast
import pandas as pd
import torch
from hybrid_method import HybridMethod
import argparse
import json

NQ_DATAPATH = 'nq/orig_dev_filtered.json'
MODEL = '4bit/Llama-2-7b-chat-hf'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    device = args.device
    
    with open(NQ_DATAPATH, 'r') as file:
        data: List[Any] = json.load(file)

    llm = HybridMethod(
        model_name=MODEL,
        device=device
    )

    for idx, qa in data:
        if idx == 3:
            print("Completed the test 3 cycles, now finishing")
            break

        context: str = qa["context"]
        prompt: str = qa["question"]
        answers: List[str] = qa["answer"]

        cad_answer = llm.cad_generate(
            context=context,
            prompt=prompt,
            dola_layers_good=None,
            dola_layers_bad=None,
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
