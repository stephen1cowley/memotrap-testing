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

    qa = data[-5]

    # if idx == 3:
    #     print("Completed the test 3 cycles, now finishing")
    #     break

    context: str = qa["context"]
    question: str = qa["question"]
    answers: List[str] = qa["answer"]

    cad_answer = llm.cad_generate_nq(
        context=context,
        question=question,
        beta=1.0,
        dola_layers_good=None,
        dola_layers_bad=None,
        max_tokens=5,
    )

    reg_answer = llm.cad_generate_nq(
        context=context,
        question=question,
        beta=0.0,
        dola_layers_good=None,
        dola_layers_bad=None,
        max_tokens=5,
    )

    if isinstance(cad_answer, str) and isinstance(reg_answer, str):
        print("Question: ", question)
        print("Answers:", repr(answers))
        print("Context:", context)
        print("Reg. Given answer:", repr(reg_answer))
        print("CAD Given answer:", repr(cad_answer))
    else:
        RuntimeError("Strings not returned by LLMs")
