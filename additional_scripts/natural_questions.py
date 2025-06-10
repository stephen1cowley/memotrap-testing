from typing import List, Any
from hybrid_method import HybridMethod
import argparse
import json
from utils import normalize_answer

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
        print("Normalized Answers:", " ".join([repr(normalize_answer(answers[i])) for i in range(len(answers))]))
        print("Context:", context)
        print("Reg. Normalized Given answer:", repr(normalize_answer(reg_answer)))
        print("CAD Normalized Given answer:", repr(normalize_answer(cad_answer)))
    else:
        RuntimeError("Strings not returned by LLMs")
