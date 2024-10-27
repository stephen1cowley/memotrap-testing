from typing import List, Any, Union, Literal, Dict
from hybrid_method import HybridMethod
import argparse
import json
from utils import normalize_answer, evaluate_nq_ans
import time

NQ_DATAPATH = 'nq/orig_dev_filtered.json'
MODEL = '4bit/Llama-2-7b-chat-hf'

def evaluate_llm(
        llm: HybridMethod,
        beta: float,
        time_0: float,
        max_time: float,
        data: List[Any],
        dola_layers_good: Union[Literal['high', 'low'], None] = None,
        dola_layers_bad: Union[Literal['high', 'low'], None] = None,
    ) -> int:
    """
    Iterate through all memotrap questions and returns the CAD's score (max 860).
    Returns an integer score of exact matches (EMs).
    """
    score: int = 0
    print("Length of data", len(data))
    for idx, qa in enumerate(data):
        context: str = qa["context"]
        question: str = qa["question"]
        answers: List[str] = qa["answer"]

        cad_answer = llm.cad_generate_nq(
            context=context,
            question=question,
            beta=1.0,
            dola_layers_good=None,
            dola_layers_bad=None,
            max_tokens=20,
        )

        if cad_answer == None: return -1  # Error with CAD generation

        if time.time() - time_0 >= max_time:
            print("Time up!")
            return score

        if idx % 100 == 0:
            print(f"{idx}. CAD answer: {repr(normalize_answer(cad_answer))}")
            print(f"{idx}. Correct answers:", " ".join([repr(normalize_answer(answers[i])) for i in range(len(answers))]))
            print("Time:", time.time() - time_0)
        if evaluate_nq_ans(cad_answer, answers):
            score += 1
    print(f"RESULT: CAD with coefficient={beta}, dola-good set to {dola_layers_good}, dola-bad set to {dola_layers_bad}, model {llm.model_name}, we achieved a score of {score}/860")
    return score



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dola-layers-good", type=str, choices=["high", "low", "None"], default="None")
    parser.add_argument("--dola-layers-bad", type=str, choices=["high", "low", "None"], default="None")
    args = parser.parse_args()

    dola_layers_good: Union[Literal["high", "low"], None] = None if args.dola_layers_good == "None" else args.dola_layers_good
    dola_layers_bad: Union[Literal["high", "low"], None] = None if args.dola_layers_bad == "None" else args.dola_layers_bad

    
    with open(NQ_DATAPATH, 'r') as file:
        data: List[Any] = json.load(file)

    time_0 = time.time()

    llm = HybridMethod(
        model_name=args.model,
        device=args.device
    )
    ex_time = time.time() - time_0
    print(f"Model load time: {ex_time:.4f}s")

    betas: List[float] = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    results: Dict[str, int] = {}

    for beta in betas:
        time_1 = time.time()
        score: int = evaluate_llm(
            llm=llm,
            beta=beta,
            time_0=time_0,
            max_time=9*60,
            data=data,
            dola_layers_good=dola_layers_good,
            dola_layers_bad=dola_layers_bad,
        )
        results[str(beta)] = score
        ex_time = time.time() - time_1
        tot_time = time.time() - time_0
        print(f"Evaluation time for beta={beta}: {ex_time:.4f}s")
        if tot_time / (60 * 60) >= 7.0: # 9 mins
            print("7 hrs up")
            break

    print("Final results:", results)
    with open(f'nq_cad_dola_{str(dola_layers_good)}_{str(dola_layers_bad)}.json', 'w') as json_file:
        json.dump(results, json_file)
        print("Successfully finished the experiment")
