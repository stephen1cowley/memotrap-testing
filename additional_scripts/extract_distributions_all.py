"""
Extracts the token probability distributions of the memotrap dataset for Simplex visualisation
"""

from hybrid_method import HybridMethod
import json
from utils import renormalize_pmf
import pandas as pd
from typing import List, Literal
import ast

NQ_DATAPATH = 'nq/orig_dev_filtered.json'
# MODEL = '4bit/Llama-2-7b-chat-hf'
MODEL = "huggyllama/llama-7b"
MEMOTRAP_DATAPATH = 'memotrap/1-proverb-ending.csv'

def unmarshall_list(data: str) -> List[str]:
    # Use ast.literal_eval to convert the string into a list
    try:
        return ast.literal_eval(data)
    except (ValueError, SyntaxError):
        # Handle cases where the data is not properly formatted
        return []

if __name__ == "__main__":
    with open('qs_of_interest.json', 'r') as file:
        qs_of_interest: List[int] = json.load(file)

    final_data = []

    llm = HybridMethod(
        model_name=MODEL,
        device="cuda"
    )

    df = pd.read_csv(MEMOTRAP_DATAPATH)

    for idx, row in df.iterrows():
        if idx not in qs_of_interest:
            continue
        
        context: str = row['prompt'].split(":")[0]  # LHS of colon
        prompt: str = row['prompt'].split(":")[1][1:]  # RHS of colon
        answer_index: Literal[0, 1] = row['answer_index']
        correct_ans: str = unmarshall_list(row['classes'])[answer_index]

        candidate_3 = llm.determine_candidate_3(
            context=context,
            prompt=prompt,
        )

        print("Question", idx)
        print(candidate_3)
        print(llm.tokenizer.decode(candidate_3))

        all_distributions = llm.cad_dola_verbose_memotrap(
            context=context,
            prompt=prompt,
            token_ids_of_interest=candidate_3,
            dola_layers_bad='high',
            dola_layers_good='high',
        )

        final_data.append({
            "num": idx,
            "question": prompt,
            "all_distributions": all_distributions
        })

    with open('final_data_all_distributions.json', 'w') as json_file:
        json.dump(final_data, json_file)
        print("Successfully finished the experiment")
        json_file.flush()
