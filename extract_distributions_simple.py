from hybrid_method import HybridMethod
import json
from utils import renormalize_pmf

NQ_DATAPATH = 'nq/orig_dev_filtered.json'
MODEL = '4bit/Llama-2-7b-chat-hf'

llm = HybridMethod(
    model_name=MODEL,
    device="cuda"
)

context = 'Write a quote that ends in the word ""happy""'
prompt = 'The customer is always'

all_distributions = llm.cad_dola_verbose_memotrap(
    context=context,
    prompt=prompt,
    token_ids_of_interest=[1492, 9796, 297],
    dola_layers_bad='high',
    dola_layers_good='high',
)

with open('all_distributions.json', 'w') as json_file:
    json.dump(all_distributions, json_file)
    print("Successfully finished the experiment")
    json_file.flush()
