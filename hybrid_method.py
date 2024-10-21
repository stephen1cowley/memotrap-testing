from typing import Literal, Any, Tuple, List, Union
import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizer, PreTrainedModel, StoppingCriteria, StoppingCriteriaList
from transformers.generation.utils import GenerateOutput

from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnPeriod(StoppingCriteria):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.stop_token = tokenizer.encode(text=".", add_special_tokens=False)[-1]
        self.tokenizer = tokenizer
    
    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        return self.tokenizer.decode(self.stop_token) in self.tokenizer.decode(input_ids[0])

class HybridMethod:
    """
    Sets up an LLM that we can do inference on.
    """
    def __init__(self, model_name: str, device: Literal['cpu', 'cuda']):
        self.model_name: str = model_name
        self.device: Literal['cpu', 'cuda'] = device
        self.tokenizer: PreTrainedTokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        self.model: PreTrainedModel = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            torch_dtype=torch.float16,
            device_map='auto',
        )
        self.stop_on_period = StopOnPeriod(self.tokenizer)

    def generate(
            self,
            input_text: str,
            dola_layers: Union[Literal['high', 'low'], None] = None,
        ) -> Union[str, None]:
        """
        Generate either with DoLa, depending on whether `dola_layers` is set to the higher or lower
        layer setting. DoLa is turned off if `dola_layers=None`.
        """
        inputs: Any = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            output_scores=False,
            return_dict_in_generate=True,
            output_hidden_states=False,
            dola_layers=dola_layers,
            max_new_tokens=4,
            min_new_tokens=1,
            stopping_criteria=StoppingCriteriaList([self.stop_on_period])
        )

        return self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    def generate_1(
            self,
            input_text: str,
            dola_layers: Union[Literal['high', 'low'], None] = None,
        ) -> Union[torch.FloatTensor, None]:
        """
        Generate a single token either with DoLa, depending on whether `dola_layers` is set to the higher or lower
        layer setting. DoLa is turned off if `dola_layers=None`.
        Returns the logits of the token.
        """
        inputs: Any = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            output_scores=True,
            return_dict_in_generate=True,
            output_hidden_states=False,
            dola_layers=dola_layers,
            max_new_tokens=1,
            min_new_tokens=1,
            stopping_criteria=StoppingCriteriaList([self.stop_on_period])
        )

        
        if isinstance(outputs.scores, Tuple):
            if isinstance(outputs.scores[0], torch.Tensor):
                return outputs.scores[0]
        return None
    
    def contrastive_decoding(
            self,
            bad_distribution: torch.Tensor,
            good_distribution: torch.Tensor,
            alpha: float = 0.1,
            beta: float = 1.0,
        ) -> int:
        """
        Take 2 distributions, do contrastive decoding with adaptive plausibility constraint
        then return the token id with highest logit. Alpha and beta default to literature values.
        """
        # Replace -inf with -1000 and inf with 1000
        bad_distribution = torch.where(bad_distribution == float('-inf'), torch.tensor(-1000.0), bad_distribution)
        bad_distribution = torch.where(bad_distribution == float('inf'), torch.tensor(1000.0), bad_distribution)
        good_distribution = torch.where(good_distribution == float('-inf'), torch.tensor(-1000.0), good_distribution)
        good_distribution = torch.where(good_distribution == float('inf'), torch.tensor(1000.0), good_distribution)

        good_probs = torch.softmax(good_distribution, dim=-1)
        thresh = alpha * float(torch.max(good_probs).item())
        plausible_ids = (good_probs > thresh).nonzero(as_tuple=True)[-1]

        max_logit = float('-inf')
        can_id = None

        for id in plausible_ids:
            id = int(id)
            logit = (1 + beta) * good_distribution[0, id] - beta * bad_distribution[0, id]
            if logit > max_logit:
                max_logit = logit
                can_id = id
        if not can_id is None:
            return can_id
        return -1

    def cad_generate(
            self,
            context: str,
            prompt: str,
            dola_layers: Union[Literal['high', 'low'], None] = None,
            alpha: float = 0.1,
            beta: float = 1.0,
            max_tokens: int = 20,
        ) -> Union[str, None]:
        """
        Given an input context and prompt, return the CAD-generated response
        """

        for _ in range(max_tokens):
            good_dis = self.generate_1(context + ": " + prompt)
            bad_dis = self.generate_1(prompt)
            if good_dis is not None and bad_dis is not None:
                next_token_id = self.contrastive_decoding(
                    bad_distribution=bad_dis,
                    good_distribution=good_dis,
                    alpha=alpha,
                    beta=beta,
                )
                if next_token_id == -1:
                    raise TypeError("contrastive_decoding failed to return correct id")
                prompt = self.tokenizer.decode(self.tokenizer.encode(prompt) + [next_token_id], skip_special_tokens=True)

                if self.tokenizer.decode(next_token_id) == ".":
                    break  # Stop generating after the sentence is ended
            else: raise  
        return context + ": " + prompt  # Assuming the space was taken out before context and prompt passed in

    def log_probs(
            self,
            prompt: str,
            answer: str
        ) -> float:
        """
        Returns log probs of the answer bit
        """
        input_text = prompt + answer
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        prefix_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        continue_ids = input_ids[0, prefix_ids.shape[-1]:]

        outputs = self.model(input_ids)[0].squeeze(0)
        outputs = outputs.log_softmax(-1)  # logits to log probs

        # skip tokens in the prompt -- we only care about the answer
        outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]

        # get logprobs for each token in the answer
        log_probs = outputs[range(outputs.shape[0]), continue_ids].mean().item()
        return log_probs
