"""
Main class for evaluating CAD and Additive CAD on the MemoTrap and Natural Questions benchmarks
"""

from typing import Literal, Any, Tuple, List, Union, Dict, Optional
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizer, PreTrainedModel, StoppingCriteria, StoppingCriteriaList
from transformers.generation.utils import ModelOutput


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
        if not isinstance(outputs, ModelOutput): return None
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
        if not isinstance(outputs, ModelOutput): return None
        if isinstance(outputs.scores, Tuple):
            if isinstance(outputs.scores[0], torch.Tensor):
                return outputs.scores[0]
        return None

    def generate_1_final_dis(
            self,
            input_text: str,
            dola_layers: Union[Literal['high', 'low'], None] = None,
        ) -> Any:
        """
        Like `generate_1()` but for outputting the actual probability distributions at the final layer
        """
        inputs: Any = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            output_scores=True,
            return_dict_in_generate=True,
            output_hidden_states=True,
            dola_layers=dola_layers,
            max_new_tokens=1,
            min_new_tokens=1,
            stopping_criteria=StoppingCriteriaList([self.stop_on_period])
        )
        if not isinstance(outputs, ModelOutput): return None

        return outputs.scores[0]  # type: ignore

    def generate_1_each_dis(
            self,
            input_text: str,
            dola_layers: Union[Literal['high', 'low'], None] = None,
        ) -> Any:
        """
        Like `generate_1()` but for outputting the actual probability distributions at each layer
        """
        inputs: Any = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            output_scores=True,
            return_dict_in_generate=True,
            output_hidden_states=True,
            dola_layers=dola_layers,
            max_new_tokens=1,
            min_new_tokens=1,
            stopping_criteria=StoppingCriteriaList([self.stop_on_period])
        )
        if not isinstance(outputs, ModelOutput): return None

        # Extract hidden states at each layer
        hidden_states = outputs.hidden_states[0] #type: ignore

        with torch.no_grad():
            # Calculate logits at each layer
            layer_logits = []
            for hidden_state in hidden_states: #type: ignore
                logits = self.model.lm_head(hidden_state)
                layer_logits.append(logits[:, -1, :])  # Shape: [batch_size, sequence_length, vocab_size]

        # print(layer_logits)
        # return self.logits_to_pmf(outputs.scores[0]) #type: ignore
        return [dis for dis in layer_logits] #type: ignore

    def logits_to_pmf(
            self,
            distribution: torch.Tensor,
            token_ids_of_interest: Optional[List[int]] = None,
        ) -> Dict[str, float]:
        """
        Takes the tensor distribution and returns something a bit more readable.
        """

        pmf: Dict[str, float] = {}
        probs = torch.softmax(distribution, dim=-1)

        if token_ids_of_interest:
            plausible_ids = token_ids_of_interest
        else:
            _, plausible_ids = torch.topk(distribution, 10, dim=-1)
            plausible_ids = plausible_ids[0]

        for id in plausible_ids:
            token: str = self.tokenizer.decode(id)
            pmf[token] = float(probs[0, id])

        return self.renormalize_pmf(pmf)

    def cad_dola_verbose_memotrap(
            self,
            context: str,
            prompt: str,
            token_ids_of_interest: Union[List[int], None] = None,
            dola_layers_good: Union[Literal['high', 'low'], None] = None,
            dola_layers_bad: Union[Literal['high', 'low'], None] = None,
            alpha: float = 0.1,
            betas: List[float] = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        ) -> Any:
        "Return the probability distributions of interest for the next token"

        each_dis_with_context = self.generate_1_each_dis(
            input_text=context + ': ' + prompt,
            dola_layers=None
        )
        print("Extracted each_dis_with_context")

        each_dis_no_context = self.generate_1_each_dis(
            input_text=prompt,
            dola_layers=None
        )
        print("Extracted each_dis_no_context")

        dis_no_context = self.generate_1_final_dis(
            input_text=prompt,
            dola_layers=None
        )

        dis_with_context = self.generate_1_final_dis(
            input_text=context + ': ' + prompt,
            dola_layers=None
        )

        each_cad_dis: Dict[str, Any] = {}

        for beta in betas:
            cad_dis = self.contrastive_decoding_verb(
                bad_distribution=dis_no_context,
                good_distribution=dis_with_context,
                alpha=alpha,
                beta=beta,
            )
            each_cad_dis[str(beta)] = self.logits_to_pmf(cad_dis, token_ids_of_interest=token_ids_of_interest)

        print("Extracted prob dis of each CAD")

        dola_dis_no_context = self.generate_1_final_dis(
            input_text=prompt,
            dola_layers=dola_layers_bad
        )

        dola_dis_with_context = self.generate_1_final_dis(
            input_text=context + ': ' + prompt,
            dola_layers=dola_layers_good
        )

        dola_each_cad_dis: Dict[str, Any] = {}

        for beta in betas:
            dola_cad_dis = self.contrastive_decoding_verb(
                bad_distribution=dola_dis_no_context,
                good_distribution=dola_dis_with_context,
                alpha=alpha,
                beta=beta,
            )
            dola_each_cad_dis[str(beta)] = self.logits_to_pmf(dola_cad_dis, token_ids_of_interest=token_ids_of_interest)

        print("Extracted DoLa prob dis of each CAD")

        return {
            "each_dis_with_context": [self.logits_to_pmf(dis, token_ids_of_interest=token_ids_of_interest) for dis in each_dis_with_context],
            "dis_with_context": self.logits_to_pmf(dis_with_context, token_ids_of_interest=token_ids_of_interest),
            "each_dis_no_context": [self.logits_to_pmf(dis, token_ids_of_interest=token_ids_of_interest) for dis in each_dis_no_context],
            "dis_no_context": self.logits_to_pmf(dis_no_context, token_ids_of_interest=token_ids_of_interest),
            "each_cad_dis": each_cad_dis,
            "dola_dis_with_context": self.logits_to_pmf(dola_dis_with_context, token_ids_of_interest=token_ids_of_interest),
            "dola_dis_no_context": self.logits_to_pmf(dola_dis_no_context, token_ids_of_interest=token_ids_of_interest),
            "dola_each_cad_dis": dola_each_cad_dis,
        }

    def renormalize_pmf(self, pmf: Dict[str, float]) -> Dict[str, float]:
        "Make sure a dict pmf has values that sum to 1.0"
        Z = sum([prob for prob in pmf.values()])
        norm_pmf: Dict[str, float] = {}
        for key in pmf:
            norm_pmf[key] = pmf[key] / Z if Z !=0 else 1 / len(pmf)
        return norm_pmf

    def determine_candidate_3(
            self,
            context: str,
            prompt: str,
        ) -> List[Any]:
        """
        For visualisation on the probability simplex, we want to determine which are the best top 3 token
        ids that we're interested in. The first is the correct answer with CAD of 1.0. The second is the
        incorrect answer with CAD of 0.0.
        """
        good_dis = self.generate_1_final_dis(
            input_text=context + ": " + prompt,
        )
        bad_dis = self.generate_1_final_dis(
            input_text=prompt,
        )
        cad_dis = self.contrastive_decoding_verb(
            good_distribution=good_dis,
            bad_distribution=bad_dis
        )
        dola_good_dis = self.generate_1_final_dis(
            input_text=context + ": " + prompt,
            dola_layers='high'
        )
        dola_bad_dis = self.generate_1_final_dis(
            input_text=prompt,
            dola_layers='high'
        )
        dola_cad_dis = self.contrastive_decoding_verb(
            good_distribution=dola_good_dis,
            bad_distribution=dola_bad_dis
        )

        _, bad_ids = torch.topk(bad_dis, 3, dim=-1)
        bad_ids = bad_ids.flatten().tolist()
        best_cad_vals, best_cad_id = torch.topk(cad_dis, 2, dim=-1)
        best_cad_ids = best_cad_id.flatten().tolist()
        best_cad_vals = best_cad_vals.flatten().tolist()

        best_dola_cad_vals, best_dola_cad_id = torch.topk(dola_cad_dis, 2, dim=-1)
        best_dola_cad_id = best_dola_cad_id.flatten().tolist()
        best_dola_cad_vals = best_dola_cad_vals.flatten().tolist()

        if best_cad_vals[1] > 0 and best_dola_cad_id[0] not in best_cad_ids:  # Check not -inf
            return best_cad_ids + [best_dola_cad_id[0]] 
        elif best_dola_cad_vals[1] > 0 and best_cad_ids[0] not in best_dola_cad_id: # Check not -inf
            return [best_cad_ids[0]] + best_dola_cad_id
        elif bad_ids[1] not in [best_cad_ids[0], best_dola_cad_id[0]]:
            return [best_cad_ids[0]] + [best_dola_cad_id[0]] + [bad_ids[1]]
        else:
            return [best_cad_ids[0]] + [best_dola_cad_id[0]] + [bad_ids[2]]

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
        # good and bad distributions are of shape (1, 32000)
        bad_distribution = torch.where(bad_distribution == float('-inf'), torch.tensor(-1000.0), bad_distribution)
        bad_distribution = torch.where(bad_distribution == float('inf'), torch.tensor(1000.0), bad_distribution)
        good_distribution = torch.where(good_distribution == float('-inf'), torch.tensor(-1000.0), good_distribution)
        good_distribution = torch.where(good_distribution == float('inf'), torch.tensor(1000.0), good_distribution)

        good_probs = torch.softmax(good_distribution, dim=-1)
        thresh = alpha * float(torch.max(good_probs).item())
        plausible_ids = (good_probs > thresh).nonzero(as_tuple=True)[-1]

        max_logit = float('-inf')
        can_id = None

        # go through all plausible id logits, and find the maximum post-contrasting
        for id in plausible_ids:
            id = int(id)
            logit = (1 + beta) * good_distribution[0, id] - beta * bad_distribution[0, id]
            if logit > max_logit:
                max_logit = logit
                can_id = id
        if can_id is not None:
            return can_id
        return -1

    def contrastive_decoding_novel(
            self,
            bad_distribution: torch.Tensor,
            good_distribution: torch.Tensor,
            gamma: float = 0.0,
        ) -> int:
        """
        Take 2 distributions, do contrastive decoding with adaptive plausibility constraint
        then return the token id with highest logit. Alpha and beta default to literature values.
        """
        # Replace -inf with -1000 and inf with 1000
        # good and bad distributions are of shape (1, 32000)
        bad_distribution = torch.where(bad_distribution == float('-inf'), torch.tensor(-1000.0), bad_distribution)
        bad_distribution = torch.where(bad_distribution == float('inf'), torch.tensor(1000.0), bad_distribution)
        good_distribution = torch.where(good_distribution == float('-inf'), torch.tensor(-1000.0), good_distribution)
        good_distribution = torch.where(good_distribution == float('inf'), torch.tensor(1000.0), good_distribution)

        good_probs = torch.softmax(good_distribution, dim=-1)
        bad_probs = torch.softmax(bad_distribution, dim=-1)

        new_probs = bad_probs + (good_probs - bad_probs) * (10**gamma)
        max_index = torch.argmax(new_probs).item()

        if max_index is not None:
            return int(max_index)
        return -1

    def contrastive_decoding_verb(
            self,
            bad_distribution: torch.Tensor,
            good_distribution: torch.Tensor,
            alpha: float = 0.1,
            beta: float = 1.0,
        ) -> torch.Tensor:
        """
        Take 2 distributions, do contrastive decoding with adaptive plausibility constraint
        then return the output distribution. Alpha and beta default to literature values.
        """
        # Replace -inf with -1000 and inf with 1000
        bad_distribution = torch.where(bad_distribution == float('-inf'), torch.tensor(-1000.0), bad_distribution)
        bad_distribution = torch.where(bad_distribution == float('inf'), torch.tensor(1000.0), bad_distribution)
        good_distribution = torch.where(good_distribution == float('-inf'), torch.tensor(-1000.0), good_distribution)
        good_distribution = torch.where(good_distribution == float('inf'), torch.tensor(1000.0), good_distribution)

        good_probs = torch.softmax(good_distribution, dim=-1)
        thresh = alpha * float(torch.max(good_probs).item())
        plausible_ids = (good_probs > thresh).nonzero(as_tuple=True)[-1]

        out_dis = torch.Tensor([[float('-inf')] * 32000])  # Initially assume every token has probability 0

        for id in plausible_ids:
            id = int(id)
            logit = (1 + beta) * good_distribution[0, id] - beta * bad_distribution[0, id]
            out_dis[0, id] = logit
        return out_dis.to(self.device)

    def cad_generate_memotrap(
            self,
            context: str,
            prompt: str,
            dola_layers_good: Union[Literal['high', 'low'], None] = None,
            dola_layers_bad: Union[Literal['high', 'low'], None] = None,
            alpha: float = 0.1,
            beta: float = 1.0,
            gamma: Union[float, None] = None,
            max_tokens: int = 20,
        ) -> Union[str, None]:
        """
        Given an input context and prompt, return the CAD-generated response
        """

        for _ in range(max_tokens):
            good_dis = self.generate_1(
                input_text=context + ": " + prompt,
                dola_layers=dola_layers_good
            )
            bad_dis = self.generate_1(
                input_text=prompt,
                dola_layers=dola_layers_bad
            )
            if good_dis is not None and bad_dis is not None:
                if gamma is None:
                    next_token_id = self.contrastive_decoding(
                        bad_distribution=bad_dis,
                        good_distribution=good_dis,
                        alpha=alpha,
                        beta=beta,
                    )
                elif gamma is not None:
                    next_token_id = self.contrastive_decoding_novel(
                        bad_distribution=bad_dis,
                        good_distribution=good_dis,
                        gamma=gamma
                    )
                if next_token_id == -1:
                    raise TypeError("contrastive_decoding failed to return correct id")
                prompt = self.tokenizer.decode(self.tokenizer.encode(prompt) + [next_token_id], skip_special_tokens=True)

                if self.tokenizer.decode(next_token_id) == ".":
                    break  # Stop generating after the sentence is ended
            else: raise TypeError("generate_1 failed to return correct logits")  
        return context + ": " + prompt  # Assuming the space was taken out before context and prompt passed in

    def cad_generate_nq(
            self,
            context: str,
            question: str,
            dola_layers_good: Union[Literal['high', 'low'], None] = None,
            dola_layers_bad: Union[Literal['high', 'low'], None] = None,
            alpha: float = 0.1,
            beta: float = 1.0,
            gamma: Union[float, None] = None,
            max_tokens: int = 20,
            prompt_id: str = "1",
        ) -> Union[str, None]:
        """
        Given an input context and prompt, return the CAD-generated response
        """
        if prompt_id == "1":
            sys_prompt_context: str = "Instruction: read the given information and answer the corresponding question.\n\n"
            sys_prompt_no_context: str = "Instruction: answer the corresponding question.\n\n"
        elif prompt_id == "2":  # Original CAD paper
            sys_prompt_context: str = ""
            sys_prompt_no_context: str = ""
        elif prompt_id == "3":  # COIECD paper
            sys_prompt_context: str = "Given the following information: "
            sys_prompt_no_context: str = "Answer the following question based on your internal knowledge with one or few words: "
        elif prompt_id == "4":  # AdaCAD paper
            sys_prompt_context: str = ""
            sys_prompt_no_context: str = "Answer the following question: Question: "
        else:
            raise ValueError("Invalid prompt id")
        output: str = " "

        for _ in range(max_tokens):
            if prompt_id == "1":
                good_dis = self.generate_1(
                    input_text=sys_prompt_context + context + "\nQ: " + question + "\nA:" + output,
                    dola_layers=dola_layers_good,
                )
                bad_dis = self.generate_1(
                    input_text=sys_prompt_no_context + "Q: " + question + "\nA:" + output,
                    dola_layers=dola_layers_bad,
                )
            elif prompt_id == "2":  # Original CAD paper
                good_dis = self.generate_1(
                    input_text=sys_prompt_context + context + " " + question + output,
                    dola_layers=dola_layers_good,
                )
                bad_dis = self.generate_1(
                    input_text=sys_prompt_no_context + question + output,
                    dola_layers=dola_layers_bad,
                )
            elif prompt_id == "3":  # COIECD paper
                good_dis = self.generate_1(
                    input_text=sys_prompt_context + context + "\nAnswer the following question based on the given information with one or few words: " + question + "\nAnswer:" + output,
                    dola_layers=dola_layers_good,
                )
                bad_dis = self.generate_1(
                    input_text=sys_prompt_no_context + question + "\nAnswer:" + output,
                    dola_layers=dola_layers_bad,
                )
            elif prompt_id == "4":  # AdaCAD paper
                good_dis = self.generate_1(
                    input_text=sys_prompt_context + context + "\nUsing only the references listed above, answer the following question:\nQuestion: " + question + ".\nAnswer:\n" + output,
                    dola_layers=dola_layers_good,
                )
                bad_dis = self.generate_1(
                    input_text=sys_prompt_no_context + question + ". Answer:\n" + output,
                    dola_layers=dola_layers_bad,
                )
            else:
                raise ValueError("Invalid prompt id")

            if good_dis is not None and bad_dis is not None:
                if gamma is None:
                    next_token_id = self.contrastive_decoding(
                        bad_distribution=bad_dis,
                        good_distribution=good_dis,
                        alpha=alpha,
                        beta=beta,
                    )
                elif gamma is not None:
                    next_token_id = self.contrastive_decoding_novel(
                        bad_distribution=bad_dis,
                        good_distribution=good_dis,
                        gamma=gamma,
                    )
                if next_token_id == -1:
                    raise TypeError("contrastive_decoding failed to return correct id")

                # NOTE: the following is a non-standard way of decoding (we decode/encode a running string)
                # This is less efficient than keeping everything as tokens, but is more intuitive
                # This approach was kept constant across all experiments
                output = self.tokenizer.decode(self.tokenizer.encode(output) + [next_token_id], skip_special_tokens=True)

                stopping_symbols = [".", "\n"]
                for stopping_symbol in stopping_symbols:
                    if stopping_symbol in output:
                        return output
            else:
                return None
        return output  # Assuming the space was taken out before context and prompt passed in

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
