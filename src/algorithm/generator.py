import numpy as np
import logging
import torch
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

class BasicGenerator:
    def __init__(self, model_name_or_path):
        logger.info(f"Loading model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model_config = AutoConfig.from_pretrained(model_name_or_path,
                    trust_remote_code = "falcon" in model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto",
                    trust_remote_code = "falcon" in model_name_or_path)
        if self.model_config.model_type in ["llama", "phi3"] and "Llama-3" not in model_name_or_path:
            self.space_token = "▁"
        else:
            self.space_token = self.tokenizer.tokenize(' ')[0]
        
        self.sep_ids = [self.tokenizer.encode(token)[-1] for token in ['\n']]
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, input_text, max_length, return_logprobs=False):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        if return_logprobs:
            outputs = self.model.generate(
                input_ids = input_ids, 
                attention_mask = attention_mask,
                max_new_tokens = max_length, 
                return_dict_in_generate = True, 
                output_scores = True,
            )
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )

            generated_tokens = outputs.sequences[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0]) # text = "".join(tokens)
            tokens = [self.tokenizer.decode(t) for t in generated_tokens[0]]
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            return text, tokens, logprobs
        
        else:
            outputs = self.model.generate(
                input_ids = input_ids, 
                max_new_tokens = max_length, 
                attention_mask = attention_mask,
            )
            generated_tokens = outputs[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0])
            return text, None, None
    
    def generate_attn(self, input_text, max_length, solver="max", use_entropy = False, use_logprob = False):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            max_new_tokens = max_length, 
            return_dict_in_generate = True, 
            output_scores = True,
        )
        generated_tokens = outputs.sequences[:, input_length:]
        tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens[0])
        text = self.tokenizer.decode(generated_tokens[0])

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens):
            if i == 0 or t.startswith(self.space_token) or generated_tokens[0][i] in self.sep_ids or tokens[i-1] == '</s>':
                range_.append([i, i])
            else:
                range_[-1][-1] += 1

        # attention
        atten = self.model(generated_tokens, output_attentions=True).attentions[-1][0]
        if solver == "max": 
            mean_atten, _ = torch.max(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
        elif solver == "avg":
            mean_atten = torch.sum(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
            for i in range(mean_atten.shape[0]):
                mean_atten[i] /= (mean_atten.shape[0] - i)
        elif solver == "last_token":
            mean_atten = torch.mean(atten[:, -1], dim=0)
        else:
            raise NotImplementedError
        if mean_atten.shape[0] > 1 and tokens[0] == '</s>':
            mean_atten = mean_atten / sum(mean_atten[1:]).item()
        # mean_atten = mean_atten[tl:tr]
            
        # regular tokens
        seqlist = []
        attns = []
        for r in range_:
            tokenseq = "".join(tokens[r[0]: r[1]+1]).replace(self.space_token, "")
            value = sum(mean_atten[r[0]: r[1]+1]).item()
            seqlist.append(tokenseq)
            attns.append(value)

        # -log prob
        if use_logprob:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            seqlogprobs = []
            for r in range_:
                logprobseq = sum(logprobs[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqlogprobs.append(logprobseq)
        else:
            seqlogprobs = None

        # entropy
        if use_entropy:
            tmp = []
            for v in outputs.scores:
                tmp.append(v.cpu())
            softmax_probs = softmax(tmp, axis=-1)
            entropies = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)
            entropies = [v[0] for v in entropies]
            seqentropies = []
            for r in range_:
                entropyseq = sum(entropies[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqentropies.append(entropyseq) 
        else:
            seqentropies = None 

        return text, seqlist, attns, seqlogprobs, seqentropies
