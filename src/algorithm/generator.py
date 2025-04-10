import numpy as np
import logging
import torch
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from .utils import Counter

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

class BasicGenerator:
    def __init__(self, model):
        logger.info(f"Loading model from {model}")
        self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side='left')
        self.model_config = AutoConfig.from_pretrained(model,
                    trust_remote_code = "falcon" in model)
        self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", torch_dtype=torch.bfloat16,
                    trust_remote_code = "falcon" in model)
        if self.model_config.model_type in ["llama", "phi3"] and "Llama-3" not in model:
            self.space_token = "▁"
        else:
            self.space_token = self.tokenizer.tokenize(' ')[0]
        
        self.sep_ids = [self.tokenizer.encode(token)[-1] for token in ['\n']]
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.counter = Counter()

    def generate(self, input_text, max_length, return_logprobs=True, num_return_sequences=1, return_entropies=True, temperature=1.0):
        input = self.tokenizer(input_text, return_tensors="pt", padding=True)
        input = input.to(self.model.device)
        return_dict = dict()
        
        outputs = self.model.generate(
            input_ids = input['input_ids'], 
            attention_mask = input['attention_mask'],
            max_new_tokens = max_length, 
            num_return_sequences = num_return_sequences,
            return_dict_in_generate = True, 
            output_scores = True,
            temperature=temperature
        )
        input_length = input['input_ids'].shape[-1]
        generated_tokens = outputs.sequences[:, input_length:]
        texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        # generate output_scores shape: tuple of tensor -> (max_length, batch_size, vocab_size)
        if return_logprobs:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            if num_return_sequences == 1:
                # text = self.tokenizer.decode(generated_tokens[0]) # text = "".join(tokens)
                tokens = [[self.tokenizer.decode(t) for t in each_generated_tokens] for each_generated_tokens in generated_tokens]
                logprobs = [[p.cpu().numpy().item() for p in logprobs] for logprobs in transition_scores]
                assert len(tokens) == len(logprobs)
            else:
                tokens = [[self.tokenizer.decode(t) for t in generated_tokens[i]] for i in range(num_return_sequences)]
                logprobs = [transition_scores[i] for i in range(num_return_sequences)]
                logprobs = [[p.cpu().numpy().item() for p in row] for row in logprobs]
            if return_entropies:
                tmp = []
                for v in outputs.scores:
                    tmp.append(v.cpu())
                softmax_probs = softmax(tmp, axis=-1)
                entropies = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)
                # seqentropies = [v[0] for v in entropies]
                return_dict['entropies'] = entropies.T.tolist()

            return_dict['text'] = texts
            return_dict['tokens'] = tokens
            return_dict['logprobs'] = logprobs
        
        else:
            # tokens_cnts = [tokens.shape[-1] - input_length for tokens in generated_tokens]
            # self.counter.add_generate(texts, tokens_cnts)
            return_dict['text'] = texts
        
        return return_dict
    
    def generate_attn(self, input_text, max_length, solver="max", use_entropy = False, use_logprob = False):
        # Handle both single string and list of strings input
        if isinstance(input_text, str): input_text = [input_text]
            
        return_dict = dict()

        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)

        outputs = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask = inputs['attention_mask'],
            max_new_tokens = max_length, 
            return_dict_in_generate = True, 
            output_scores = True,
        )

        input_length = inputs['input_ids'].shape[-1]
        generated_tokens = outputs.sequences[:, input_length:]
        texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        return_dict['text'] = texts
        return_dict['tokens'] = []
        return_dict['attentions'] = []
        if use_logprob:
            return_dict['logprobs'] = []
        if use_entropy:
            return_dict['entropies'] = []
        
        for i in range(len(input_text)):
            tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens[i])
            
            # merge tokens
            range_ = []
            for j, t in enumerate(tokens):
                if j == 0 or t.startswith(self.space_token) or generated_tokens[i][j] in self.sep_ids or (j > 0 and tokens[j-1] == '</s>'):
                    range_.append([j, j])
                else:
                    range_[-1][-1] += 1
            
            # attention
            atten = self.model(generated_tokens[i].unsqueeze(0), output_attentions=True).attentions[-1][0]
            if solver == "max": 
                mean_atten, _ = torch.max(atten, dim=1)
                mean_atten = torch.mean(mean_atten, dim=0)
            elif solver == "avg":
                mean_atten = torch.sum(atten, dim=1)
                mean_atten = torch.mean(mean_atten, dim=0)
                for j in range(mean_atten.shape[0]):
                    mean_atten[j] /= (mean_atten.shape[0] - j)
            elif solver == "last_token":
                mean_atten = torch.mean(atten[:, -1], dim=0)
            else:
                raise NotImplementedError
            if mean_atten.shape[0] > 1 and tokens[0] == '</s>':
                mean_atten = mean_atten / sum(mean_atten[1:]).item()
                
            # regular tokens
            seqlist = []
            attns = []
            for r in range_:
                tokenseq = "".join(tokens[r[0]: r[1]+1]).replace(self.space_token, "")
                value = sum(mean_atten[r[0]: r[1]+1]).item()
                seqlist.append(tokenseq)
                attns.append(value)
            
            return_dict['tokens'].append(seqlist)
            return_dict['attentions'].append(attns)
            
            # -log prob
            if use_logprob:
                transition_scores = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )
                logprobs = transition_scores[i]
                logprobs = [p.cpu().numpy() for p in logprobs]
                assert len(tokens) == len(logprobs)
                seqlogprobs = []
                for r in range_:
                    logprobseq = sum(logprobs[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                    seqlogprobs.append(logprobseq)
                return_dict['logprobs'].append(seqlogprobs)
            
            # entropy
            if use_entropy:
                tmp = []
                for v in outputs.scores:
                    tmp.append(v.cpu())
                softmax_probs = softmax(tmp, axis=-1)
                entropies = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)
                entropies = [v[i] for v in entropies]
                seqentropies = []
                for r in range_:
                    entropyseq = sum(entropies[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                    seqentropies.append(entropyseq)
                return_dict['entropies'].append(seqentropies)
        
        return return_dict
