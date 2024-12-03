from .base_rag import BasicRAG
from .utils import nlp
from math import exp
import numpy as np

class DynamicThresholdingRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def modifier(self, text, tokens, logprobs, entropies):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        tid = 0
        prev = ""
        for sid, sent in enumerate(sentences):
            if 'the answer is' in sent.lower():
                if prev != "":
                    prev += " "
                prev += sent
                break
            pos = 0
            tr = tid
            while tr < len(tokens):
                apr = sent[pos:].find(tokens[tr].strip())
                if apr == -1:
                    break
                pos += apr + len(tokens[tr].strip())
                tr += 1
            probs = [exp(v) for v in logprobs[tid:tr]]
            probs = np.array(probs)
            cur_entropies = np.array(entropies[tid:tr])
            if (tid == tr):
                continue
            hallucination_list = (probs < cur_entropies / self.upper_bound)
            if np.any(hallucination_list): # hallucination
                # replace all hallucinated tokens in current sentence with [xxx]
                curr = sent
                pos = 0
                for hallucination, tok in zip(hallucination_list, tokens[tid:tr]):
                    tok = tok.strip()
                    apr = curr[pos:].find(tok) + pos
                    if hallucination:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(tok):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(tok)
                return prev, curr, True
            if prev != "":
                prev += " "
            prev += sent
            tid = tr
        
        # No hallucination
        return prev, None, False
    
    def inference(self, question, demo, case):
        # assert self.query_formulation == "direct"
        text = ""
        exemplars = "".join([d["case"]+"\n" for d in demo])
        hallucination = False
        curr = ""
        while True:
            if curr != None and len(curr) > 0:
                if self.query_formulation == "direct":
                    retrieve_question = curr.replace("[xxx]", "")
                elif self.query_formulation == "forward_all":
                    tmp_all = [question, text, ptext]
                    retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
                else:
                    raise NotImplemented
                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                prompt = exemplars + "Context:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[{i+1}] {doc}\n"
                prompt += "Answer in the same format as before. Please ensure that the final sentence of the answer starts with \"So the answer is\".\n"
            else:
                prompt = exemplars
            prompt += case + " " + text
            return_dict = self.generator.generate(
                prompt, 
                self.generate_max_length, 
                return_logprobs=True,
                return_entropies=True
            )
            new_text, tokens, logprobs, entropies = return_dict['text'], return_dict['tokens'], return_dict['logprobs'], return_dict['entropies']
            if hallucination:
                text = text.strip() + " " + self.get_top_sentence(new_text)
                prompt =  exemplars + case + " " + text.strip()
                return_dict = self.generator.generate(
                    prompt, 
                    self.generate_max_length, 
                    return_logprobs=True,
                    return_entropies=True
                )
                new_text, tokens, logprobs, entropies = return_dict['text'], return_dict['tokens'], return_dict['logprobs'], return_dict['entropies']
            ptext, curr, hallucination = self.modifier(new_text, tokens, logprobs, entropies)
            if self.use_counter == True:
                self.counter.add_generate(new_text, tokens)
                self.counter.hallucinated += hallucination
            if not hallucination:
                # DEBUGGG: try refine the final answer
                final_answer = ""
                text = text.strip() + " " + new_text.strip()
                so_truncate = text.find('. So ')
                thus_truncate = text.find('. Thus, ')
                if so_truncate >=0 or thus_truncate >=0:
                    truncate_idx = so_truncate if thus_truncate < 0 else thus_truncate if so_truncate < 0 else min(so_truncate, thus_truncate)
                    text = text[:truncate_idx + 1]
                    prompt = exemplars + case + " " + text.strip()
                    final_answer = self.generator.generate(
                        prompt, 
                        self.generate_max_length
                    )['text']
                if '\nQuestion' in final_answer:
                    final_answer = final_answer[:final_answer.find('\nQuestion')]
                text = text.strip() + " " + final_answer.strip()
                break
                
            text = text.strip() + " " + ptext.strip()
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or "the answer is" in text:
                break
        return text.strip()
