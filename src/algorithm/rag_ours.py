from .base_rag import BasicRAG
from .utils import nlp
from math import exp
import numpy as np

class OurRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def modifier(self, text, tokens, logprobs, entropies, threshold_scale):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        tid = 0
        prev_tokens_count = 0
        prev = ""
        for sid, sent in enumerate(sentences):
            if 'the answer is' in sent.lower():
                prev = " ".join(sentences[:sid+1])
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
            hallucination_list = (probs < (cur_entropies / 2.5))
            if np.any(hallucination_list): # hallucination
                # keep sentences before hallucination 
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # replace all hallucinated tokens in current sentence with [xxx]
                curr = sent
                pos = 0
                # # 这里改成了替换掉最大的那个，而不是所有的
                # max_prob = 0
                # for prob, tok in zip(probs, tokens[tid:tr+1]):
                #     max_prob = max(prob, max_prob)
                for prob, hallucination, tok in zip(probs, hallucination_list, tokens[tid:tr+1]):
                    tok = tok.strip()
                    apr = curr[pos:].find(tok) + pos
                    if hallucination:
                    # if prob == max_prob:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(tok):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(tok)
                return prev, curr, True, prev_tokens_count
            prev_tokens_count += tr - tid
            tid = tr
        
        # No hallucination
        return prev, None, False, None
    
    def inference(self, question, demo, case):
        # assert self.query_formulation == "direct"
        text = ""
        tokens_count = 0
        exemplars = "".join([d["case"]+"\n" for d in demo])
        hallucination = False
        curr = ""
        iter = 0
        threshold_scale = 1
        while True:
            # old_len = len(text)
            # if tokens_count == 0 or hallucination:
            if curr != None and len(curr) > 0:
                # if tokens_count == 0 and len(curr) == 0:
                #     retrieve_question = question
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
            # else:
            #     prompt = exemplars + case + " " + text

            # DEBUG: for analysis token's confidence
            # new_text, tokens, logprobs, entropies = self.generator.generate(
            #     prompt, 
            #     self.generate_max_length, 
            #     return_logprobs=True
            # )
            # if any(s.endswith('Question') for s in tokens):
            #     truncate_id = [i for i, s in enumerate(tokens) if s.endswith("Question")][0]
            #     tokens = tokens[:truncate_id]
            #     logprobs = logprobs[:truncate_id]
            #     entropies = entropies[:truncate_id]
            #     new_text = new_text[:new_text.find('Question')]
            # confidence_list = []
            # for token, logprob, entropy in zip(tokens,[logprob.item() for logprob in logprobs], [entropy.item() for entropy in entropies]):
            #     confidence_list.append((token, exp(logprob), entropy))
            # return new_text, confidence_list
            # End of DEBUGGGG
            new_text, tokens, logprobs, entropies = self.generator.generate(
                prompt, 
                self.generate_max_length - tokens_count, 
                return_logprobs=True
            )
            if hallucination:
                sentences = [sent.text.strip() for sent in nlp(new_text).sents]
                for sentence in sentences:
                    if sentence.strip() != "":
                        text = text.strip() + " " + sentence.strip()
                        break
                prompt =  exemplars + case + " " + text.strip()
                new_text, tokens, logprobs, entropies = self.generator.generate(
                    prompt, 
                    self.generate_max_length - tokens_count, 
                    return_logprobs=True
                )
                if "the answer is" in text:
                    break
            ptext, curr, hallucination, ptext_tokens_count = self.modifier(new_text, tokens, logprobs, entropies, threshold_scale)
            # ptext, curr, hallucination, ptext_tokens_count = self.modifier(new_text, tokens, logprobs)
            if self.use_counter == True:
                self.counter.add_generate(new_text, tokens)
                self.counter.hallucinated += hallucination
            if not hallucination:
                # DEBUGGG: try refine the final answer
                final_answer = ""
                text = text.strip() + " " + new_text.strip()
                so_truncate = text.find('. So ')
                thus_truncate = text.find('. Thus, ')
                # thus_truncate = -1
                if so_truncate >=0 or thus_truncate >=0:
                    truncate_idx = so_truncate if thus_truncate < 0 else thus_truncate if so_truncate < 0 else min(so_truncate, thus_truncate)
                    text = text[:truncate_idx]
                    prompt = exemplars + case + " " + text.strip()
                    final_answer, _, _ = self.generator.generate(
                        prompt, 
                        self.generate_max_length - tokens_count
                    )
                # END OF DEBUGGGG
                text = text.strip() + " " + final_answer.strip()
                tokens_count += len(tokens)
                break
                
            text = text.strip() + " " + ptext.strip()
            tokens_count += ptext_tokens_count
            if tokens_count > self.generate_max_length or "the answer is" in text:
                break
            threshold_scale *= 2
        return text.strip()
