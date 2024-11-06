from .base_rag import BasicRAG
from .utils import nlp
from math import exp
import numpy as np

class FlareRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        tid = 0
        prev_tokens_count = 0
        for sid, sent in enumerate(sentences):
            if 'the answer is' in sent.lower():
                break
            pos = 0
            tr = tid
            while tr < len(tokens):
                apr = sent[pos:].find(tokens[tr].strip())
                if apr == -1:
                    break
                pos += apr + len(tokens[tr].strip())
                tr += 1
            probs = [1 - exp(v) for v in logprobs[tid:tr]]
            probs = np.array(probs)
            if (tid == tr):
                continue
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if sid > 0 and p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination 
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # replace all hallucinated tokens in current sentence with [xxx]
                curr = sent
                pos = 0
                # # 这里改成了替换掉最大的那个，而不是所有的
                # max_prob = 0
                # for prob, tok in zip(probs, tokens[tid:tr+1]):
                #     max_prob = max(prob, max_prob)
                for prob, tok in zip(probs, tokens[tid:tr+1]):
                    tok = tok.strip()
                    apr = curr[pos:].find(tok) + pos
                    if prob > self.hallucination_threshold:
                    # if prob == max_prob:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(tok):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(tok)
                return prev, curr, True, prev_tokens_count
            prev_tokens_count += tr - tid
            tid = tr
        
        # No hallucination
        return text, None, False, None
    
    def modifier_new(self, text, tokens, logprobs, entropies):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        tid = 0
        prev_tokens_count = 0
        for sid, sent in enumerate(sentences):
            if 'the answer is' in sent.lower():
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
        return text, None, False, None
    
    def inference(self, question, demo, case):
        # assert self.query_formulation == "direct"
        text = ""
        tokens_count = 0
        exemplars = "".join([d["case"]+"\n" for d in demo])
        hallucination = False
        curr = ""
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
                text = text.strip() + " " + sentences[0].strip()
                prompt =  exemplars + case + " " + text.strip()
                new_text, tokens, logprobs, entropies = self.generator.generate(
                    prompt, 
                    self.generate_max_length - tokens_count, 
                    return_logprobs=True
                )
            ptext, curr, hallucination, ptext_tokens_count = self.modifier_new(new_text, tokens, logprobs, entropies)
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
                        self.generate_max_length / 2
                    )
                # END OF DEBUGGGG
                text = text.strip() + " " + final_answer.strip()
                tokens_count += len(tokens)
                break
                
            text = text.strip() + " " + ptext.strip()
            tokens_count += ptext_tokens_count
            if tokens_count > self.generate_max_length:
                break
        return text

class EntityFlareRAG(FlareRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        entity = []
        for sent in sentences:
            doc = nlp(sent)
            li = [ent.text for ent in doc.ents]
            entity.append(li)
        
        belonging = [-1] * len(text)
        pos = 0
        for tid, tok in enumerate(tokens):
            apr = text[pos:].find(tok) + pos
            assert apr != -1
            for j in range(pos, apr+len(tok)):
                belonging[j] = tid
            pos = apr + len(tok)
        
        entity_intv = []
        for sid, sent in enumerate(sentences):
            tmp = []
            pos = text.find(sent)
            for ent in entity[sid]:
                apr = text[pos:].find(ent) + pos
                el = belonging[apr]
                er = belonging[apr + len(ent) - 1]
                tmp.append((el, er))
                pos = apr + len(ent)
            entity_intv.append(tmp)

        entity_prob = []
        for ent_itv_per_sent in entity_intv:
            tmp = []
            for itv in ent_itv_per_sent:
                probs = np.array(logprobs[itv[0]:itv[1]+1])
                p = {
                    "avg": np.mean,
                    "max": np.max,
                    "min": np.min,
                    "first": lambda x: x[0] if len(x) > 0 else 0
                }.get(self.entity_solver, lambda x: 0)(probs)
                tmp.append(p)
            entity_prob.append(tmp)

        tid = 0
        prev_tokens_count = 0
        for sid, sent in enumerate(sentences):
            if 'the answer is' in sent.lower():
                break
            if len(entity_prob[sid]) == 0:
                continue
            pos = 0
            tr = tid
            while tr < len(tokens):
                apr = sent[pos:].find(tokens[tr].strip())
                if apr == -1:
                    break
                pos += apr + len(tokens[tr].strip())
                tr += 1
            probs = [1 - exp(v) for v in entity_prob[sid]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination 
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # replace all hallucinated entities in current sentence with [xxx]
                curr = sent
                pos = 0
                for prob, ent in zip(probs, entity[sid]):
                    apr = curr[pos:].find(ent) + pos
                    if prob > self.hallucination_threshold:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(ent):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(ent)
                return prev, curr, True, prev_tokens_count
            prev_tokens_count += tr - tid
            tid = tr
        # No hallucination
        return text, None, False, None

    def inference(self, question, demo, case):
        return super().inference(question, demo, case)
