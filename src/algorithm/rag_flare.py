from math import exp

import numpy as np

from .base_rag import BasicRAG
from .utils import nlp


class FlareRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def modifier(self, text, tokens, logprobs, first_iter):
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
            probs = np.array([exp(v) for v in logprobs[tid:tr]])
            if (tid == tr):
                continue
            if (first_iter or sid > 0) and np.min(probs) < self.threshold: # hallucination
                # replace all hallucinated tokens in current sentence with [xxx]
                curr = sent
                pos = 0
                # max_prob = 0
                # for prob, tok in zip(probs, tokens[tid:tr+1]):
                #     max_prob = max(prob, max_prob)
                for prob, tok in zip(probs, tokens[tid:tr]):
                    tok = tok.strip()
                    apr = curr[pos:].find(tok) + pos
                    if prob < self.threshold:
                    # if prob == max_prob:
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
    
    def inference(self, questions, demos, cases):
        batch_size = len(questions)
        texts = [""] * batch_size
        hallucinations = [False] * batch_size
        currents = [""] * batch_size
        first_iters = [True] * batch_size
        generatings = [True] * batch_size

        while any(generatings):
            retrieve_questions = []
            for i in range(batch_size):
                if not generatings[i]:
                    retrieve_questions.append(None)
                    continue
                if first_iters[i] and len(currents[i]) == 0:
                    retrieve_questions.append(questions[i])
                elif self.query_formulation == "direct":
                    retrieve_questions.append(currents[i].replace("[xxx]", ""))
                elif self.query_formulation == "forward_all":
                    tmp_all = [questions[i], texts[i], currents[i]]
                    retrieve_questions.append(" ".join(s for s in tmp_all if len(s) > 0))
                else:
                    raise NotImplementedError

            # Filter out None retrieve_questions
            filtered_questions = [q for q in retrieve_questions if q is not None]
            valid_indices = [i for i, q in enumerate(retrieve_questions) if q is not None]
            docs_batch = self.retrieve(filtered_questions, topk=self.retrieve_topk)

            # Reconstruct the full docs_batch with None values
            full_docs_batch = [None] * batch_size
            for idx, docs in zip(valid_indices, docs_batch):
                full_docs_batch[idx] = docs

            prompts = []
            for i, (demo, case, text, docs) in enumerate(zip(demos, cases, texts, full_docs_batch)):
                if not generatings[i]:
                    prompts.append(None)
                    continue
                exemplars = "".join([d["case"] + "\n" for d in demo])
                prompt = exemplars + "Context:\n"
                for j, doc in enumerate(docs):
                    prompt += f"[{j+1}] {doc}\n"
                prompt += "Answer in the same format as before. Please ensure that the final sentence of the answer starts with \"So the answer is\".\n"
                prompt += case + " " + text
                prompts.append(prompt)

            return_dict = self.generator.generate(
                [p for p in prompts if p is not None],
                self.generate_max_length,
                return_logprobs=True
            )

            new_texts = return_dict['text']
            tokens_batch = return_dict['tokens']
            logprobs_batch = return_dict['logprobs']

            new_text_idx = 0
            for i in range(batch_size):
                if not generatings[i]:
                    continue

                ptext, curr, hallucination = self.modifier(
                    new_texts[new_text_idx], tokens_batch[new_text_idx], logprobs_batch[new_text_idx], first_iters[i]
                )
                new_text_idx += 1

                if self.use_counter:
                    self.counter.add_generate(new_texts[new_text_idx - 1], tokens_batch[new_text_idx - 1])
                    self.counter.hallucinated += hallucination

                if not hallucination:
                    texts[i] = texts[i].strip() + " " + ptext.strip()
                    generatings[i] = False
                else:
                    texts[i] = texts[i].strip() + " " + ptext.strip()
                    currents[i] = curr
                    hallucinations[i] = hallucination

                # Check stopping conditions
                tokens_count = len(self.generator.tokenizer.encode(texts[i]))
                first_iters[i] = False
                if tokens_count > self.generate_max_length or "the answer is" in texts[i]:
                    generatings[i] = False

        results = [text.strip() for text in texts]
        inference_results = dict(text=results)
        return inference_results

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
                return prev, curr, True
            tid = tr
        # No hallucination
        return text, None, False

    def inference(self, question, demo, case):
        return super().inference(question, demo, case)
