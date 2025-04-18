from math import exp

import numpy as np

from .base_rag import BasicRAG
from .utils import nlp
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("flare.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


DEBUG = os.environ.get("DEBUG", "0") == "1"


def mark_hallucinated_tokens(sentence, sentence_tokens, probabilities, threshold):
    """Mark tokens with probabilities below threshold as [xxx]."""
    marked_sentence = sentence
    position = 0

    for prob, token in zip(probabilities, sentence_tokens):
        token = token.strip()
        token_position = marked_sentence[position:].find(token) + position

        if prob < threshold:
            # Replace low-confidence token with [xxx]
            marked_sentence = (
                marked_sentence[:token_position]
                + "[xxx]"
                + marked_sentence[token_position + len(token) :]
            )
            position = token_position + len("[xxx]")
        else:
            position = token_position + len(token)
    return marked_sentence


def find_token_range_for_sentence(sentence, tokens, start_index):
    """Helper function to find which tokens belong to a sentence."""
    position = 0
    token_range = start_index

    while token_range < len(tokens):
        # Find the current token in the remaining part of the sentence
        token_position = sentence[position:].find(tokens[token_range].strip())
        if token_position == -1:
            break

        # Move past this token in the sentence
        position += token_position + len(tokens[token_range].strip())
        token_range += 1

    return token_range


class FlareRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def modifier(self, text, tokens, logprobs, first_iter):
        """
        Analyze text for hallucinations by detecting low-confidence tokens.
        """

        # Split text into sentences and remove empty ones
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        token_index = 0
        processed_text = ""

        for sentence_id, sentence in enumerate(sentences):
            # Handle the final answer separately (usually concludes with "the answer is")
            if "the answer is" in sentence.lower():
                if processed_text:
                    processed_text += " "
                processed_text += sentence
                break

            # Find which tokens belong to the current sentence
            sentence_token_range = find_token_range_for_sentence(
                sentence, tokens, token_index
            )
            if sentence_token_range == token_index:
                continue

            # Calculate probabilities for all tokens in the sentence
            token_probabilities = np.array(
                [exp(v) for v in logprobs[token_index:sentence_token_range]]
            )

            if (first_iter or sentence_id > 0) and np.min(
                token_probabilities
            ) < self.threshold:
                # Hallucination detected - mark low confidence tokens
                marked_sentence = mark_hallucinated_tokens(
                    sentence,
                    tokens[token_index:sentence_token_range],
                    token_probabilities,
                    self.threshold,
                )
                if DEBUG:
                    logger.info("----------------")
                    logger.info(f"original sentence: {sentence}")
                    logger.info(f"masked_sentence: {marked_sentence}")
                return processed_text, marked_sentence, True

            # No hallucination in this sentence, add it to processed text
            if processed_text:
                processed_text += " "
            processed_text += sentence
            token_index = sentence_token_range

        # No hallucination found in any sentence
        return processed_text, None, False

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
                    new_texts[new_text_idx],
                    tokens_batch[new_text_idx],
                    logprobs_batch[new_text_idx],
                    first_iters[i],
                )
                new_text_idx += 1

                if self.use_counter:
                    self.counter.add_generate(
                        new_texts[new_text_idx - 1], tokens_batch[new_text_idx - 1]
                    )
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
                if (
                    tokens_count > self.generate_max_length
                    or "\nQuestion" in texts[i]
                    or ". Question" in texts[i]
                ):
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
