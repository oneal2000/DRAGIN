from .base_rag import BasicRAG
from .utils import nlp
import numpy as np
import torch
from .semantic_entropy import *

class SemanticEntropyRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
        self.entailment_model = EntailmentDeberta()
    
    def modifier(self, text, tokens, attentions, weight):
        pass
    
    def infer_question_from_responses(self, responses):
        prompt = "\n".join([f'Answer: {response}' for response in responses])
        prompt += '\nGiven the above answers, ask a question to which the possible answer is presented above.\n'
        prompt += 'Question: '
        question, _, _ = self.generator.generate(prompt, 30)
        if '\nAnswer' in question:
            question = question[:question.find('\nAnswer')]
        return question
    
    def is_hallucination(self, prompt):
        new_texts, tokens_list, logprobs_list= self.generator.generate(
            prompt, 
            self.sample_max_length,
            num_return_sequences = 10,
            return_logprobs=True,
            return_entropies=False,
        )
        responses = []
        response_tokens_logprobs = []
        for new_text, tokens, logprobs in zip(new_texts, tokens_list, logprobs_list):
            sentences, pos_list = self.extract_sentence_token_positions(new_text, tokens)
            responses.append(sentences[0])
            response_tokens_logprobs.append(logprobs[pos_list[0][0]: pos_list[0][1]])
            
        semantic_ids = get_semantic_ids(
            responses, model=self.entailment_model,
            strict_entailment=self.strict_entailment, example=None)
        
        response_mean_logprobs = list(map(np.mean, response_tokens_logprobs))
        log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, response_mean_logprobs, agg='sum_normalized')
        pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
        
        if pe > self.threshold:
            return True, self.infer_question_from_responses(responses)
        
        return False, None
        
    def inference(self, question, demo, case):
        text = ""
        exemplars = "".join([d["case"]+"\n" for d in demo])
        hallucination = False
        curr = ""
        while True:
            prompt = exemplars
            hallucination, curr = self.is_hallucination(prompt + case + " " + text)
            if hallucination:
                retrieve_question = curr
                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                prompt = exemplars + "Context:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[{i+1}] {doc}\n"
                prompt += "Answer in the same format as before. Please ensure that the final sentence of the answer starts with \"So the answer is\".\n"

            prompt += case + " " + text
            
            new_text, tokens, _ = self.generator.generate(
                prompt, 
                self.generate_max_length,
                temperature=0.1
            )
            if self.use_counter == True:
                self.counter.add_generate(new_text, tokens)
                self.counter.hallucinated += hallucination
                
            text = text.strip() + " " + self.get_top_sentence(new_text).strip()
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or "the answer is" in text:
                break
        return text.strip()
