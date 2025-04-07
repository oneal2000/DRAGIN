from .base_rag import BasicRAG


class FixSentenceRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, questions, demos, cases):
        assert self.query_formulation == "direct"
        texts = [""] * len(questions)
        retrieve_questions = questions
        generatings = [True] * len(questions)
        while True:
            old_len = [len(text) for text in texts]
            docs = self.retrieve(retrieve_questions, topk=self.retrieve_topk)
        # 对 topk 个 passage 生成 prompt
            prompts = []
            doc_id = 0
            for generating, text, demo, case in zip(generatings, texts, demos, cases):
                if not generating:
                    continue
                prompt = "".join([d["case"]+"\n" for d in demo])
                prompt += "Context:\n"
                for i, doc in enumerate(docs[doc_id]):
                    prompt += f"[{i+1}] {doc}\n"
                prompt += "Answer in the same format as before.\n"
                prompt += case + " " + text
                prompts.append(prompt)
                doc_id += 1

            return_dict = self.generator.generate(prompts, self.generate_max_length)

            new_texts = [self.get_top_sentence(new_text) for new_text in return_dict['text']]
            new_text_id = 0
            for i in range(len(generatings)):
                if not generatings[i]:
                    continue
                texts[i] = (texts[i].strip() + " " + new_texts[new_text_id].strip()).strip()
                new_text_id += 1
            
            # 判断 token 的个数要少于 generate_max_length 
            tokens_list = self.generator.tokenizer.batch_encode_plus(texts)['input_ids']
            tokens_cnts = [len(tokens) - 1 for tokens in tokens_list]
            new_text_id = 0
            retrieve_questions = []
            for i in range(len(generatings)):
                if not generatings[i]:
                    continue
                if tokens_cnts[i] > self.generate_max_length or len(texts[i]) <= old_len[i] or "\nQuestion" in texts[i] or ". Question" in texts[i]:
                    generatings[i] = False
                else:
                    retrieve_questions.append(new_texts[new_text_id].strip())
                new_text_id += 1
            if all(not generating for generating in generatings):
                break
        inference_results = dict(text=texts)
        return inference_results
