# Note for exps:
- flare-fixed:
    + 5 is for the prompt "Please use the information from the context above and your own knowledge and answer step by step in the same format as before. Please ensure that the final sentence the answer starts with \"So the answer is\".\n"
    + 6 is for the prompt "Ansert in the same format as before..."
    + 7 is like 6 but use exemplars when regenerating the answer.
    + 8 is like 7 without question as initial retrieval
    + 9 is like 5 but use "Let's think step by step" when regenerating final answer. ==> FAILED

# Qualitative observation:
- Final answer seems to be affected by the in-context exemplars, even if the chain-of-thought is correct:
    + Example: Which film came out earlier, Out Of The Rain or Les R\u00e9sultats Du F\u00e9minisme? ==> Out Of The Rain was released in 1991. Les R\u00e9sultats Du F\u00e9minisme was released in 1978. Thus, Out Of The Rain came out earlier. So the answer is Out Of The Rain.
    ==> Tendency to choose the firstly appeared entity in similar questions
