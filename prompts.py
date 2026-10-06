PROMPTS = {
    "arxiv_query": """
        Generate a GOOD short search query for finding a relevant paper on arXiv. 
        Rules:
        - Use only 2-4 key concepts.
        - Focus on the main research topic, not details from the question.
        - Do not include words like query, authors, approach, results, method.
        - Do not try to represent every part of the question.
        - Prefer terminology likely to appear in the paper title or abstract.
        - Do not use quotation marks.
        - Return only the search query.
        User question:
        {prompt}
        EXAMPLES:
        QUESTION
        How do authors minimize queries, achieve optimal tgt alignment,
        post-process, prompting in black-box generative AI?
        BAD QUERY
        query minimization target alignment post processing prompting
        black box generative models
        GOOD QUERY
        black box generative AI alignment
        """,
    "arxiv_query_retry": """
       The previous arXiv search query:
       {query}
       returned no results. It's a BAD query.
       Generate a NEW and BROADER search query for this question:
       {prompt}
       Rules:
       - Use only 2-3 key concepts.
       - Do not repeat the previous query.
       - Use broader academic terminology.
       - Focus on the main research topic.
       - Do not use quotation marks.
       - Return only the search query.
   """
}