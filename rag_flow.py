import logging
import re
from dotenv import load_dotenv
from ArxivSearch import ArxivSearch
from sentence_transformers import CrossEncoder
from prompts import PROMPTS
load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")



class RAGFlow:

    def __init__(self, params):
        self.params = params

        self.model = params["model"]
        self.tokenizer = params["tokenizer"]

        self.web_query_max_new_t = params["web_query_max_new_t"]
        self.titles_limit = params["titles_limit"]
        self.top_k_titles = params["top_k_titles"]

        self.chunk_size = params["chunk_size"]
        self.overlap_fraction = params["chunk_overlap_fraction"]
        self.top_k_chunks = params["top_k_chunks"]

        self.answer_max_new_t = params["answer_max_new_t"]

        self.arxiv = ArxivSearch()
        self.titles_to_urls = {}



    def _generate_arxiv_query(self, system_prompt, prompt):

        messages = [
            {"role": "user", "content": system_prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt"
        ).to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.web_query_max_new_t
        )
        output_ids = generated_ids[0][
            len(model_inputs.input_ids[0]):
        ].tolist()
        query = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True
        )
        logger.info(
            f"user prompt: {prompt}, web query: {query}"
        )

        return query


    def _retrieve_top_k_titles(self, prompt):
        system_prompt = PROMPTS["arxiv_query"].format(
            prompt=prompt
        )
        query = self._generate_arxiv_query(system_prompt, prompt).strip().strip('"').strip("'")
        self.titles_to_urls = self.arxiv.search_titles(
            query,
            limit=self.titles_limit
        )
        if self.titles_to_urls.get("status") == "timeout":
            system_prompt = PROMPTS["arxiv_query_retry"].format(
                prompt=prompt,
                query=query
            )
            query = (
                self._generate_arxiv_query(system_prompt, prompt).strip().strip('"').strip("'"))
            logger.info(f"Retry query: {query}")
            self.titles_to_urls = self.arxiv.search_titles(
                query,
                limit=self.titles_limit
            )

        titles = list(self.titles_to_urls.keys())
        ranks = cross_encoder.rank(
            prompt,
            titles
        )
        top_titles = [
            titles[rank["corpus_id"]]
            for rank in ranks[:self.top_k_titles]
        ]
        return top_titles


    def _retrieve_articles(self, prompt):
        top_titles = self._retrieve_top_k_titles(prompt)
        articles = []
        for title in top_titles:
            url = self.titles_to_urls[title]
            text = self.arxiv.get_article_text(url)

            articles.append({
                "article_url": url,
                "title": title,
                "text": text
            })

        return articles


    def __word_splitter(self, text):
        text = re.sub(r"\s+", " ", text).strip()
        return text.split()


    def _get_chunks_fixed_size_with_overlap(self, articles):
        chunks = []
        overlap = int(
            self.chunk_size * self.overlap_fraction
        )
        step = self.chunk_size - overlap

        for article in articles:
            url = article["article_url"]
            title = article["title"]
            text = article["text"]
            text_words = self.__word_splitter(text)
            count = 1
            for i in range(0, len(text_words), step):
                chunk_words = text_words[
                    i:i + self.chunk_size
                ]
                chunk = {
                    "article_url": url,
                    "title": title,
                    "chunk_index": count,
                    "text": " ".join(chunk_words)
                }
                chunks.append(chunk)
                count += 1

        return chunks


    def _select_top_k_chunks(self, chunks, prompt):
        chunk_texts = [
            chunk["text"]
            for chunk in chunks
        ]
        ranks = cross_encoder.rank(
            prompt,
            chunk_texts
        )
        top = []
        for rank in ranks[:self.top_k_chunks]:
            chunk = chunks[rank["corpus_id"]]
            top.append(chunk)
        return top


    def _generate_answer(self, chunks, prompt):
        context = ""
        for item in chunks:
            context += f"""
                Title: {item["title"]}
                Content: {item["text"]}
                URL: {item["article_url"]}
                ---
                """

        prompt = f"""
                Answer the question using only the information provided in the context. Include the url of the content that 
                you have used for generating the answer.
                If the context does not contain enough information to answer the question, say:
                "I don't have enough information in the provided context."
                Do not use information that is not supported by the context.
                Context:
                {context}  
                Question:
                {prompt}
                Answer:
                """

        messages = [
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt"
        ).to(self.model.device)
        logger.info(
            f"Answer generation input tokens: "
            f"{model_inputs['input_ids'].shape[1]}"
        )
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.answer_max_new_t
        )
        output_ids = generated_ids[0][
            len(model_inputs.input_ids[0]):
        ].tolist()
        answer = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True
        )
        return answer


    # def run(self, prompt):
    #     articles = self._retrieve_articles(prompt)
    #     chunks = self._get_chunks_fixed_size_with_overlap(
    #         articles
    #     )
    #     top_chunks = self._select_top_k_chunks(chunks, prompt)
    #     answer = self._generate_answer(top_chunks, prompt)
    #     return answer

    def run(self, prompt):
        try:
            articles = self._retrieve_articles(prompt)
            if not articles:
                return {
                    "answer": None,
                    "status": "failed",
                    "error": "No articles found on arXiv"
                }
            chunks = self._get_chunks_fixed_size_with_overlap(articles)
            if not chunks:
                return {
                    "answer": None,
                    "status": "failed",
                    "error": "No chunks generated"
                }
            top_chunks = self._select_top_k_chunks(chunks, prompt)
            answer = self._generate_answer(top_chunks, prompt)
            return {
                "answer": answer,
                "status": "success",
                "error": None
            }

        except Exception as e:
            logger.error(f"RAG failed: {e}")
            return {
                "answer": None,
                "status": "failed",
                "error": str(e)
            }