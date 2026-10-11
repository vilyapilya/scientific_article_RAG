import logging
import re
from ArxivSearch import ArxivSearch
from sentence_transformers import CrossEncoder
from prompts import PROMPTS
from langchain_core.messages import SystemMessage, HumanMessage



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")



class RAGFlow:

    def __init__(self, params):
        self.params = params

        self.classifier = params["classifier"]
        self.query_model = params["query_model"]
        self.answer_model = params["answer_model"]
        self.titles_limit = params["titles_limit"]
        self.top_k_titles = params["top_k_titles"]
        self.chunk_size = params["chunk_size"]
        self.overlap_fraction = params["chunk_overlap_fraction"]
        self.top_k_chunks = params["top_k_chunks"]

        self.arxiv = ArxivSearch()
        self.titles_to_urls = {}


    def _generate_arxiv_query(self, prompt):
        system_prompt = PROMPTS["arxiv_query"].format(
            prompt=prompt
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]

        response = self.query_model.invoke(messages)
        query = response.content.strip()
        return query.strip().strip('"').strip("'")

    def _retrieve_top_k_titles(self, query, prompt):
        # query = self._generate_arxiv_query(prompt)
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
                self._generate_arxiv_query(prompt))
            logger.info(f"Retry query: {query}")
            self.titles_to_urls = self.arxiv.search_titles(
                query,
                limit=self.titles_limit
            )
            if self.titles_to_urls.get("status") == "timeout":
                return []

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


    def _retrieve_articles(self, top_titles, prompt):
        # top_titles = self._retrieve_top_k_titles(prompt)
        articles = []
        for title in top_titles:
            url = self.titles_to_urls[title]
            text = self.arxiv.get_article_text(url)
            if text is None:
                print(f"Skipping failed article: {title}")
                continue
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

    def __augment_prompt(self, chunks,  prompt):
        context = ""
        for item in chunks:
            context += f"""
                            Title: {item["title"]}
                            Content: {item["text"]}
                            URL: {item["article_url"]}
                            ---
                            """
        return PROMPTS["answer_prompt"].format(context=context, prompt=prompt)

    def _generate_answer(self, chunks, prompt, from_arxiv=True):
        if from_arxiv:
            prompt = self.__augment_prompt(chunks, prompt)

        messages = [
            HumanMessage(content=prompt)
        ]

        response = self.answer_model.invoke(messages)

        text = response.content.strip()

        # Input token count reported by Ollama
        input_tokens = response.usage_metadata.get(
            "input_tokens", 0
        ) if response.usage_metadata else 0

        logger.info(
            f"Answer generation input tokens: {input_tokens}"
        )

        if not from_arxiv:
            text = (
                "No relevant arXiv papers were found. "
                "The following answer is based on the model's "
                "general knowledge and is not grounded in "
                f"retrieved sources: {text}"
            )

        answer = {
            "text": text,
            "token_usage": input_tokens
        }

        return answer

    def _classify_promt(self, prompt):
        system_prompt = PROMPTS["classification_prompt"]

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]

        response = self.classifier.invoke(messages)

        return response.content.strip()

    def run(self, prompt):
        try:
            prompt_class = self._classify_promt(prompt)
            if(prompt_class == "not_valid"):
                return ("Please ask questions only realted to the following topics: "
                        "physics, computer science, mathematics, statistics, electrical engineering, "
                        "economics, or quantative finance")
            query = self._generate_arxiv_query(prompt)
            logger.info(
                f"user prompt: {prompt}, web query: {query}"
            )
            top_titles = self._retrieve_top_k_titles(query, prompt)
            articles = self._retrieve_articles(top_titles, prompt)
            if not articles:
                answer = self._generate_answer([], prompt, from_arxiv=False)
                return {
                    "answer": answer,
                    "status": "articles not found",
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
                "answer": answer["text"],
                "status": "success",
                "error": None,
                "token_usage": answer["token_usage"],
                "chunks": top_chunks
            }

        except Exception as e:
            logger.error(f"RAG failed: {e}")
            return {
                "answer": None,
                "status": "failed",
                "error": str(e)
            }