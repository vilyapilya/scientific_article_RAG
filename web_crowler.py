# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
#
# from urllib.parse import quote_plus
#
# from pypdf import PdfReader
#
# import requests
# import io
#
#
# # --------------------------------------------------
# # Извлекаем текст из PDF
# # --------------------------------------------------
#
# def extract_pdf_text(pdf_url):
#     print(f"Downloading PDF: {pdf_url}")
#
#     response = requests.get(
#         pdf_url,
#         timeout=30,
#         headers={
#             "User-Agent": "Mozilla/5.0"
#         }
#     )
#
#     response.raise_for_status()
#
#     # PDF хранится только в памяти
#     pdf_file = io.BytesIO(response.content)
#
#     reader = PdfReader(pdf_file)
#
#     pages = []
#
#     for page in reader.pages:
#         text = page.extract_text()
#
#         if text:
#             pages.append(text)
#
#     return "\n".join(pages)
#
#
# # --------------------------------------------------
# # Ищем статьи на arXiv
# # --------------------------------------------------
#
# def search_arxiv(query, limit=5):
#
#     driver = webdriver.Chrome()
#
#     try:
#
#         # Создаём URL поиска напрямую
#         search_url = (
#             "https://arxiv.org/search/"
#             "?query="
#             + quote_plus(query)
#             + "&searchtype=all"
#         )
#
#         print(f"Searching: {query}")
#         print(search_url)
#
#         driver.get(search_url)
#
#         wait = WebDriverWait(driver, 10)
#
#         # Ждём результаты
#         wait.until(
#             EC.presence_of_all_elements_located(
#                 (By.CSS_SELECTOR, "li.arxiv-result")
#             )
#         )
#
#         results = driver.find_elements(
#             By.CSS_SELECTOR,
#             "li.arxiv-result"
#         )
#
#         print(f"Found {len(results)} results")
#
#         articles = []
#
#         # --------------------------------------------------
#         # Берём первые N статей
#         # --------------------------------------------------
#
#         for result in results[:limit]:
#
#             title = result.find_element(
#                 By.CSS_SELECTOR,
#                 "p.title"
#             ).text.strip()
#
#             # Первая ссылка ведёт на /abs/
#             abstract_link = result.find_element(
#                 By.CSS_SELECTOR,
#                 "p.list-title a"
#             )
#
#             abstract_url = abstract_link.get_attribute("href")
#
#             # Например:
#             #
#             # https://arxiv.org/abs/2401.12345
#             #
#             # превращаем в:
#             #
#             # https://arxiv.org/pdf/2401.12345
#
#             pdf_url = abstract_url.replace(
#                 "/abs/",
#                 "/pdf/"
#             )
#
#             print()
#             print("=" * 60)
#             print(title)
#             print("=" * 60)
#
#             # Получаем полный текст PDF
#             try:
#                 text = extract_pdf_text(pdf_url)
#
#             except Exception as error:
#
#                 print("Could not read PDF:")
#                 print(error)
#
#                 text = ""
#
#             article = {
#                 "title": title,
#                 "url": abstract_url,
#                 "pdf_url": pdf_url,
#                 "text": text
#             }
#
#             articles.append(article)
#
#         return articles
#
#     finally:
#
#         driver.quit()
#
#
# # --------------------------------------------------
# # Запуск программы
# # --------------------------------------------------
#
# if __name__ == "__main__":
#
#     articles = search_arxiv(
#         query="retrieval augmented generation",
#         limit=5
#     )
#
#     print()
#     print("DONE")
#     print(f"Articles received: {len(articles)}")
#
#     # Показываем результат
#     for index, article in enumerate(articles, start=1):
#
#         print()
#         print("#" * 80)
#         print(f"ARTICLE {index}")
#         print("#" * 80)
#
#         print("TITLE:")
#         print(article["title"])
#
#         print("\nURL:")
#         print(article["url"])
#
#         print("\nTEXT LENGTH:")
#         print(len(article["text"]))
#
#         print("\nTEXT PREVIEW:")
#
#         # Только первые 1500 символов,
#         # чтобы не завалить терминал
#         print(article["text"][:1500])

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from urllib.parse import quote_plus
import LLM_model

def search_arxiv(query, limit=20):

    driver = webdriver.Chrome()

    try:
        # Создаем URL поиска
        search_url = (
            "https://arxiv.org/search/"
            "?query="
            + quote_plus(query)
            + "&searchtype=all"
        )

        driver.get(search_url)

        wait = WebDriverWait(driver, 10)

        # Ждем появления результатов
        wait.until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, "li.arxiv-result")
            )
        )

        results = driver.find_elements(
            By.CSS_SELECTOR,
            "li.arxiv-result"
        )

        titles = []

        # Берем первые 20 статей
        for result in results[:limit]:

            title = result.find_element(
                By.CSS_SELECTOR,
                "p.title"
            ).text.strip()

            titles.append(title)

        return titles

    finally:
        driver.quit()


if __name__ == "__main__":

    titles = search_arxiv(
        query="retrieval augmented generation",
        limit=20
    )

    for index, title in enumerate(titles, start=1):
        print(f"{index}. {title}")

