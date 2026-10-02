from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
import pymupdf
import time
from urllib.parse import quote_plus


class ArxivSearch:
    def search_titles(self, query, limit=20):
        driver = webdriver.Chrome()

        try:
            search_url = (
                    "https://arxiv.org/search/"
                    "?query="
                    + quote_plus(query)
                    + "&searchtype=all"
            )

            driver.get(search_url)

            wait = WebDriverWait(driver, 10)

            wait.until(
                EC.presence_of_all_elements_located(
                    (By.CSS_SELECTOR, "li.arxiv-result")
                )
            )

            results = driver.find_elements(
                By.CSS_SELECTOR,
                "li.arxiv-result"
            )

            articles = {}

            for result in results[:limit]:
                # Title
                title = result.find_element(
                    By.CSS_SELECTOR,
                    "p.title"
                ).text.strip()

                # Ссылка /abs/
                article_url = result.find_element(
                    By.CSS_SELECTOR,
                    "p.list-title a"
                ).get_attribute("href")

                # Делаем PDF URL
                pdf_url = article_url.replace(
                    "/abs/",
                    "/pdf/"
                )

                articles[title] = pdf_url
            return articles

        finally:
            driver.quit()

    def search_titles_1(self, query, limit=20):
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

            # Берем первые N статей
            for result in results[:limit]:
                title = result.find_element(
                    By.CSS_SELECTOR,
                    "p.title"
                ).text.strip()

                titles.append(title)
            return titles
        finally:
            driver.quit()


    def get_article_text(self, pdf_url):
        response = requests.get(
            pdf_url,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()

        pdf = pymupdf.open(
            stream=response.content,
            filetype="pdf"
        )

        text = ""

        for page in pdf:
            text += page.get_text() + "\n"

        pdf.close()

        return text