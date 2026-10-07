from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from urllib.parse import quote_plus


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

