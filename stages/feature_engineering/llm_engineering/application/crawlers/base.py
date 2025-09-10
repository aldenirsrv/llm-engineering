import time
from abc import ABC, abstractmethod
from tempfile import mkdtemp

import chromedriver_autoinstaller
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

from llm_engineering.domain.documents import NoSQLBaseDocument


class BaseCrawler(ABC):
    model: type[NoSQLBaseDocument]

    @abstractmethod
    def extract(self, link: str, **kwargs) -> None: ...


class BaseSeleniumCrawler(BaseCrawler, ABC):
    def __init__(self, scroll_limit: int = 5) -> None:
        # Chrome options
        options = webdriver.ChromeOptions()

        # Flags para estabilidade em ambientes headless/macOS
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--log-level=3")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-background-networking")
        options.add_argument("--ignore-certificate-errors")
        options.add_argument("--window-size=1920,1080")

        # Usar headless antigo por compatibilidade
        options.add_argument("--headless")

        # Temp dirs isolados (evita conflito com perfis locais do Chrome)
        options.add_argument(f"--user-data-dir={mkdtemp()}")
        options.add_argument(f"--data-path={mkdtemp()}")
        options.add_argument(f"--disk-cache-dir={mkdtemp()}")
        options.add_argument("--remote-debugging-port=9226")

        # Permite que classes filhas adicionem configs extras
        self.set_extra_driver_options(options)

        self.scroll_limit = scroll_limit

        # Garante que o chromedriver correto será usado
        chromedriver_path = chromedriver_autoinstaller.install()
        service = Service(chromedriver_path)

        self.driver = webdriver.Chrome(
            service=service,
            options=options,
        )

    def set_extra_driver_options(self, options: Options) -> None:
        """Override em classes filhas para adicionar configs extras ao driver"""
        pass

    def login(self) -> None:
        pass

    def scroll_page(self) -> None:
        """Scroll through the LinkedIn page based on the scroll limit."""
        current_scroll = 0
        last_height = self.driver.execute_script("return document.body.scrollHeight")

        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(5)

            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height or (
                self.scroll_limit and current_scroll >= self.scroll_limit
            ):
                break

            last_height = new_height
            current_scroll += 1