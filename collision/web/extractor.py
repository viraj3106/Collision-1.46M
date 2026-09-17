import re
import urllib.parse
from typing import List
from collision.web.schemas import WebDocument

class WebPageExtractor:
    """
    HTML text extraction and cleaning engine.
    Extracts high-signal readable body text while stripping boilerplate,
    navigation, headers, footers, and scripts.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"[\r\t\f\v]", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = re.sub(r"[ ]+", " ", text)
        return text.strip()

    def extract(self, html_content: str, url: str, title: str = "") -> WebDocument:
        """
        Parses HTML and extracts structured WebDocument containing readable paragraphs.
        """
        domain = urllib.parse.urlparse(url).netloc or "web"
        if not html_content or not html_content.strip():
            return WebDocument(url=url, title=title, domain=domain, text="", extracted_paragraphs=[])

        # If input is already clean plain text
        if "<html" not in html_content.lower() and "<body" not in html_content.lower() and "<p" not in html_content.lower():
            clean = self.clean_text(html_content)
            paragraphs = [p.strip() for p in clean.split("\n\n") if len(p.strip()) > 15]
            return WebDocument(url=url, title=title, domain=domain, text=clean, extracted_paragraphs=paragraphs)

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract title if not provided
            if not title and soup.title and soup.title.string:
                title = soup.title.string.strip()

            # Remove noise tags
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "svg", "iframe", "noscript"]):
                tag.decompose()

            paragraphs: List[str] = []
            for elem in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
                txt = elem.get_text(strip=True)
                if len(txt) > 20 and txt not in paragraphs:
                    paragraphs.append(txt)

            if not paragraphs:
                raw_text = soup.get_text(separator="\n")
                lines = [line.strip() for line in raw_text.splitlines() if len(line.strip()) > 20]
                paragraphs = list(dict.fromkeys(lines))

            full_text = self.clean_text("\n\n".join(paragraphs))
            return WebDocument(
                url=url,
                title=title or "Web Document",
                domain=domain,
                text=full_text,
                extracted_paragraphs=paragraphs
            )

        except Exception:
            # Regex fallback
            clean_html = re.sub(r"<(script|style|nav|footer|header)[^>]*>.*?</\1>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
            clean_text = re.sub(r"<[^>]+>", " ", clean_html)
            text_result = self.clean_text(clean_text)
            paras = [p.strip() for p in text_result.split("\n\n") if len(p.strip()) > 20]
            return WebDocument(
                url=url,
                title=title or "Web Document",
                domain=domain,
                text=text_result,
                extracted_paragraphs=paras
            )
