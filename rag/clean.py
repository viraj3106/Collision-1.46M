import re

def clean_extracted_text(text: str) -> str:
    """
    Cleans raw text by normalizing whitespace, removing redundant lines, and stripping control chars.
    """
    if not text:
        return ""
    # Normalize unicode whitespace
    text = re.sub(r'[\r\t\f\v]', ' ', text)
    # Collapse multiple consecutive blank lines
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    # Collapse multiple spaces
    text = re.sub(r'[ ]+', ' ', text)
    return text.strip()

def extract_text_from_html(html_content: str) -> str:
    """
    Extracts high-quality body paragraphs and headings while removing navigation, boilerplate, ads, and scripts.
    """
    if not html_content or not html_content.strip():
        return ""

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove noisy tags
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "svg", "iframe", "noscript"]):
            tag.decompose()

        # Extract text from remaining headings and paragraphs
        paragraphs = []
        for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
            txt = element.get_text(strip=True)
            if len(txt) > 20:  # Filter out tiny boilerplate strings
                paragraphs.append(txt)

        if not paragraphs:
            # Fallback to general text extraction
            raw_text = soup.get_text(separator="\n")
            lines = [line.strip() for line in raw_text.splitlines() if len(line.strip()) > 20]
            paragraphs = lines

        return clean_extracted_text("\n\n".join(paragraphs))

    except Exception:
        # Regex fallback if BeautifulSoup fails
        clean_html = re.sub(r'<(script|style|nav|footer|header)[^>]*>.*?</\1>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        clean_text = re.sub(r'<[^>]+>', ' ', clean_html)
        return clean_extracted_text(clean_text)
