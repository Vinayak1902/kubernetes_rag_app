from bs4 import BeautifulSoup
import logfire

def parse_html(file_path: str):
    """
    This function parses the html file using BeatifulSoup
    It cleans the scripts, styles and extracts readable text for RAG
    """
    with logfire.span(" HTML Parsing", filename=file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            soup = BeautifulSoup(content, "html.parser")

            # 1. Remove Junk (Scripts, Styles, Metadata)
            for script in soup(["script", "style", "meta", "noscript"]):
                script.decompose()
            
            # 2. Extract text
            # seperator = "\n" inserts a newline after every html tag
            text = soup.get_text(separator="\n")

            # 3. Clean extracted text
            # split text into lines and remove leading/trailing whitespaces
            lines = (line.strip() for line in text.splitlines())

            # further split lines whenever there are double spaces
            # and remove extra spaces from each phrase
            chunks = (
                phrase.strip()
                for line in lines
                for phrase in line.split("  ")
            )

            # keep only non empty chunks and join them using newline characters
            text_clean = '\n'.join(chunk for chunk in chunks if chunk)

            # return the cleaned text
            return text_clean
        
        except Exception as e:

            logfire.error(f"HTML Parse Failed: {e}")

            raise e

