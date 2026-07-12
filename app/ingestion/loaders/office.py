import logfire
from unstructured.partition.auto import partition

def parse_office(file_path: str):
    """
    Parses Office documents (.docx, .pptx) using the unstructured library.
    Unline pdfs, these formats are structured and lightweight, so they are processed locally.
    """
    with logfire.span("Office Document Parsing", filename=file_path):
        try:
            # unstructured automatically detects if it is docx or pptx
            # extracting its contents into structured elements
            elements = partition(filename=file_path)

            # convert each extracted element into string and join all together with newline separators
            full_text = "\n".join([str(el) for el in elements])

            # check if the extracted text is empty or contains only whitespace
            if not full_text.strip():
                logfire.warning(f"Unstructured returned empty text for {file_path}")
            else:
                logfire.info(f"Successfully parsed {len(full_text)} characters")
            
            return full_text
        except Exception as e:
            logfire.error(f"Office Parse Failed: {e}")
            raise e