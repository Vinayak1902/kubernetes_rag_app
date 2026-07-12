from typing import List
import logfire

def chunk_text(text: str, chunk_size: int = 1500) -> List[str]:
    """
    Simple semantic-ish chunker that splits by paragraphs.
    Ensure chunks do not exceed the specified size.
    """
    with logfire.span("Text Chunking", text_length=len(text)):
        if not text.strip():
            return []
        
        # If the input text is empty or contains only whitespaces, return an empty list
        if not text.strip():
            return []
        
        # Split the text into paragraphs using double newline as the separator
        paragraphs = text.split("\n\n")

        # List to store the final text chunks
        chunks = []

        # Variable to build one chunk at a time
        current_chunk = ""

        # Iterate through each paragraph
        for p in paragraphs:

            # Check whether adding this paragraph keeps the chunk withing the size limit
            if len(current_chunk) + len(p) < chunk_size:

                # Add the paragraph to the current chunk
                current_chunk += p + "\n\n"

            else:

                # If the current chunk is not empty, save it to the chunk list
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # Start a new chunk with the current paragraph
                current_chunk = p + "\n\n"

        # After processing all paragraphs, save the last chunk if it not empty
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Remove any empty or whitespace-only chunks
        valid_chunks = [c for c in chunks if c.strip()]

        # Log the total number of generated chunks
        logfire.info(f"Generated {len(valid_chunks)} chunks")

        # Return the final list of valid chunks'
        return valid_chunks
            