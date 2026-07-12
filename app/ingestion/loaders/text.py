import logfire

def parse_text(file_path: str):
    """
    Parses plain text files.
    """

    # start a Logfire tracing span for this text parsing operation
    with logfire.span("Text Parsing", filename=file_path):

        try:
            # open the text file in read mode
            # use utf-8 encoding and ignore invalid characters
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:

                # read the entire contents of the file and return it
                return f.read()
        
        except Exception as e:

            # log any errors encountered while reading the file
            logfire.error(f"Text Parse Failed: {e}")

            raise e