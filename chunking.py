import pandas as pd
import os
import re
import fitz
import hashlib

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        pdf_document = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text("text")
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""
def chunk_text(text, token_size):
    """Chunks text into fixed size considering sentence or paragraph cut off."""
    word_size = int(token_size * (3/4))
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    chunks = []
    current_chunk = []
    current_chunk_word_count = 0
    for sentence in sentences:
        words_in_sentence = sentence.split()
        if current_chunk_word_count + len(words_in_sentence) <= word_size:
            current_chunk.append(sentence)
            current_chunk_word_count += len(words_in_sentence)
        else:
            chunks.append(" ".join(current_chunk).strip())
            current_chunk = [sentence]
            current_chunk_word_count = len(words_in_sentence)
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())
    return chunks
def words_size_to_tokens_size(words_size):
    return int(words_size * (4/3))
def check_chunks_integrity(original_text, chunks):
    """
    Verifies if the concatenated chunks reconstruct the original text exactly.
    
    Args:
    - original_text (str): The original text extracted from the PDF.
    - chunks (list of str): A list of text chunks.

    Returns:
    - bool: True if the concatenated chunks match the original text, False otherwise.
    """
    reconstructed_text = ' '.join(chunks)
    normalized_original = re.sub(r'\s+', ' ', original_text).strip()
    normalized_reconstructed = re.sub(r'\s+', ' ', reconstructed_text).strip()
    return normalized_original == normalized_reconstructed
def generate_chunk_id(chunk_text):
    """
    Generates a unique ID for a chunk based on its text.
    
    Args:
    - chunk_text (str): The text of the chunk.

    Returns:
    - str: A unique identifier generated from the chunk text.
    """
    return hashlib.md5(chunk_text.encode('utf-8')).hexdigest()
def process_folder_to_dataframe(main_folder_path):
    data_list = []
    for subfolder_name in os.listdir(main_folder_path):
        subfolder_path = os.path.join(main_folder_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            bank_name = subfolder_name
            print(f"Processing Bank: {bank_name}")
            for file_name in os.listdir(subfolder_path):
                file_path:str = os.path.join(subfolder_path, file_name)
                if os.path.isfile(file_path) and file_path.endswith('.pdf'):
                    text = extract_text_from_pdf(file_path)
                    chunks = chunk_text(text, 500)
                    if not check_chunks_integrity(text, chunks):
                        print(f"Error: Chunks do not match the original text for {file_path}")
                    data_list.extend([{"bank": bank_name, "chunk": chunk} for chunk in chunks])
                    pass
    id_set=set()
    chunk_set=set()
    for data in data_list:
        words_size=len(data["chunk"].split())
        hash_id=generate_chunk_id(data["chunk"])
        tokens_size=words_size_to_tokens_size(words_size)
        data["hash"]=hash_id
        data["approximate_tokens_size"]=tokens_size
        id_set.add(hash_id)
        chunk_set.add(data["chunk"])
    if len(id_set)!=len(data_list):
        print("Error: Duplicate ID found")
    result_dataframe = pd.DataFrame(data_list)
    return result_dataframe

main_folder_path = "EarningRelease"

df=process_folder_to_dataframe(main_folder_path)
os.makedirs("dataset", exist_ok=True)
df.to_json(os.path.join("dataset","chunked_transcripts.jsonl"), orient="records", lines=True)