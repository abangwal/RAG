"""
Contains Utility functions for LLM and Database module. Along with some other misllaneous functions.
"""

from pymupdf import pymupdf
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
import base64
import hashlib
import ollama
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
TOGETHER_API = os.getenv("TOGETHER_API_KEY")


def get_preview_pdf(file_bytes: bytes):
    """Returns first 3 pages of a PDF file."""

    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    sliced_doc = pymupdf.open()
    sliced_doc.insert_pdf(doc, from_page=0, to_page=2)

    return sliced_doc.tobytes()


def count_tokens(string: str) -> int:
    """Returns number of tokens in inputted string."""

    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text=string))


def create_refrences(retrieved_docs):
    """Create a refrences of chunks/pecies used in generating reponse, in markdown format"""

    refrences = ""
    for doc in retrieved_docs:
        try:
            chunk_imgs = eval(doc["metadata"]["images"])
        except:
            chunk_imgs = None
        chunk = doc["document"]

        if chunk_imgs:
            chunk_split = chunk.split("<img src='")
            chunk_with_img = ""

            if len(chunk_split) > 1:
                for i in range(0, len(chunk_split) - 1):
                    img_bytes = chunk_imgs[i]
                    base64_str = base64.b64encode(img_bytes).decode("utf-8")
                    chunk_with_img += (
                        chunk_split[i].strip()
                        + f"\n<img src='data:image/png;base64,{base64_str}'>\n"
                        + chunk_split[i + 1][3:]
                    )
            else:
                chunk_with_img = chunk

            refrences += (
                f"###### {doc['metadata']['file_name']}\n\n{chunk_with_img}\n\n"
            )
        else:
            chunk = doc["document"]
            refrences += f"###### {doc['metadata']['file_name']}\n\n{chunk}\n\n**Distance : {doc['distance']}**\n\n"

    return refrences


def generate_file_id(file_bytes):
    """Generate a Unique file ID for given file."""

    hash_obj = hashlib.sha256()
    hash_obj.update(file_bytes[:4096])
    file_id = hash_obj.hexdigest()[:63]
    return file_id


def extract_content_from_docx(docx_content):
    """Extract content (text) from DOCX file"""
    doc = Document(docx_content)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    content = "\n".join(full_text)
    return content


def extract_content_from_pdf(pdf_content):
    """Extereact content (Image + text) from PDF files."""

    doc = pymupdf.open(stream=pdf_content, filetype="pdf")
    DOCUMENT = ""
    pil_images = []

    for page in doc:

        blocks = page.get_text_blocks()  # type: ignore
        images = page.get_images()  # type: ignore

        # Create a list of all elements (text blocks and images) with their positions
        elements = [(block[:4], block[4], "text") for block in blocks]

        img_list = []
        for img in images:
            try:
                img_bbox = page.get_image_rects(img[0])[0]  # type: ignore
                if len(img_bbox) > 0:
                    img_data = (img_bbox, img[0], "image")
                    img_list.append(img_data)
                else:
                    continue
            except Exception as e:
                print("Exception :", e)
                pass

        elements.extend(img_list)

        # Sort elements by their vertical position (top coordinate)
        elements.sort(key=lambda x: x[0][1])

        for element in elements:
            if element[2] == "text":
                DOCUMENT += element[1]
            else:
                xref = element[1]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Save the image
                image = image_bytes
                pil_images.append(image)
                DOCUMENT += f"\n<img src='{len(pil_images)-1}'>\n\n"
    return DOCUMENT, pil_images


def chunk_document(document, chunk_size=200, overlap=10, encoding_name="cl100k_base"):
    """Split/Chunk Document with Recursive splitting strategy"""

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""], keep_separator=True
    ).from_tiktoken_encoder(
        encoding_name=encoding_name, chunk_size=chunk_size, chunk_overlap=overlap
    )
    chunks = splitter.split_text(document)
    return chunks


def generate_embedding_ollama(
    texts: List[str], embedding_model: str
) -> List[List[float]]:
    """Generate Embeddings for the givien pieces of texts."""

    embeddings = []
    for text in texts:
        embedding = ollama.embeddings(model=embedding_model, prompt=text)["embedding"]
        embeddings.append(list(embedding))

    return embeddings


def generate_embedding(texts: List[str], embedding_model: str) -> List[List[float]]:
    """Generate Embeddings for the givien pieces of texts."""

    client = OpenAI(api_key=TOGETHER_API, base_url="https://api.together.xyz/v1")
    embeddings_response = client.embeddings.create(
        input=texts, model="BAAI/bge-large-en-v1.5"
    ).data
    embeddings = [i.embedding for i in embeddings_response]
    return embeddings
