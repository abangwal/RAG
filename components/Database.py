"""
Contain Wrapper Class for ChormaDB client, that can process and store documents and retrive document chunks.
"""

from io import BytesIO
from typing import List
import uuid
import warnings
import chromadb
import re
from .utils import (
    generate_file_id,
    chunk_document,
    generate_embedding,
    extract_content_from_docx,
    extract_content_from_pdf,
)


class AdvancedClient:

    def __init__(self, vector_database_path: str = "vectorDB") -> None:
        self.client = chromadb.PersistentClient(path=vector_database_path)
        self.exsisting_collections = [
            collection.name for collection in self.client.list_collections()
        ]
        self.selected_collections = []

    def create_or_get_collection(
        self, file_names: List[str], file_types: List[str], file_datas
    ):
        collections = []
        for data in zip(file_names, file_types, file_datas):

            file_name, file_type, file_data = data
            file_id = generate_file_id(file_bytes=file_data)
            file_exisis = file_id in self.exsisting_collections

            if file_exisis:
                collection = self.client.get_collection(name=file_id)

            else:
                collection = self.client.create_collection(name=file_id)
                file_buffer = BytesIO(file_data)

                if file_type == "pdf":
                    document, pil_images = extract_content_from_pdf(file_buffer)
                    chunks = chunk_document(document)
                    ids = [f"{uuid.uuid4()}_id_{x}" for x in range(1, len(chunks) + 1)]
                    embeddings = generate_embedding(
                        chunks, embedding_model="znbang/bge:small-en-v1.5-q8_0"
                    )
                    metadatas = []

                    for chunk in chunks:
                        imgs_found = re.findall(
                            pattern=r"<img\s+src='([^']*)'>", string=chunk
                        )
                        chunk_imgs = []
                        if len(imgs_found) > 0:
                            for img in imgs_found:
                                chunk_imgs.append(pil_images[int(img)])
                        metadatas.append(
                            {"images": str(chunk_imgs), "file_name": file_name}
                        )

                elif file_type == "docx":
                    document = extract_content_from_docx(file_buffer)
                    chunks = chunk_document(document)
                    ids = [f"{uuid.uuid4()}_id_{x}" for x in range(1, len(chunks) + 1)]

                    embeddings = generate_embedding(
                        chunks, embedding_model="znbang/bge:small-en-v1.5-q8_0"
                    )
                    metadatas = [{"file_name": file_name} for _ in chunks]

                else:
                    raise Exception(
                        f"Given format '.{file_type}' is currently not supported."
                    )

                collection.add(
                    ids=ids,
                    embeddings=embeddings,  # type: ignore
                    documents=chunks,
                    metadatas=metadatas,  # type: ignore
                )
                # save
                try:
                    self.client.get_collection("UNION").add(
                        ids=ids,
                        embeddings=embeddings,  # type: ignore
                        documents=chunks,
                        metadatas=metadatas,  # type: ignore
                    )
                except:
                    self.client.create_collection("UNION").add(
                        ids=ids,
                        embeddings=embeddings,  # type: ignore
                        documents=chunks,
                        metadatas=metadatas,  # type:ignore
                    )
            collections.append(collection)

        self.selected_collections = collections

    def retrieve_chunks(self, query: str, number_of_chunks: int = 3):
        if len(self.selected_collections) == 0:

            warnings.warn(
                message=f"No collection is selected using all the exsisting collections, total collections : {len(self.exsisting_collections)}"
            )
            collections = [self.client.get_collection("UNION")]
            self.selected_collections = collections
        else:
            collections = self.selected_collections

        query_emb = generate_embedding(
            [query], embedding_model="znbang/bge:small-en-v1.5-q8_0"
        )

        retrieved_docs = []

        for collection in collections:
            results = collection.query(
                query_embeddings=query_emb,
                n_results=5,
                include=["documents", "metadatas", "distances"],
            )

            for i in range(len(results["ids"][0])):
                retrieved_docs.append(
                    {
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "collection": collection.name,
                    }
                )

        retrieved_docs = sorted(retrieved_docs, key=lambda x: x["distance"])

        return retrieved_docs[:number_of_chunks]
