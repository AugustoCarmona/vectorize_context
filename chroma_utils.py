import pathlib
from typing import Iterator, Sequence


def batched_records(
    ids: Sequence[str],
    documents: Sequence[str],
    metadatas: Sequence[dict],
    batch_size: int = 166,
) -> Iterator[tuple[list[str], list[str], list[dict]]]:
    if batch_size <= 0:
        raise ValueError("batch_size debe ser mayor a cero.")

    total_records = len(documents)
    if len(ids) != total_records or len(metadatas) != total_records:
        raise ValueError("ids, documents y metadatas deben tener la misma longitud.")

    for start_idx in range(0, total_records, batch_size):
        end_idx = min(start_idx + batch_size, total_records)
        yield (
            list(ids[start_idx:end_idx]),
            list(documents[start_idx:end_idx]),
            list(metadatas[start_idx:end_idx]),
        )


def build_embedding_function(embedding_func_name: str):
    from chromadb.utils import embedding_functions

    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_func_name
    )


def list_collection_names(chroma_client) -> set[str]:
    collection_names: set[str] = set()
    for collection in chroma_client.list_collections():
        collection_names.add(collection.name if hasattr(collection, "name") else collection)
    return collection_names


def get_collection(
    chroma_path: pathlib.Path,
    collection_name: str,
    embedding_func_name: str,
):
    import chromadb

    chroma_client = chromadb.PersistentClient(str(chroma_path))
    available_collections = list_collection_names(chroma_client)

    if collection_name not in available_collections:
        raise RuntimeError(
            f"La colección {collection_name} no existe en {chroma_path}. Ejecutá `python3 app.py ingest`."
        )

    return chroma_client.get_collection(
        name=collection_name,
        embedding_function=build_embedding_function(embedding_func_name),
    )


def build_chroma_collection(
    chroma_path: pathlib.Path,
    collection_name: str,
    embedding_func_name: str,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    distance_func_name: str = "cosine",
    *,
    reset: bool = True,
    batch_size: int = 166,
) -> int:
    """
    Crea o actualiza una colección en ChromaDB utilizando un modelo de embeddings.
    """
    import chromadb

    if not documents:
        raise ValueError("No hay documentos para indexar.")

    chroma_client = chromadb.PersistentClient(str(chroma_path))
    existing_collections = list_collection_names(chroma_client)

    if reset and collection_name in existing_collections:
        chroma_client.delete_collection(collection_name)
        existing_collections.remove(collection_name)

    embedding_func = build_embedding_function(embedding_func_name)

    if collection_name in existing_collections:
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=embedding_func,
        )
    else:
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=embedding_func,
            metadata={"hnsw:space": distance_func_name},
        )

    write_batch = getattr(collection, "upsert", None) or collection.add

    total_indexed = 0
    for batch_ids, batch_documents, batch_metadatas in batched_records(
        ids, documents, metadatas, batch_size=batch_size
    ):
        write_batch(
            ids=batch_ids,
            documents=batch_documents,
            metadatas=batch_metadatas,
        )
        total_indexed += len(batch_ids)

    return total_indexed
