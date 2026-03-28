import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence

from car_data_etl import prepare_car_reviews_data
from chroma_utils import build_chroma_collection, get_collection

DEFAULT_DATA_GLOB = str(Path("data/archive") / "*.csv")
CHROMA_PATH = Path("car_review_embeddings")
EMBEDDING_FUNC_NAME = "multi-qa-MiniLM-L6-cos-v1"
COLLECTION_NAME = "car_reviews"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_CONFIG_PATH = Path("config.json")
DEFAULT_RESULT_COUNT = 10
DEFAULT_OFFLINE_REVIEW_LIMIT = 5

SYSTEM_PROMPT = """You are a customer success employee at a large car dealership.
Answer the user's question using only the following car reviews.
If the reviews do not contain enough information, say that clearly.

Car reviews:
{reviews}
"""


def load_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Optional[str]]:
    settings: dict[str, Any] = {}

    if config_path.exists():
        try:
            settings = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"El archivo de configuración {config_path} no contiene JSON válido."
            ) from exc

        if not isinstance(settings, dict):
            raise ValueError(
                f"El archivo de configuración {config_path} debe contener un objeto JSON."
            )

    api_key = (
        os.getenv("OPENAI_API_KEY")
        or settings.get("openai-secret-key")
        or settings.get("openai_api_key")
    )
    model = (
        os.getenv("OPENAI_MODEL")
        or settings.get("openai-model")
        or settings.get("openai_model")
        or DEFAULT_CHAT_MODEL
    )

    return {"api_key": api_key, "model": model}


def build_context(reviews: Sequence[str]) -> str:
    cleaned_reviews = [review.strip() for review in reviews if review and review.strip()]

    if not cleaned_reviews:
        raise ValueError("No hay reviews disponibles para construir el contexto.")

    reviews_block = "\n".join(f"- {review}" for review in cleaned_reviews)
    return SYSTEM_PROMPT.format(reviews=reviews_block)


def build_openai_client(api_key: str) -> Any:
    try:
        import httpx
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Faltan dependencias para usar OpenAI. Ejecutá `pip install -r requirements.txt`."
        ) from exc

    return OpenAI(api_key=api_key, http_client=httpx.Client(timeout=30.0))


def shorten_text(text: str, max_length: int = 280) -> str:
    normalized_text = " ".join(text.split())
    if len(normalized_text) <= max_length:
        return normalized_text

    return normalized_text[: max_length - 3].rstrip() + "..."


def build_offline_answer(
    question: str, reviews: Sequence[str], reason: Optional[str] = None
) -> str:
    cleaned_question = question.strip()
    visible_reviews = [review for review in reviews if review and review.strip()]

    if not visible_reviews:
        raise RuntimeError("No hay reviews disponibles para responder en modo local.")

    lines = [
        "Modo local activado: no se llamó a OpenAI.",
        f'Pregunta: "{cleaned_question}"',
    ]

    if reason:
        lines.append(f"Motivo: {reason}")

    lines.append("Reviews recuperadas:")
    for index, review in enumerate(
        visible_reviews[:DEFAULT_OFFLINE_REVIEW_LIMIT], start=1
    ):
        lines.append(f"{index}. {shorten_text(review)}")

    if len(visible_reviews) > DEFAULT_OFFLINE_REVIEW_LIMIT:
        lines.append(
            f"Mostrando {DEFAULT_OFFLINE_REVIEW_LIMIT} de {len(visible_reviews)} reviews relevantes."
        )

    return "\n".join(lines)


def retrieve_reviews(question: str, results_count: int = DEFAULT_RESULT_COUNT) -> list[str]:
    if results_count <= 0:
        raise ValueError("La cantidad de resultados debe ser mayor a cero.")

    try:
        collection = get_collection(CHROMA_PATH, COLLECTION_NAME, EMBEDDING_FUNC_NAME)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "No se pudo abrir la colección local de ChromaDB."
        ) from exc

    try:
        filtered_results = collection.query(
            query_texts=[question],
            n_results=results_count,
            include=["documents"],
            where={"Rating": {"$gte": 3}},
        )

        documents = (filtered_results.get("documents") or [[]])[0]
        if documents:
            return documents

        fallback_results = collection.query(
            query_texts=[question],
            n_results=results_count,
            include=["documents"],
        )
        return (fallback_results.get("documents") or [[]])[0]
    except Exception as exc:
        raise RuntimeError(
            "No se pudieron recuperar reviews desde ChromaDB. Verificá que la colección exista y que el modelo de embeddings esté disponible localmente."
        ) from exc


def ask_question(
    question: str,
    *,
    model: str,
    openai_client: Any,
    documents: Sequence[str],
) -> str:
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("La pregunta no puede estar vacía.")

    if not documents:
        raise RuntimeError(
            "No se encontraron reviews relevantes. Ejecutá primero `python3 app.py ingest`."
        )

    try:
        completion = openai_client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": build_context(documents)},
                {"role": "user", "content": cleaned_question},
            ],
        )
    except Exception as exc:
        raise RuntimeError(f"No se pudo completar la consulta con OpenAI: {exc}") from exc

    answer = completion.choices[0].message.content
    if not answer:
        raise RuntimeError("OpenAI devolvió una respuesta vacía.")

    return answer.strip()


def generate_chat_response(
    question: str,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    model_override: Optional[str] = None,
    results_count: int = DEFAULT_RESULT_COUNT,
    offline: bool = False,
) -> str:
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("La pregunta no puede estar vacía.")

    documents = retrieve_reviews(cleaned_question, results_count=results_count)
    if not documents:
        raise RuntimeError(
            "No se encontraron reviews relevantes. Ejecutá primero `python3 app.py ingest`."
        )

    if offline:
        return build_offline_answer(
            cleaned_question,
            documents,
            reason="Se solicitó explícitamente el modo offline.",
        )

    settings = load_settings(config_path)
    api_key = settings.get("api_key")
    model = model_override or settings.get("model") or DEFAULT_CHAT_MODEL

    if not api_key:
        return build_offline_answer(
            cleaned_question,
            documents,
            reason="No se encontró una API key de OpenAI.",
        )

    try:
        openai_client = build_openai_client(api_key)
        return ask_question(
            cleaned_question,
            model=model,
            openai_client=openai_client,
            documents=documents,
        )
    except RuntimeError as exc:
        return build_offline_answer(cleaned_question, documents, reason=str(exc))


def ingest_reviews(
    *,
    years: Sequence[int],
    data_path: str = DEFAULT_DATA_GLOB,
    reset: bool = True,
    batch_size: int = 166,
) -> int:
    chroma_car_reviews_dict = prepare_car_reviews_data(data_path, years)
    return build_chroma_collection(
        CHROMA_PATH,
        COLLECTION_NAME,
        EMBEDDING_FUNC_NAME,
        chroma_car_reviews_dict["ids"],
        chroma_car_reviews_dict["documents"],
        chroma_car_reviews_dict["metadatas"],
        reset=reset,
        batch_size=batch_size,
    )


def run_interactive_chat(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    model_override: Optional[str] = None,
    results_count: int = DEFAULT_RESULT_COUNT,
    offline: bool = False,
) -> None:
    if offline:
        print("Chat listo en modo local. No se llamará a OpenAI.")
    else:
        print(
            "Chat listo. Si OpenAI no está disponible, te mostraré el contexto recuperado localmente."
        )

    while True:
        try:
            question = input("\nAsk me: ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\nInterrumpido por el usuario.")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit", "q"}:
            break

        try:
            print(
                generate_chat_response(
                    question,
                    config_path=config_path,
                    model_override=model_override,
                    results_count=results_count,
                    offline=offline,
                )
            )
        except Exception as exc:  # pragma: no cover - mantiene vivo el loop interactivo
            print(f"Error: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI para indexar reviews de autos en ChromaDB y consultarlas con OpenAI."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Ruta al archivo de configuración JSON con la API key de OpenAI.",
    )

    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser(
        "ingest", help="Construye o actualiza la colección vectorial."
    )
    ingest_parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2017],
        help="Años de vehículos a indexar.",
    )
    ingest_parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_GLOB,
        help="Ruta o patrón glob a los CSV del dataset.",
    )
    ingest_parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Hace upsert sobre la colección existente en lugar de recrearla.",
    )
    ingest_parser.add_argument(
        "--batch-size",
        type=int,
        default=166,
        help="Cantidad de documentos por lote al insertar en ChromaDB.",
    )

    chat_parser = subparsers.add_parser(
        "chat", help="Inicia un chat interactivo o responde una sola pregunta."
    )
    chat_parser.add_argument(
        "--question",
        help="Si se informa, responde una sola pregunta y termina.",
    )
    chat_parser.add_argument(
        "--model",
        help="Modelo de OpenAI a usar. Si no se informa, se toma de OPENAI_MODEL o config.json.",
    )
    chat_parser.add_argument(
        "--results",
        type=int,
        default=DEFAULT_RESULT_COUNT,
        help="Cantidad máxima de reviews recuperadas desde ChromaDB.",
    )
    chat_parser.add_argument(
        "--offline",
        action="store_true",
        help="No llama a OpenAI y devuelve solo el contexto recuperado localmente.",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "chat"
    question = getattr(args, "question", None)
    model = getattr(args, "model", None)
    results = getattr(args, "results", DEFAULT_RESULT_COUNT)
    offline = getattr(args, "offline", False)

    try:
        if command == "ingest":
            total_reviews = ingest_reviews(
                years=args.years,
                data_path=args.data_path,
                reset=not args.keep_existing,
                batch_size=args.batch_size,
            )
            print(f"Colección actualizada con {total_reviews} reviews.")
            return

        if command == "chat":
            if question:
                print(
                    generate_chat_response(
                        question,
                        config_path=args.config,
                        model_override=model,
                        results_count=results,
                        offline=offline,
                    )
                )
                return

            run_interactive_chat(
                config_path=args.config,
                model_override=model,
                results_count=results,
                offline=offline,
            )
            return

        parser.error(f"Comando desconocido: {command}")
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    main()
