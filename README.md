# vectorized-context-rag

Chatbot de consola para hacer RAG sobre reviews de autos usando embeddings, ChromaDB y OpenAI. El dataset incluido proviene de [Kaggle](https://www.kaggle.com/datasets/ankkur13/edmundsconsumer-car-ratings-and-reviews) y la idea original del ejercicio está inspirada en el artículo de Real Python sobre ChromaDB.

El flujo del proyecto ahora quedó separado en dos pasos:

1. `ingest`: procesa los CSV, genera embeddings e indexa las reviews en ChromaDB.
2. `chat`: consulta la colección vectorial y usa OpenAI para responder preguntas con ese contexto. Si no hay API key, no hay conectividad o se usa `--offline`, devuelve el contexto recuperado localmente.

## Requisitos

- Python 3.9 o superior
- Un entorno virtual recomendado
- API key de OpenAI solo para respuestas generadas por el LLM

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuración

Podés usar variables de entorno:

```bash
export OPENAI_API_KEY="tu-openai-api-key"
export OPENAI_MODEL="gpt-4o-mini"
```

O crear un `config.json` a partir de `config.example.json`:

```json
{
  "openai-secret-key": "tu-openai-api-key",
  "openai-model": "gpt-4o-mini"
}
```

Si no tenés créditos de OpenAI, igual podés ejecutar `ingest` para explorar la parte de ETL y vectorización. Además, `chat` puede funcionar en modo local mostrando las reviews más relevantes sin llamar a OpenAI.

## Uso

Construir la colección vectorial:

```bash
python3 app.py ingest
```

Indexar varios años:

```bash
python3 app.py ingest --years 2016 2017 2018
```

Reutilizar una colección existente sin recrearla:

```bash
python3 app.py ingest --keep-existing
```

Abrir el chat interactivo:

```bash
python3 app.py chat
```

Abrir el chat en modo local, sin llamadas a OpenAI:

```bash
python3 app.py chat --offline
```

Hacer una sola pregunta y salir:

```bash
python3 app.py chat --question "What do owners think about the Volkswagen New Beetle?"
```

## Estructura

- `app.py`: CLI principal para `ingest` y `chat`
- `car_data_etl.py`: limpieza y preparación del dataset
- `chroma_utils.py`: creación y acceso a la colección de ChromaDB
- `data/archive/`: CSV originales del dataset

## Mejoras aplicadas

- Se corrigió el batching de inserción en ChromaDB para no perder documentos al final de cada lote.
- Se migró la integración de OpenAI al cliente moderno.
- Se separó la indexación del chat para evitar reconstruir la base en cada ejecución.
- Se agregó un modo offline y fallback automático a contexto local cuando OpenAI no está disponible.
- Se mejoró el manejo de configuración y errores.
- Se simplificó `requirements.txt` a dependencias directas.
