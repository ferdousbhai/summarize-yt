import os

from modal import App, Image, Secret

import logging

logging.basicConfig(level=logging.INFO)

app = App("summarize-youtube")

youtube_transcript_api_image = Image.debian_slim(python_version="3.12").run_commands(
    "pip install youtube_transcript_api"
)

pytube_image = Image.debian_slim(python_version="3.12").run_commands(
    "pip install pytube"
)

nlp_image = Image.debian_slim(python_version="3.12").run_commands(
    "pip install spacy",
    "python -m spacy download en_core_web_lg",
)

ai_image = Image.debian_slim(python_version="3.12").run_commands(
    "pip install openai instructor"
)

tiktoken_image = Image.debian_slim(python_version="3.12").run_commands(
    "pip install tiktoken"
)

db_image = Image.debian_slim(python_version="3.12").run_commands(
    "pip install sqlmodel psycopg2-binary"
)


def get_youtube_video_id(url: str) -> str | None:
    """Get the video ID from a YouTube URL."""
    from urllib.parse import urlparse, parse_qs

    parsed_url = urlparse(url)
    hostname = parsed_url.hostname

    if hostname == "youtu.be":
        return parsed_url.path[1:]
    if hostname in ("www.youtube.com", "youtube.com"):
        if parsed_url.path == "/watch":
            query_params = parse_qs(parsed_url.query)
            return query_params.get("v", [None])[0]
        if parsed_url.path.startswith("/embed/"):
            return parsed_url.path.split("/")[2]
        if parsed_url.path.startswith("/v/"):
            return parsed_url.path.split("/")[2]
    return None


@app.function(image=youtube_transcript_api_image)
def get_youtube_video_captions(video_id: str) -> str | None:
    """Use this function to get captions from a YouTube video."""
    from youtube_transcript_api import YouTubeTranscriptApi

    try:
        captions = YouTubeTranscriptApi.get_transcript(video_id)
        if captions:
            return " ".join(line["text"] for line in captions)
    except Exception as e:
        logging.error(f"Error getting captions for video: {e}")
    return None


@app.function(image=pytube_image)
def get_youtube_video_info(video_id: str) -> dict:
    """Get the video info from a YouTube video."""
    from pytube import YouTube

    try:
        yt = YouTube(f"https://youtu.be/{video_id}")
        video_info = {
            "title": yt.title,
            "author": yt.author,
            "length": yt.length,
            "thumbnail_url": yt.thumbnail_url,
            "publish_date": yt.publish_date,
        }
        return video_info
    except Exception as e:
        logging.error(f"Error getting video info for video: {e}")
        return {}


@app.function(image=nlp_image)
def get_sentences(text: str) -> list[str]:
    """Get sentences from a string."""
    import spacy

    nlp = spacy.load("en_core_web_lg")
    return [sent.text for sent in nlp(text).sents]


@app.function(image=tiktoken_image)
def get_token_counts(
    input_list: list[str], encoding_name: str = "cl100k_base"
) -> list[int]:
    """Count tokens in input texts."""
    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)
    return [len(encoding.encode(text)) for text in input_list]


def create_chunks(text_segments: list[str], max_tokens: int = 4095 * 0.95) -> list[str]:
    """
    Combines text_segments into larger chunks without exceeding a specified token count.
    """
    output_chunks = []
    current_chunk = ""
    current_chunk_token_count = 0

    token_counts = get_token_counts.remote(text_segments)

    for text_segment, token_count in zip(text_segments, token_counts):
        if current_chunk_token_count + token_count > max_tokens:
            output_chunks.append(current_chunk)
            current_chunk = text_segment
            current_chunk_token_count = token_count
        else:
            current_chunk += "\n\n" + text_segment
            current_chunk_token_count += token_count
    if current_chunk:
        output_chunks.append(current_chunk)
    return output_chunks


@app.function(image=ai_image, secrets=[Secret.from_name("openai")])
def summarize(
    chunk: str, model: str = "gpt-4o", temperature: float = 0.8
) -> list[dict]:
    """Summarize a chunk of text using OpenAI."""
    import instructor
    from pydantic import BaseModel
    from openai import OpenAI

    class Section(BaseModel):
        title: str
        text: str

    messages = [
        {
            "role": "system",
            "content": "Summarize the main ideas and key points of the following captions snippet. Ensure the text is concise, memorable, and engaging. Avoid prepositional phrases. If encounter product placement or other advertisement, exclude them. Each key idea should be broken down into a seperate section with its own title and description. Each section should be relatively short, under 300 words. Note that the provided captions might be part of a longer video so do not rush to conclusion. The reader's time is extremely valuable, so if some section doesn't have high-quality content, exclude them. The final output must have very high signal:noise.",
        },
        {"role": "user", "content": chunk},
    ]
    client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    sections = client.chat.completions.create(
        model=model,
        response_model=list[Section],
        messages=messages,
        temperature=temperature,
    )
    return [section.dict() for section in sections]


@app.function(image=db_image, secrets=[Secret.from_name("video-data")])
def run(
    url: str, model: str = "gpt-4o", load_from_db: bool = True, save_to_db: bool = True
) -> str:
    """Summarize a YouTube video and save the summary to a database."""
    from datetime import datetime
    from sqlmodel import Field, SQLModel, create_engine, Session, select
    from sqlalchemy.exc import SQLAlchemyError
    from uuid import uuid4
    from itertools import chain

    class Summary(SQLModel, table=True):
        id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
        video_id: str = Field(foreign_key="video.video_id")
        content: str
        author: str

    class Video(SQLModel, table=True):
        video_id: str = Field(primary_key=True)
        title: str
        author: str
        length: int
        thumbnail_url: str
        publish_date: datetime
        caption_sentences: str
        created_at: datetime = Field(default_factory=datetime.now)

    def initialize_database():
        """Initialize the database and create tables if they don't exist."""
        engine = create_engine(os.environ["DATABASE_URL_DEV"])
        SQLModel.metadata.create_all(engine)
        return engine

    def save_to_db(
        engine, video_id: str, sentences: list[str], summary_content: str, model: str
    ):
        """Save the video and summary to the database."""
        try:
            with Session(engine) as session:
                video_info = get_youtube_video_info.remote(video_id)
                video = Video(
                    video_id=video_id,
                    **video_info,
                    caption_sentences="\n".join(sentences),
                )
                summary = Summary(
                    video_id=video_id, content=summary_content, author=model
                )

                session.add(video)
                session.add(summary)
                session.commit()
                logging.info("Saved video and summary to db.")
        except SQLAlchemyError as e:
            logging.error(f"Error saving to the database: {e}")
            session.rollback()
        except Exception as e:
            logging.error(f"Unexpected error: {e}")

    def get_summaries(engine, video_id: str) -> list[Summary] | None:
        """Check if the summary already exists in the database."""
        try:
            with Session(engine) as session:
                statement = select(Summary).where(Summary.video_id == video_id)
                summaries = session.exec(statement).all()
                if summaries:
                    logging.info("Loaded summary from db.")
                    return summaries
        except SQLAlchemyError as e:
            logging.error(f"Error getting summary from the database: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return None

    video_id = get_youtube_video_id(url)
    if not video_id:
        return "Invalid YouTube URL."

    engine = initialize_database()

    if load_from_db:
        summaries = get_summaries(engine, video_id)
        if summaries:
            return summaries[0].content  # v1 of the app supports only one summary

    captions = get_youtube_video_captions.remote(video_id)
    if not captions:
        return "No captions available for this video."
    logging.info("Loaded captions from YouTube.")

    sentences = get_sentences.remote(captions)
    logging.info(f"{len(sentences)} sentence(s) found.")
    chunks = create_chunks(sentences)
    logging.info(f"Summarizing {len(chunks)} chunk(s).")
    chunk_summaries = list(chain.from_iterable(summarize.map(chunks)))
    combined_summary_md = "\n\n".join(
        f"## {section['title']}\n{section['text']}" for section in chunk_summaries
    )

    if save_to_db:
        save_to_db(engine, video_id, sentences, combined_summary_md, model)
    return combined_summary_md


################################################################################
# Testing
################################################################################


# modal run summarize_yt::test_get_youtube_video_info --url ...
@app.local_entrypoint()
def test_get_youtube_video_info(url: str):
    video_id = get_youtube_video_id(url)
    print(get_youtube_video_info.remote(video_id))


# modal run summarize_yt::test_summarize_chunk --chunk "..."
@app.local_entrypoint()
def test_summarize_chunk(chunk: str):
    print(summarize.remote(chunk))


# modal run summarize_yt::test_summarize_yt --url ...
@app.local_entrypoint()
def test_summarize_yt(url: str):
    video = run.remote(url)
    print(video)
