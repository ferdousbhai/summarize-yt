import os

from modal import App, Image, Secret, Volume

import logging

logging.basicConfig(level=logging.INFO)

app = App("summarize-youtube")

volume = Volume.from_name("youtube-data", create_if_missing=True)

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

sqlmodel_image = Image.debian_slim(python_version="3.12").run_commands(
    "pip install sqlmodel"
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

    yt = YouTube(f"https://youtu.be/{video_id}")
    return {
        "title": yt.title,
        "author": yt.author,
        "length": yt.length,
        "thumbnail_url": yt.thumbnail_url,
        "publish_date": yt.publish_date,
    }


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


def create_chunks(text_segments: list[str], max_tokens: int = 4095 * 0.8) -> list[str]:
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
def summarize(chunk: str, model: str = "gpt-4o") -> list[dict]:
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
            "content": "Summarize the main ideas and key points in the style of Paul Graham's essays from the following captions snippet. Focus solely on the ideas and lessons, not on the speakers or individuals. Ensure the text is concise, memorable, and engaging. Avoid prepositional phrases. If encounter product placement or other advertisement, exclude them. Each key idea should be broken down into a seperate section with its own title and description. Each section should be relatively short, under 300 words. Note that the provided captions might be part of a longer video so do not rush to conclusion. The reader's time is extremely valuable, if some section doesn't have high-quality content, exclude them. The final output must have very high signal:noise.",
        },
        {"role": "user", "content": chunk},
    ]
    client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    sections = client.chat.completions.create(
        model=model,
        response_model=list[Section],
        messages=messages,
        temperature=0.2,
    )
    return [section.dict() for section in sections]


@app.function(image=sqlmodel_image, volumes={"/youtube_data": volume})
def run(
    url: str, model: str = "gpt-4o", load_from_db: bool = True, save_to_db: bool = True
) -> str:
    """Summarize a YouTube video and save the summary to a database."""
    from datetime import datetime
    from sqlmodel import Field, SQLModel, create_engine, Session, select
    from itertools import chain

    class Video(SQLModel, table=True):
        video_id: str = Field(primary_key=True)
        title: str
        author: str
        length: int
        thumbnail_url: str
        publish_date: datetime
        captions: str
        summary: str
        created_at: datetime = Field(default_factory=datetime.now)

    def initialize_database():
        """Initialize the database and create tables if they don't exist."""
        engine = create_engine("sqlite:////youtube_data/youtube.db")
        SQLModel.metadata.create_all(engine)
        return engine

    def get_video(engine, video_id: str) -> Video | None:
        """Check if the video already exists in the database."""
        with Session(engine) as session:
            statement = select(Video).where(Video.video_id == video_id)
            video = session.exec(statement).first()
            if video:
                return video

    def save_video(engine, video: Video):
        """Save the video summary to the database."""
        with Session(engine) as session:
            session.add(video)
            session.commit()

    video_id = get_youtube_video_id(url)
    if not video_id:
        return "Invalid YouTube URL."

    engine = initialize_database()

    if load_from_db:
        video = get_video(engine, video_id)
        if video:
            logging.info("Loaded summary from db.")
            return video.summary

    captions = get_youtube_video_captions.remote(video_id)
    if not captions:
        return "No captions available for this video."
    logging.info("Loaded captions from YouTube.")

    sentences = get_sentences.remote(captions)
    logging.info(f"{len(sentences)} sentence(s) found.")
    chunks = create_chunks(sentences)
    logging.info(f"Summarizing {len(chunks)} chunk(s).")
    summaries = list(chain.from_iterable(summarize.map(chunks)))
    summary_text = "\n\n".join(
        f"## {section['title']}\n{section['text']}" for section in summaries
    )

    video = Video(
        video_id=video_id,
        captions=captions,  # raw_captions
        summary=summary_text,
        chunks=chunks,  # chunk embeddings? For Q&A
        chunk_summaries=summaries,
        **get_youtube_video_info.remote(video_id),
    )

    if save_to_db:
        save_video(engine, video)
        logging.info("Saved summary to db.")

    return summary_text


################################################################################
# Testing
################################################################################


# modal run summarize_yt::test_get_youtube_video_info --url https://www.youtube.com/watch\?v\=QIsLZTXHFYY
@app.local_entrypoint()
def test_get_youtube_video_info(url: str):
    video_id = get_youtube_video_id(url)
    print(get_youtube_video_info.remote(video_id))


# modal run summarize_yt::test_summarize_chunk --chunk "This is a really long piece of test chunk."
@app.local_entrypoint()
def test_summarize_chunk(chunk: str):
    print(summarize.remote(chunk))


# modal run summarize_yt::test_summarize_yt --url https://www.youtube.com/watch\?v\=QIsLZTXHFYY
@app.local_entrypoint()
def test_summarize_yt(url: str):
    video = run.remote(url)
    print(video)
