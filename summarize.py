from modal import App, Image, Secret

app = App("summarize-youtube")

youtube_image = Image.debian_slim(python_version="3.12").run_commands(
    "pip install youtube_transcript_api"
)

nlp_image = Image.debian_slim(python_version="3.12").run_commands(
    "pip install spacy",
    "python -m spacy download en_core_web_lg",
)

openai_image = Image.debian_slim(python_version="3.12").run_commands(
    "pip install openai"
)

tiktoken_image = Image.debian_slim(python_version="3.12").run_commands(
    "pip install tiktoken"
)


@app.function()
def get_youtube_video_id(url: str) -> str | None:
    """Helper function to get the video ID from a YouTube URL."""
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


@app.function(image=youtube_image)
def get_youtube_video_captions(url: str) -> str:
    """Use this function to get captions from a YouTube video."""
    from youtube_transcript_api import YouTubeTranscriptApi

    if not url:
        return "No URL provided"

    try:
        video_id = get_youtube_video_id.remote(url)
    except Exception as e:
        return (
            f"Error getting video ID from URL, please provide a valid YouTube url: {e}"
        )

    try:
        captions = YouTubeTranscriptApi.get_transcript(video_id)
        if captions:
            return " ".join(line["text"] for line in captions)
        return "No captions found for video"
    except Exception as e:
        return f"Error getting captions for video: {e}"


@app.function(image=nlp_image)
def get_sentences(text: str) -> list[str]:
    """Get sentences from a string."""
    import spacy

    nlp = spacy.load("en_core_web_lg")
    return [sent.text for sent in nlp(text).sents]


@app.function(image=tiktoken_image)
def tokenize(text: str, encoding_name: str = "cl100k_base") -> list[str]:
    """Tokenize a string."""
    import tiktoken

    return tiktoken.get_encoding(encoding_name).encode(text)


@app.function()
def combine_sentences(sentences: list[str], max_tokens: int = 4095 * 0.9) -> list[str]:
    """
    Combines sentences into larger chunks without exceeding a specified token count.
    """
    output_chunks = []
    current_chunk = ""
    current_chunk_token_count = 0

    sentence_tokens = tokenize.map(sentences)

    for sentence, sentence_tokens in zip(sentences, sentence_tokens):
        sentence_token_count = len(sentence_tokens)
        if current_chunk_token_count + sentence_token_count > max_tokens:
            output_chunks.append(current_chunk)
            current_chunk = sentence
            current_chunk_token_count = sentence_token_count
        else:
            current_chunk += sentence
            current_chunk_token_count += sentence_token_count
    if current_chunk:
        output_chunks.append(current_chunk)
    return output_chunks


@app.function(image=openai_image, secrets=[Secret.from_name("openai")])
def summarize_chunk(chunk: str) -> str:
    """Summarize a chunk of text."""
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You're given a transcript of a chunk of video. Rewrite this text in summarized form, including the main ideas and key points. The summary should be written in a style that's engaging and memorable. You should not focus on the speakers, only the ideas and key points.",
            },
            {"role": "user", "content": chunk},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


@app.function(image=openai_image, secrets=[Secret.from_name("openai")])
def combine_chunks(accumulated_summaries: list[str]) -> str:
    """Given a list of summaries, create a final summary of the video."""
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You're given summaries of a video in chunks. Rewrite the key ideas and key points in a style that's engaging and memorable. The text should have educational value, concise. You should not focus on the speakers, only the ideas and key points.",
            },
            {"role": "user", "content": "\n\n".join(accumulated_summaries)},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


@app.function()
def summarize_video(text_chunks: list[str], model: str = "gpt-4o"):
    accumulated_summaries: list[str] = list(summarize_chunk.map(text_chunks))
    if len(accumulated_summaries) > 1:
        return combine_chunks.remote(accumulated_summaries)
    return accumulated_summaries[0]


# 3. Each chunk should be summarized -> Should be done in parallell
# 4. The summaries should be combined into a single summary for each chapter
# 5. Unit test everything
# 6. Build a react front-end
# 7. Add a database for storing the summaries by video id
# 8. Run through $500 budget
# 9. Add a subscription model

################################################################################
# Testing
################################################################################

YT_LINK = "https://www.youtube.com/watch?v=7DegEwUQzSA"


@app.local_entrypoint()
def test_get_sentences():
    link = YT_LINK
    transcript = get_youtube_video_captions.remote(link)
    sentences = get_sentences.remote(transcript)
    print(f"{len(sentences)} sentences found")
    for sentence in sentences:
        print(sentence)
    # save sentences to file
    with open(".data/sentences.txt", "w") as f:
        for line in sentences:
            f.write(line + "\n")


@app.local_entrypoint()
def test_combine_sentences():  # TODO: investigate "Task was destroyed but it is pending!"
    import pickle

    # load sentences from file:
    with open(".data/sentences.txt", "r") as f:
        sentences = f.readlines()
    chunks = combine_sentences.remote(sentences)
    print(f"{len(chunks)} chunks found")

    print(chunks[0])

    # save the chunks locally
    with open(".data/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


@app.local_entrypoint()
def test_chunk_token_limits():
    """Test the token limits for the chunking function."""
    import tiktoken
    import pickle

    with open(".data/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    encoding = tiktoken.get_encoding("cl100k_base")

    for chunk in chunks:
        tokens = encoding.encode(chunk)
        assert len(tokens) <= 4095, f"Chunk exceeds 4095 tokens: {len(tokens)} tokens"

    print("All chunks are within the token limit.")


@app.local_entrypoint()
def test_summarize_video():
    import pickle

    with open(".data/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    summaries = summarize_video.remote(chunks)

    with open(".data/summaries.txt", "w") as f:
        f.write(summaries)

    print(summaries)
