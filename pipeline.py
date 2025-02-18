
import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings, VectorSearchVectorStoreDatastore
from google.cloud import storage

# Set a similarity threshold.
# Here, higher scores mean more similar (with 1 being identical).
SIMILARITY_THRESHOLD = 0.9

# Helper: Get existing chunk IDs from Cloud Storage for a given video_id
def get_existing_chunk_ids(video_id, bucket_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    prefix = f"{video_id}_chunk_"
    blobs = list(bucket.list_blobs(prefix=prefix))
    ids = []
    for blob in blobs:
        # Expect blob names like "videoid_chunk_0", "videoid_chunk_1", etc.
        try:
            number_part = blob.name.split("_chunk_")[1]
            ids.append(int(number_part))
        except Exception:
            continue
    return ids

# Step 1: Fetch YouTube Transcript for a given video URL
def fetch_youtube_transcript(video_url):
    video_id = video_url.split("v=")[-1]
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        print(f"Could not retrieve transcript for video {video_url}. Skipping. Error: {e}")
        return None
    formatter = TextFormatter()
    plain_text = formatter.format_transcript(transcript)
    return plain_text

# Step 2: Preprocess the Text
def preprocess_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'(?<=\w)\n(?=\w)', ' ', text)
    return text

# Step 3: Create Semantic Chunks from text
def create_semantic_chunks(text, chunk_size=350, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Step 4: Store Chunks in the Vector Store with Duplicate Check, Extended Metadata, and Auto-Incremented IDs
def store_embeddings_in_vector_store(chunks, video_url):
    # Define Vertex AI and vector store parameters
    PROJECT_ID = "sample-client-5464560543"
    REGION = "us-central1"
    INDEX_ID = "---------5356757312"
    ENDPOINT_ID = "------10756756736"
    EMBEDDING_MODEL = "text-embedding-005"
    BUCKET = "sample_rag_bucket_001"

    # Extract video_id from the URL
    video_id = video_url.split("v=")[-1]

    # Get existing chunk IDs from the Cloud Storage bucket for this video_id
    existing_ids = get_existing_chunk_ids(video_id, BUCKET)
    if existing_ids:
        next_index = max(existing_ids) + 1
    else:
        next_index = 0

    # Create embeddings instance and rebuild if necessary
    embedding_model = VertexAIEmbeddings(model=EMBEDDING_MODEL)
    embedding_model.model_rebuild()

    # Initialize the vector store
    vector_store = VectorSearchVectorStoreDatastore.from_components(
        project_id=PROJECT_ID,
        region=REGION,
        index_id=INDEX_ID,
        gcs_bucket_name=BUCKET,
        endpoint_id=ENDPOINT_ID,
        embedding=embedding_model,
        stream_update=True,
    )

    new_texts = []
    new_metadatas = []
    new_ids = []
    skipped_count = 0

    for i, chunk in enumerate(chunks):
        # Compute embedding for the current chunk.
        new_embedding = embedding_model.embed_documents([chunk])[0]

        # Use similarity search by vector with score.
        results = vector_store.similarity_search_by_vector_with_score(new_embedding, k=1)
        metadata = {
            "text": chunk,
            "chunk_text": chunk,
            "chunk_index": i,
            "source_link": video_url
        }
        print(f"\nChunk {i} metadata:")
        print(metadata)

        if results:
            existing_doc, score = results[0]
            print(f"Similarity score (cosine similarity): {score:.3f}")
            if score > SIMILARITY_THRESHOLD:
                print("Existing similar chunk found:")
                print(existing_doc.page_content)
                print("=> This chunk will be SKIPPED.\n")
                skipped_count += 1
                continue
            else:
                print("No sufficiently similar chunk found; this chunk will be inserted.\n")
        else:
            print("No similar documents returned. Inserting new chunk.\n")

        # Assign a new unique id based on video_id and next_index.
        new_id = f"{video_id}_chunk_{next_index}"
        next_index += 1

        new_texts.append(chunk)
        new_metadatas.append(metadata)
        new_ids.append(new_id)

    if new_texts:
        vector_store.add_texts(new_texts, metadatas=new_metadatas, ids=new_ids)
        print(f"\nInserted {len(new_texts)} new datapoints into the vector store (skipped {skipped_count} duplicates).")
    else:
        print("\nNo new datapoints to insert; all chunks appear to be duplicates.")

# Process a single video URL through the full pipeline
def process_video(video_url):
    print(f"\n=== Processing video: {video_url} ===")
    print("Fetching transcript...")
    transcript = fetch_youtube_transcript(video_url)
    
    # If no transcript is found, skip further processing for this video.
    if transcript is None:
        print(f"Skipping video {video_url} due to missing transcript.\n")
        return

    print("Preprocessing transcript...")
    processed_text = preprocess_text(transcript)

    print("Creating semantic chunks...")
    chunks = create_semantic_chunks(processed_text)
    print(f"Created {len(chunks)} chunks.")

    print("Storing embeddings in the vector store with duplicate check and enhanced metadata...")
    store_embeddings_in_vector_store(chunks, video_url)
    print(f"=== Finished processing video: {video_url} ===\n")

# Main execution function for multiple video links
def main():
    video_urls = ["https://www.youtube.com/watch?v=aa528jbZDeI","https://www.youtube.com/watch?v=n2LEZcyBlkA","https://www.youtube.com/watch?v=cm8ax_ejv5o"]
    for url in video_urls:
        process_video(url)

if __name__ == "__main__":
    main()

