# Import necessary libraries
import os
import asyncio
import random
import logging
import sys
from collections import deque
from typing import List
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import emoji
import multiprocessing
from tqdm import tqdm
import faiss

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('deduper.log')
    ]
)
logger = logging.getLogger(__name__)

# Define settings class for configuration management
class Settings(BaseSettings):
    # Configuration parameters with default values
    BUFFER_SIZE: int = 1000
    SIMILARITY_THRESHOLD: float = 0.9
    MODEL_NAME: str = 'all-MiniLM-L6-v2'
    MAX_SENTENCE_LENGTH: int = 1000
    BATCH_SIZE: int = 32
    CSV_FILENAME: str = "/Users/rcs713/Downloads/NascentTA/additional_data.csv"
    COLUMN_NAME: str = "title"
    MAX_WAIT_TIME: float = 2.0
    MIN_WAIT_TIME: float = 0.01
    SPIKE_PROBABILITY: float = 0.1
    SPIKE_MAX_WAIT: float = 0.01
    SPIKE_MIN_WAIT: float = 0.001
    INDEX_UPDATE_FREQUENCY: int = 1000

    model_config = SettingsConfigDict(env_file=".env")

# Cache the settings to avoid repeated disk reads
@lru_cache()
def get_settings():
    return Settings()

# Preprocess a single sentence
def preprocess_sentence(sentence: str) -> str:
    sentence = emoji.demojize(sentence)  # Convert emojis to text
    sentence = sentence.lower()  # Convert to lowercase
    sentence = re.sub(r'[^\w\s:#]', '', sentence)  # Remove special characters except '#' and ':'
    tokens = word_tokenize(sentence)  # Tokenize the sentence
    stop_words = set(stopwords.words('english'))  # Get English stop words
    # Remove stop words but keep emoji representations and hashtags
    tokens = [word for word in tokens if word not in stop_words or word.startswith(':') or word.startswith('#')]
    stemmer = PorterStemmer()  # Initialize stemmer
    # Apply stemming to words that are not emoji representations or hashtags
    tokens = [stemmer.stem(word) if not (word.startswith(':') or word.startswith('#')) else word for word in tokens]
    return ' '.join(tokens)  # Join tokens back into a sentence

# Process a chunk of sentences (used for multiprocessing)
def process_chunk(chunk):
    return [preprocess_sentence(sentence) for sentence in chunk if isinstance(sentence, str)]

# Main class for semantic deduplication
class SemanticDeduper:
    def __init__(self, similarity_threshold: float = None):
        self.settings = get_settings()
        self.model = SentenceTransformer(self.settings.MODEL_NAME)
        self.similarity_threshold = similarity_threshold or self.settings.SIMILARITY_THRESHOLD
        self.processed_count = 0
        self.duplicate_count = 0
        self.buffer = deque(maxlen=self.settings.BUFFER_SIZE)  # FIFO buffer
        self.index = None  # Faiss index for similarity search
        self.embedding_to_sentence = {}  # Mapping of embeddings to sentences

    # Initialize the Faiss index
    def initialize_index(self, dim):
        self.index = faiss.IndexFlatIP(dim)

    # Process a single sentence
    async def process_sentence(self, sentence: str) -> bool:
        try:
            self.processed_count += 1

            # Truncate long sentences
            if len(sentence) > self.settings.MAX_SENTENCE_LENGTH:
                logger.warning(f"Sentence exceeds maximum length: {len(sentence)} > {self.settings.MAX_SENTENCE_LENGTH}")
                sentence = sentence[:self.settings.MAX_SENTENCE_LENGTH]

            preprocessed = preprocess_sentence(sentence)
            embedding = self.model.encode([preprocessed])[0]

            if self.index is None:
                self.initialize_index(embedding.shape[0])

            is_duplicate = self._check_similarity(embedding)

            if is_duplicate:
                self.duplicate_count += 1
            else:
                # Implement FIFO behavior when buffer is full
                if len(self.buffer) >= self.settings.BUFFER_SIZE:
                    removed_sentence = self.buffer.popleft()
                    removed_embedding = self.embedding_to_sentence.pop(removed_sentence)
                    self._remove_from_index(removed_embedding)

                self.buffer.append(preprocessed)
                self.index.add(embedding.reshape(1, -1))
                self.embedding_to_sentence[preprocessed] = embedding

            await self.update_index()

            return is_duplicate
        except Exception as e:
            logger.error(f"Error processing sentence: {e}", exc_info=True)
            raise

    # Remove an embedding from the Faiss index
    def _remove_from_index(self, embedding):
        D, I = self.index.search(embedding.reshape(1, -1), self.index.ntotal)
        if D[0][0] > 0.9999:  # Threshold for considering it the same vector
            self.index.remove_ids(np.array([I[0][0]]))

    # Process a batch of sentences
    async def process_sentences(self, sentences: List[str]):
        # Preprocess sentences using multiprocessing
        preprocessed = []
        with multiprocessing.Pool() as pool:
            chunk_size = 10000
            chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
            preprocessed = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks), desc="Preprocessing"))
        
        preprocessed = [item for sublist in preprocessed for item in sublist]
        
        self.processed_count = len(preprocessed)
        
        # Process sentences in batches
        batch_size = self.settings.BATCH_SIZE
        for i in tqdm(range(0, len(preprocessed), batch_size), desc="Processing batches"):
            batch = preprocessed[i:i+batch_size]
            embeddings = self.model.encode(batch, show_progress_bar=False)
            
            if self.index is None:
                self.initialize_index(embeddings.shape[1])
            
            for j, embedding in enumerate(embeddings):
                if self.index.ntotal == 0:
                    self.buffer.append(batch[j])
                    self.index.add(embedding.reshape(1, -1))
                    self.embedding_to_sentence[batch[j]] = embedding
                else:
                    is_duplicate = self._check_similarity(embedding)
                    if not is_duplicate:
                        if len(self.buffer) >= self.settings.BUFFER_SIZE:
                            removed_sentence = self.buffer.popleft()
                            removed_embedding = self.embedding_to_sentence.pop(removed_sentence)
                            self._remove_from_index(removed_embedding)

                        self.buffer.append(batch[j])
                        self.index.add(embedding.reshape(1, -1))
                        self.embedding_to_sentence[batch[j]] = embedding
                    else:
                        self.duplicate_count += 1

            await self.update_index()

    # Periodically update the Faiss index
    async def update_index(self):
        if self.processed_count % self.settings.INDEX_UPDATE_FREQUENCY == 0:
            new_index = faiss.IndexFlatIP(self.index.d)
            new_index.add(self.index.reconstruct_n(0, self.index.ntotal))
            self.index = new_index

    # Check if an embedding is similar to any in the index
    def _check_similarity(self, embedding: np.ndarray) -> bool:
        if self.index.ntotal > 0:
            D, _ = self.index.search(embedding.reshape(1, -1), 1)
            return D[0][0] > self.similarity_threshold
        return False

    # Get metrics about the deduplication process
    def get_metrics(self):
        return {
            "processed_count": self.processed_count,
            "duplicate_count": self.duplicate_count,
            "unique_count": len(self.buffer),
            "unique_ratio": 1 - (self.duplicate_count / self.processed_count) if self.processed_count > 0 else 1
        }

    # Get the list of unique sentences
    def get_unique_sentences(self):
        return list(self.buffer)

# Initialize FastAPI app
app = FastAPI()
deduper = SemanticDeduper()

# API endpoint to process a single sentence
@app.post("/process")
async def process_sentence(sentence: str):
    try:
        is_duplicate = await deduper.process_sentence(sentence)
        return {"sentence": sentence, "is_duplicate": is_duplicate}
    except Exception as e:
        logger.error(f"Error processing sentence: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to get current metrics
@app.get("/metrics")
async def get_metrics():
    return deduper.get_metrics()

# API endpoint to get unique sentences
@app.get("/sentences")
async def get_sentences():
    return {"unique": deduper.get_unique_sentences()}

# Function to read sentences from a CSV file
def read_sentences_from_csv(filename: str, column_name: str) -> List[str]:
    try:
        df = pd.read_csv(filename, usecols=[column_name], encoding='utf-8', on_bad_lines='skip')
        sentences = df[column_name].dropna().tolist()
        logger.info(f"Read {len(sentences)} sentences from {filename}")
        return sentences
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}", exc_info=True)
        return []

# Simulate a stream of sentences with random wait times
async def simulate_stream(sentences):
    settings = get_settings()
    for sentence in sentences:
        wait_time = random.uniform(settings.MIN_WAIT_TIME, settings.MAX_WAIT_TIME)
        if random.random() < settings.SPIKE_PROBABILITY:
            wait_time = random.uniform(settings.SPIKE_MIN_WAIT, settings.SPIKE_MAX_WAIT)
        await asyncio.sleep(wait_time)
        yield sentence

# Process a stream of sentences
async def process_stream(deduper, stream):
    async for sentence in stream:
        is_duplicate = await deduper.process_sentence(sentence)
        logger.info(f"Processed: '{sentence[:30]}...', Duplicate: {is_duplicate}")

# Run a simulation of the deduplication process
async def run_simulation(sentences: List[str]):
    stream = simulate_stream(sentences)
    await process_stream(deduper, stream)
    
    logger.info("Simulation complete. Results:")
    logger.info(f"Total sentences processed: {deduper.processed_count}")
    logger.info(f"Duplicate sentences found: {deduper.duplicate_count}")
    logger.info(f"Unique sentences: {len(deduper.buffer)}")
    logger.info(f"\nFinal Metrics: {deduper.get_metrics()}")

# Main execution
if __name__ == "__main__":
    import uvicorn
    import time
    
    settings = get_settings()
    sentences = read_sentences_from_csv(settings.CSV_FILENAME, settings.COLUMN_NAME)

    if sentences:
        start_time = time.time()
        asyncio.run(run_simulation(sentences))
        end_time = time.time()
        logger.info(f"Total runtime: {end_time - start_time:.2f} seconds")
    else:
        logger.error("No sentences were read from the CSV file. Exiting.")

    # Uncomment the following line to run as a FastAPI server
    # uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
