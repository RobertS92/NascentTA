# NascentTA
Repo for Nascent technical assignment

# Semantic Sentence De-duper for Streaming Data

## Approach and Design Decisions

This system aims to implement a scalable semantic sentence de-duplication solution for streaming data. Key design decisions include:

1. Use of SentenceTransformer for semantic encoding: This efficiently converts sentences into semantic embeddings.
2. Faiss for similarity search: Provides fast and efficient similarity comparisons, important for maintaining low latency.
3. FIFO buffer implementation: So that the system can handle a continuous stream of data while maintaining a fixed buffer size.
4. Asynchronous processing: Allows for efficient data stream handling and potential spikes in activity.
5. Configurable parameters: Enables easy tuning of the system for different use cases and performance requirements.

## Handling Different Buffer Sizes

The system uses a deque with a configurable maximum length for the buffer. When the buffer reaches its maximum size, the oldest sentences are removed to make room for new ones (FIFO principle). The Faiss index is updated accordingly to maintain consistency with the buffer.

Performance characteristics for different buffer sizes:
- Small buffers (tens to hundreds): Fast processing, but may miss some semantic duplicates over a longer time frame (would need improvement for production).
- Medium buffers (thousands): Good balance between processing speed and duplicate detection over time.
- Large buffers (tens of thousands): More comprehensive duplicate detection, but may impact processing speed and memory usage.

## Assumptions about the Incoming Stream

1. The stream consists of text sentences in a single language (English in this implementation).
2. Sentences may contain emojis and special characters which is important to domain-specific crypto data.
3. The stream has varying intervals between messages (1ms to 1s) with occasional spikes in activity.
4. The incoming data is generally well-formed, but the system includes error handling for robustness (thinking with production in mind).

## Potential Optimizations for Scaling

1. Distributed processing: Implement a distributed version using technologies like Apache Kafka for message queuing and multiple worker nodes for processing.
2. GPU acceleration: Utilize GPU for sentence encoding and similarity computations.
3. Optimized index updates: Implement more efficient strategies for updating the Faiss index, such as batch updates or periodic rebuilding.
4. Caching: Implement a caching layer for frequently seen or processed sentences.
5. Database integration: For very large scale operations, consider integrating a distributed database for storing and querying embeddings.

## Balancing Accuracy and Processing Speed

The system balances accuracy and speed through:
1. Configurable similarity threshold: Allows tuning the trade-off between detecting more potential duplicates and processing speed.
2. Efficient similarity search with Faiss: Provides fast similarity computations even for large datasets.
3. Batch processing: Allows for efficient processing of multiple sentences at once, improving throughput.
4. Asynchronous operations: Enables the system to handle incoming sentences without blocking.

## Integration into a Larger Streaming Data System

This component can be integrated into a larger streaming data system as follows:

1. Message Queue Integration: Use Apache Kafka or RabbitMQ to feed sentences into the de-duper.
2. Microservices Architecture: Deploy the de-duper as a microservice in a containerized environment (e.g., Kubernetes).
3. API Gateway: Implement an API gateway for routing requests and load balancing.
4. Monitoring and Logging: Integrate with systems like Prometheus and ELK stack for monitoring and log analysis.
5. Output Stream: Implement an output stream for processed sentences, potentially using another message queue.

## Runtime Parameter Changes

While not implemented in this version, runtime parameter changes could be handled by:
1. Implementing a configuration service that the de-duper periodically checks for updates.
2. Using a message-based system to signal configuration changes to the de-duper.
3. Implementing a "hot reload" mechanism that can update parameters without restarting the service.

## Dataset and Stream Simulation

For this implementation, we used [describe your dataset here]. The stream was simulated using Python's asyncio library, with random wait times between sentences to mimic real-world conditions. The simulation includes occasional "spikes" of activity with shorter wait times.

Preprocessing steps include:
1. Lowercasing
2. Removing punctuation while preserving emojis and hashtags
3. Tokenization
4. Removing stop words
5. Stemming (except for emojis and hashtags to account for crypto-specific context)

## Performance Testing

To measure performance:
1. Implement timing decorators on key methods to measure processing time.
2. Log latency for each processed sentence and calculate moving averages.
3. Test with different buffer sizes (e.g., 100, 1000, 10000) and report average latency and duplicate detection rates.
4. Simulate different stream velocities and measure system performance under various loads.

## Data
1. I used a combination of crypto-related Reddit posts from Kaggle and also synthetic data generated from open-source models to get more robust data to work with
2. I chose this type of data because I found it more relevant to simulate the actual use case of the position


