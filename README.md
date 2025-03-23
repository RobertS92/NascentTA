
# Semantic Sentence De-duplication for Streaming Data

## Overview

This project implements a **scalable, real-time semantic sentence de-duplication system** for streaming text data. It is designed to handle **high-throughput streams** where identifying **semantically similar sentences** quickly and efficiently is critical. The system is optimized for **low-latency performance**, **memory efficiency**, and **streaming scalability**, making it suitable for production-level streaming applications.

---

## Core Design & Approach

Key components and design decisions:

1. **Sentence Encoding with SentenceTransformer**  
   Utilizes pre-trained transformer models to convert sentences into dense semantic embeddings.

2. **Fast Similarity Search with Faiss**  
   Faiss enables rapid similarity comparisons, crucial for real-time performance.

3. **FIFO Buffer with Deque**  
   Maintains a fixed-size sliding window of recent sentence embeddings for efficient streaming processing.

4. **Asynchronous Processing with Asyncio**  
   Ensures non-blocking, concurrent handling of incoming sentences and similarity checks.

5. **Configurable Parameters**  
   Allows for flexible tuning (e.g., buffer size, similarity threshold) to adapt to different application needs.

---

## Buffer Management & Trade-offs

The system uses a **deque with a maximum length** as a sliding buffer. When full, the oldest entries are removed (FIFO logic), and the **Faiss index is updated accordingly**.

### Buffer Size Considerations:
- **Small Buffers (10s–100s):**  
  High speed, lower memory, may miss duplicates over time.
  
- **Medium Buffers (1,000s):**  
  Good trade-off between duplicate detection and speed.

- **Large Buffers (10,000+):**  
  High duplicate catch rate, increased memory and processing overhead.

---

## Streaming Assumptions

- **Input Language:** English (can be extended).
- **Special Characters:** Emojis and domain-specific characters (e.g., hashtags) are preserved.
- **Stream Timing:** Variable input intervals (1ms to 1s) with potential bursts.
- **Robustness:** Handles occasional malformed inputs with graceful error handling.

---

## Scaling & Optimization Ideas

1. **Distributed Architecture:**  
   Integrate with tools like **Apache Kafka** and **distributed processing frameworks** for scaling across nodes.

2. **GPU Acceleration:**  
   Use GPUs for **faster embedding generation** and similarity computation.

3. **Efficient Faiss Updates:**  
   Explore **batch index updates** or **periodic index rebuilding** for performance.

4. **Caching Layer:**  
   Cache frequently processed sentences to avoid redundant computations.

5. **Persistent Storage:**  
   Incorporate **distributed databases** for long-term storage and querying of embeddings at scale.

---

## Balancing Accuracy vs. Speed

- **Similarity Threshold (Configurable):**  
  Tuning this adjusts sensitivity to duplicates and affects processing speed.

- **Faiss Search Efficiency:**  
  Enables fast nearest-neighbor search over large datasets.

- **Batch Processing Support:**  
  Processes multiple inputs at once for better throughput.

- **Asynchronous Design:**  
  Ensures smooth performance during spikes in streaming data.

---

## Integration Possibilities

Designed to be easily integrated into larger systems:
1. **Message Queue Input:**  
   Compatible with **Kafka, RabbitMQ**, etc., for input streams.

2. **Microservice Deployment:**  
   Container-ready (Docker/Kubernetes) for deployment as a service.

3. **API Gateway Compatibility:**  
   Can be exposed via RESTful endpoints for flexible access.

4. **Monitoring:**  
   Supports integration with **Prometheus, ELK stack**, etc.

5. **Output Streams:**  
   Processed sentences can be emitted to a queue or database for downstream tasks.

---

## Runtime Flexibility (Future Work)

Potential for **live parameter updates** via:
- Config services (e.g., Consul, etcd)
- Messaging signals for dynamic reconfiguration
- Hot-reload mechanisms for real-time parameter changes

---

## Dataset & Stream Simulation

- **Data Used:**  
  Combination of **crypto-related Reddit posts (from Kaggle)** and **synthetic data** generated using open-source language models.

- **Why Crypto Data?**  
  Crypto-specific language often includes **emojis, hashtags, and domain-specific slang**, making it ideal to test semantic similarity robustness.

- **Simulation:**  
  Built using **Python’s asyncio** with variable sleep intervals (mimicking real-world stream activity). Included simulated **bursts** for stress testing.

- **Preprocessing Steps:**
  1. Lowercasing
  2. Emoji & hashtag preservation
  3. Punctuation removal (except emojis/hashtags)
  4. Tokenization
  5. Stop word removal
  6. Stemming (emojis/hashtags excluded)

---

## Performance Testing

Performance was evaluated using:
1. **Timing Decorators:**  
   To log processing latency per sentence.

2. **Moving Averages:**  
   For real-time performance tracking.

3. **Buffer Size Tests:**  
   Benchmarked at sizes 100, 1,000, 10,000+.

4. **Stream Load Simulations:**  
   Tested under varying input velocities to gauge system resilience.

---

## Project Purpose

This project was developed as a **portfolio piece** to showcase **stream processing, NLP, and scalable system design**. It highlights real-time **semantic analysis**, efficient **similarity search**, and **production-oriented architecture** for handling streaming text data.

---
