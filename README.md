# Conversation Processing System

A sophisticated RAG-based (Retrieval-Augmented Generation) system for processing conversations with real-time classification, context-aware responses, and answer verification.

## System Architecture

The system consists of several key components:

### Backend Components
- **API Layer** (`api.py`): FastAPI-based REST API with health monitoring
- **Conversation Agent** (`conversation_agent.py`): Core logic for processing queries
- **Classification** (`classify.py`): Real-time conversation topic classification
- **RAG Pipeline**: Context retrieval and answer generation using Pinecone and DeepSeek
- **Cache Manager** (`utils/cache_manager.py`): LRU caching with TTL for optimized performance

### Frontend Components
- React-based UI for conversation interaction
- Real-time response display with classification and verification status
- Health monitoring dashboard

## Prerequisites

- Docker and Docker Compose
- At least 4GB RAM
- API Keys:
  - Pinecone (for vector storage)
  - Together AI (for LLM access)

## Quick Start with Docker

1. **Clone the Repository**
   ```bash
   git clone https://github.com/rrlcs/conversation-system.git
   cd conversation-system
   ```

2. **Set Up Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENV=your_pinecone_environment
   PINECONE_INDEX=your_pinecone_index_name
   TOGETHER_API_KEY=your_together_api_key
   ```

3. **Build and Start Services**
   ```bash
   # Build and start in detached mode
   docker-compose up --build -d

   # View logs
   docker-compose logs -f

   # Stop services
   docker-compose down
   ```

4. **Access the Application**
   - Frontend: http://localhost:3000
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## API Endpoints

### Process Question
- **URL**: `/question`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
      "question": "Your question here"
  }
  ```
- **Response**:
  ```json
  {
      "question": "Original question",
      "answer": "Generated answer",
      "context": {
          "retrieval_results": [
              {"text": "Retrieved context", "score": 0.95}
          ],
          "prompt_text": "Used prompt",
          "additional_context": {"previous_classification": "Category"}
      },
      "classification": "Conversation category",
      "verification_result": true/false
  }
  ```

### Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Response**: Status of all system components

## Performance Optimizations

1. **Caching System**
   - LRU cache for classifications and context searches
   - Configurable TTL (default: 1 hour)
   - Cache statistics monitoring

2. **Parallel Processing**
   - Concurrent classification and context search
   - Asynchronous verification
   - Response streaming

3. **Resource Management**
   - Docker volume caching for models and pip packages
   - Automatic cleanup of expired cache entries
   - Memory-efficient batch processing

## Development Guide

### Local Development Setup

1. **Create Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

2. **Run API Locally**
   ```bash
   uvicorn api:app --reload --port 8000
   ```

3. **Run Frontend Locally**
   ```bash
   cd frontend
   npm install
   npm start
   ```

### Debugging

1. **View Logs**
   ```bash
   # View API logs
   docker logs conversation-system-api-1

   # View frontend logs
   docker logs conversation-system-frontend-1
   ```

2. **Check System Health**
   ```bash
   curl http://localhost:8000/health
   ```

3. **Monitor Cache Performance**
   ```bash
   curl http://localhost:8000/health | jq .components.cache
   ```

## Configuration

Key configuration files:
- `config.py`: System-wide settings
- `docker-compose.yml`: Container orchestration
- `Dockerfile`: API container setup
- `frontend/Dockerfile`: Frontend container setup

### Important Settings
- `ScalabilityConfig` in `config.py`:
  - `CLASSIFICATION_TIMEOUT`: 20 seconds
  - `CONTEXT_SEARCH_TIMEOUT`: 20 seconds
  - `ANSWER_GENERATION_TIMEOUT`: 25 seconds
  - `VERIFICATION_TIMEOUT`: 20 seconds
  - `BATCH_TIMEOUT`: 2.0 seconds

## Troubleshooting

1. **API Container Won't Start**
   - Check environment variables in `.env`
   - Verify API keys are valid
   - Check logs: `docker logs conversation-system-api-1`

2. **Slow Response Times**
   - Monitor cache hit rates
   - Check Pinecone query latency
   - Verify network connectivity

3. **Memory Issues**
   - Adjust Docker container memory limits
   - Monitor cache size
   - Check for memory leaks in logs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request
