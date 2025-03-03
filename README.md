# Conversation Processing System

A RAG-based system for processing conversations with classification and verification.

## Features

- Processes conversation logs using RAG (Retrieval-Augmented Generation)
- Classifies conversations by topic in real-time
- Verifies answer accuracy against context
- Provides detailed analytics
- Dockerized for easy deployment
- RESTful API interface

## Prerequisites

- Docker and Docker Compose
- Environment variables:
  - PINECONE_API_KEY
  - PINECONE_ENV
  - PINECONE_INDEX
  - TOGETHER_API_KEY

## Setup

1. Clone the repository
2. Create a `.env` file with required environment variables:
```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
PINECONE_INDEX=your_pinecone_index_name
TOGETHER_API_KEY=your_together_api_key
```

3. Build and start the services:
```bash
docker-compose up --build
```

## API Endpoints

### Process Question
- **URL**: `/process`
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
        "search_results": "Retrieved context",
        "prompt": "Used prompt"
    },
    "classification": "Conversation category",
    "verification_result": true/false,
    "analytics": {
        "message_count": 1,
        "current_classification": "Category",
        "conversation_length": 1,
        "classification_changes": 0
    }
}
```

### Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Response**: `{"status": "healthy"}`

## Scaling

The system is designed for horizontal scaling:
1. Multiple API instances can be deployed behind a load balancer
2. Pinecone handles vector search scaling
3. Docker Compose can be configured for multiple replicas

## Development

To run locally without Docker:
1. Install dependencies: `pip install -r requirements.txt`
2. Run the API: `uvicorn api:app --reload`

## Testing

Access the API documentation at `http://localhost:8000/docs` after starting the server. 