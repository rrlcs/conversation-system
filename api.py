from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from conversation_agent import ConversationAgent
import os
from dotenv import load_dotenv
import traceback
import asyncio
from contextlib import asynccontextmanager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global variable for the agent
agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the agent on startup
    global agent
    try:
        logger.info("Initializing ConversationAgent...")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_env = os.getenv("PINECONE_ENV")
        pinecone_index = os.getenv("PINECONE_INDEX")
        together_api_key = os.getenv("TOGETHER_API_KEY")
        
        if not all([pinecone_api_key, pinecone_env, pinecone_index, together_api_key]):
            missing_vars = [var for var, val in {
                "PINECONE_API_KEY": pinecone_api_key,
                "PINECONE_ENV": pinecone_env,
                "PINECONE_INDEX": pinecone_index,
                "TOGETHER_API_KEY": together_api_key
            }.items() if not val]
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Initialize agent with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                agent = ConversationAgent(
                    pinecone_api_key=pinecone_api_key,
                    pinecone_env=pinecone_env,
                    index_name=pinecone_index,
                    together_api_key=together_api_key
                )
                logger.info("ConversationAgent initialized successfully")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Attempt {attempt + 1} failed, retrying in 2 seconds...")
                await asyncio.sleep(2)
        
        # Load initial conversations with timeout
        try:
            logger.info("Loading initial conversations...")
            async with asyncio.timeout(120):  # 2 minutes timeout for initial load
                await agent.load_conversations('text_1.json')
            logger.info("Initial conversations loaded successfully")
        except asyncio.TimeoutError:
            logger.error("Timeout while loading conversations")
            raise RuntimeError("Timeout while loading initial conversations")
        except Exception as e:
            logger.error(f"Failed to load initial conversations: {str(e)}")
            raise RuntimeError(f"Failed to load initial conversations: {str(e)}")
        
        yield
        
    except Exception as e:
        logger.error(f"Error initializing agent: {str(e)}", exc_info=True)
        agent = None
        yield
    finally:
        # Cleanup on shutdown
        if agent:
            logger.info("Cleaning up agent resources...")
            try:
                await agent.cleanup()
                logger.info("Agent cleanup completed successfully")
            except Exception as e:
                logger.error(f"Error during agent cleanup: {str(e)}", exc_info=True)

app = FastAPI(
    title="Conversation Processing System",
    description="A RAG-based system for processing conversations with classification and verification",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to process")

    class Config:
        schema_extra = {
            "example": {
                "question": "What is the weather like today?"
            }
        }

class ContextInfo(BaseModel):
    retrieval_results: List[Dict[str, Any]] = Field(..., description="Results retrieved from the vector database")
    prompt_text: str = Field(..., description="The prompt text used with the language model")
    additional_context: Optional[Dict[str, Any]] = None

class ConversationResponse(BaseModel):
    question: str = Field(..., description="The original question asked by the user")
    answer: str = Field(..., description="The generated answer")
    context: ContextInfo = Field(..., description="All contextual information passed to the LLM")
    classification: str = Field(..., description="The agent's topic classification")
    verification_result: bool = Field(..., description="The verification result (True/False)")

    class Config:
        schema_extra = {
            "example": {
                "question": "What is the weather like today?",
                "answer": "I don't have access to real-time weather information.",
                "context": {
                    "retrieval_results": [
                        {"text": "Previous conversation about weather", "score": 0.95}
                    ],
                    "prompt_text": "You are an AI assistant. The user asked about weather...",
                    "additional_context": {"time": "2024-02-20T12:00:00Z"}
                },
                "classification": "Casual Chat",
                "verification_result": True
            }
        }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    error_msg = f"An error occurred: {str(exc)}"
    error_detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print(f"Error detail:\n{error_detail}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": error_msg}
    )

@app.get("/")
async def root():
    """Root endpoint to verify API is running"""
    return {
        "status": "online",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "Root endpoint"},
            {"path": "/health", "method": "GET", "description": "Health check endpoint"},
            {"path": "/question", "method": "POST", "description": "Process a question using the conversation agent"},
            {"path": "/load_conversations", "method": "POST", "description": "Load initial conversations"}
        ]
    }

@app.post("/question", response_model=ConversationResponse)
async def process_question(request: QuestionRequest):
    """Process a question using the conversation agent"""
    logger.info(f"Received question request: {request.question}")
    
    if not agent:
        logger.error("Conversation agent not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation agent not properly initialized. Check environment variables and connections."
        )
    
    try:
        # Process the question using our agent with timeout
        logger.info("Processing question with agent...")
        
        # Set a timeout for the entire operation
        async with asyncio.timeout(30):  # 30 seconds timeout
            try:
                logger.info("Starting agent.process_query...")
                response = await agent.process_query(request.question)
                logger.info("agent.process_query completed successfully")
            except asyncio.TimeoutError:
                logger.error("Timeout occurred during agent.process_query")
                raise
            except Exception as e:
                logger.error(f"Error in agent.process_query: {str(e)}", exc_info=True)
                raise
        
        logger.info("Question processed successfully, formatting response...")
        
        # Format the context information according to the API spec
        search_results = response.context.get("search_results", "")
        logger.debug(f"Raw search results type: {type(search_results)}")
        
        # Convert string search results into list format if needed
        retrieval_results = []
        if isinstance(search_results, str):
            logger.debug("Converting string search results to list format")
            # Split by the separator used in _format_context
            conversations = search_results.split("-" * 40)
            for conv in conversations:
                if conv.strip():
                    retrieval_results.append({"text": conv.strip(), "score": None})
        else:
            logger.debug("Using raw search results as retrieval_results")
            retrieval_results = search_results
        
        logger.debug(f"Creating ContextInfo with {len(retrieval_results)} results")
        context_info = ContextInfo(
            retrieval_results=retrieval_results,
            prompt_text=response.context.get("prompt", ""),
            additional_context={
                "previous_classification": response.previous_classification
            }
        )
        
        # Convert response to JSON-serializable format
        logger.debug("Creating ConversationResponse")
        response_data = ConversationResponse(
            question=response.question,
            answer=response.answer,
            context=context_info,
            classification=response.classification,
            verification_result=response.verification_result
        )
        
        logger.info("Successfully formatted response")
        return jsonable_encoder(response_data)
    
    except asyncio.TimeoutError as e:
        logger.error("Request timed out after 30 seconds", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timed out after 30 seconds. The operation took too long to complete."
        )
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Check the health of the API and its components."""
    try:
        if agent is None:
            logger.error("Health check failed: Agent is None")
            return JSONResponse(
                status_code=503,
                content={"status": "error", "error": "Agent is None"}
            )
        
        # Test vector search connection using run_in_executor for the synchronous method
        test_query = "test"
        search_result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: agent.vector_search.search(test_query, top_k=1)
        )
        
        # Get memory statistics
        memory_stats = {
            "total_messages": agent.memory.get_message_count(),
            "max_entries": agent.memory.max_entries,
            "cleanup_threshold": agent.memory.cleanup_threshold
        }
        
        # Check if conversations are loaded by checking vector search results
        conversations_loaded = len(search_result) > 0
        if not conversations_loaded:
            logger.warning("No conversations found in vector search")
        
        logger.info("Health check completed successfully")
        return {
            "status": "healthy",
            "components": {
                "vector_search": "connected" if search_result else "error",
                "conversations_loaded": conversations_loaded,
                "memory": memory_stats,
                "cache": agent.cache_manager.get_cache_stats()
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error": str(e),
                "type": type(e).__name__
            }
        )

@app.post("/load_conversations")
async def load_conversations():
    """Load initial conversations."""
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        logger.info("Loading initial conversations...")
        async with asyncio.timeout(60):  # 1 minute timeout
            await agent.load_conversations('text_1.json')
        logger.info("Initial conversations loaded successfully")
        return {"status": "success", "message": "Conversations loaded successfully"}
    except asyncio.TimeoutError:
        logger.error("Timeout while loading conversations")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Timeout while loading conversations"
        )
    except Exception as e:
        logger.error(f"Error loading conversations: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading conversations: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") 