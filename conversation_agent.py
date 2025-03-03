# Building a Conversation Processing System with Retrieval-Augmented Generation (RAG) and AI Agent
# Primary Objective
# To develop a system that:
# 1. Processes conversation logs by converting them into word vectors and storing these vectors in a vector database.
# 2. Retrieves and generates answers using a RAG (Retrieval-Augmented Generation) framework in response to user queries.
# 3. Classifies conversations by topic in real time.
# 4. Verifies the correctness of retrieved answers.
# 5. Desing a simple AI Agent task flow that incorporates with requirements provided.
# Detailed Requirements
# 1. Data Processing and Vectorization
# • Input:
# o Conversation logs provided in a readable format (e.g., JSON, CSV).
# • Action:
# o Use Python to parse and process these logs.
# o Select an appropriate word/vector model (e.g., from Hugging Face).
# o Convert the conversation logs into embeddings (word or sentence vectors).
# • Output:
# o Store the resulting embeddings in a structured vector database for fast retrieval.
 
# 2. RAG System Configuration
# • Setup:
# o Configure the RAG system so it can query the vector database for relevant information.
# • Integration:
# o Create prompts that incorporate both the user query and retrieved context from the vector database.
# o Connect this process with a large language model (e.g., ChatGPT API) to generate the final answer.
# • Functionality:
# o Handle various query types and produce contextually relevant answers.
# o Ensure the system gracefully handles edge cases or irrelevant queries.
 
# 3. Conversation Classification
# • Agent Role:
# o Implement an agent to classify conversations into predefined categories, such as:
# ▪ Casual Chat
# ▪ Discussing Hobbies
# ▪ Discussing Work
# ▪ Discussing Personal Matters
# • Real-time Processing:
# o The agent should continuously monitor ongoing messages, updating the conversation's classification category as it evolves.
# • Framework Usage:
# ▪ The agent implementation must leverage an existing AI agent framework (e.g., LangChain, CrewAi,Llama Index or similar).
# ▪ This framework should handle the agent's state, conversation flow, classification logic, and integration with RAG.
# ▪ If the chosen framework lacks built-in retrieval features, you may integrate it with an external vector database and LLM API for the RAG flow.
 
# 4. Answer Verification
# • Post-Retrieval Action:
# o After the RAG system retrieves an answer, the agent should verify its relevance and accuracy.
# • Verification Mechanism:
# o Ensure the final answer aligns with the retrieved context.
# o Alert or flag any discrepancies or low-confidence results.
 
# 5. System Packaging and API Interface
# • Dockerization:
# o Package the entire application (data processing, RAG, classification, answer verification) in a Docker image.
# o Simplify deployment and enable horizontal scaling as data volume or usage grows.
# • API Specification:
# o Local API should accept a user's question and return:
# ▪ The question itself.
# ▪ The generated answer.
# ▪ All contextual information passed to the large language model (including retrieval results and prompt text).
# ▪ The agent's topic classification.
# ▪ The agent's verification result (validating correctness).
# Additional Notes
# • Performance Considerations:
# o Optimize database queries for speed (search in embeddings).
# o Minimize latency in the RAG loop to provide quick responses.
# • Scalability:
# o Ensure the system can handle increasing data volumes and serve multiple concurrent users without significant performance degradation.
# • Security:
# o Apply best practices for data protection, especially if handling private or sensitive conversation logs.
# o Consider encryption, access control, and compliance requirements.

import os
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Together
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableSequence
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, Tool, AgentType, initialize_agent
from langchain.chains import LLMChain
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from pinecone import Pinecone
from deepseek import DeepSeekLLM
from dotenv import load_dotenv
from services.vector_search import VectorSearchService
from config import ScalabilityConfig
from utils.cache_manager import CacheManager
from utils.batch_processor import BatchRequestHandler
from utils.memory import ScalableConversationMemory
import asyncio
import logging
from classify import Classify

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# Load environment variables from .env file
load_dotenv()

@dataclass
class ConversationResponse:
    question: str
    answer: str
    context: Dict
    classification: str
    verification_result: bool
    previous_classification: str

class ConversationAgent:
    def __init__(
        self,
        pinecone_api_key: str,
        pinecone_env: str,
        index_name: str,
        together_api_key: str
    ):
        try:
            # Initialize scalability components
            self.cache_manager = CacheManager()
            self.batch_handler = BatchRequestHandler()
            
            # Initialize classifier
            self.classifier = Classify()
            
            # Initialize vector search service with improved configuration
            self.vector_search = VectorSearchService(
                pinecone_api_key=pinecone_api_key,
                pinecone_env=pinecone_env,
                index_name=index_name,
                cache_ttl=ScalabilityConfig.VECTOR_CACHE_TTL,
                batch_size=ScalabilityConfig.VECTOR_BATCH_SIZE,
                max_retries=3  # Default to 3 retries
            )
            
            # Initialize other components with scalability settings
            self.llm = DeepSeekLLM(
                api_key=together_api_key
            )
            
            # Initialize memory with cleanup threshold
            self.memory = self._setup_memory()
            
            # Setup other components
            self.tools = self._setup_tools()
            self.agent = self._setup_agent()
            
            # Initialize conversation state
            self.current_classification = None
            self.message_count = 0
            
            # Setup batch processors for common operations
            self._setup_batch_processors()
            
            print("ConversationAgent initialized successfully")
            
        except Exception as e:
            print(f"Error initializing ConversationAgent: {str(e)}")
            raise
    
    def _setup_memory(self) -> ScalableConversationMemory:
        """Setup memory with automatic cleanup."""
        return ScalableConversationMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output",
            input_key="input",
            max_entries=ScalabilityConfig.MAX_MEMORY_ENTRIES,
            cleanup_threshold=ScalabilityConfig.MEMORY_CLEANUP_THRESHOLD
        )
    
    def _setup_batch_processors(self):
        """Setup batch processors for common operations."""
        # Batch processor for embeddings
        self.embedding_processor = self.batch_handler.get_processor(
            "embeddings",
            self.vector_search.batch_embed
        )
        
        # Batch processor for classifications
        self.classification_processor = self.batch_handler.get_processor(
            "classifications",
            self._batch_classify_conversations
        )
        
    def _setup_tools(self) -> List[Tool]:
        """Setup LangChain tools for the agent."""
        return [
            Tool(
                name="classify_conversation",
                func=self._classify_conversation_tool,
                description="Classifies the conversation into categories like Casual Chat, Discussing Hobbies, Work, or Personal Matters"
            ),
            Tool(
                name="search_context",
                func=self._search_context_tool,
                description="Searches for relevant context in previous conversations"
            ),
            Tool(
                name="verify_answer",
                func=self._verify_answer_tool,
                description="Verifies if an answer is supported by the context"
            )
        ]
    
    def _setup_agent(self) -> AgentExecutor:
        """Setup the LangChain agent with proper prompt and tools."""
        # Define the agent prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an AI assistant that helps manage conversations and retrieve information.
            For each user query, you MUST follow these steps in order:
            1. Use the classify_conversation tool to determine the conversation category
            2. Use the search_context tool to find relevant information
            3. Generate an answer based on the context
            4. Use the verify_answer tool to verify your answer
            
            Always return your response in this format:
            {
                "classification": "category from classification tool",
                "context": "context from search tool",
                "answer": "your answer based on context",
                "verification": "result from verify tool"
            }
            
            If you're not sure about something, say "I don't have enough information to answer that question."
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Initialize the agent
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            agent_kwargs={"prompt": prompt}
        )
    
    def _classify_conversation_tool(self, conversation: str) -> str:
        """Tool for classifying conversations."""
        classification_prompt = PromptTemplate(
            input_variables=["conversation"],
            template="""Classify this conversation into one of these categories:
            - Casual Chat
            - Discussing Hobbies 
            - Discussing Work
            - Discussing Personal Matters
            
            Conversation: {conversation}
            Classification:"""
        )
        
        chain = classification_prompt | self.llm
        return chain.invoke({"conversation": conversation})
    
    def _search_context_tool(self, query: str) -> str:
        """Tool for searching relevant context with optimized vector search."""
        search_results = self.vector_search.search(
            query=query,
            top_k=10,
            filter_dict=None  # Add filters if needed
        )
        return self._format_context(search_results)
    
    def _verify_answer_tool(self, data: Dict[str, str]) -> bool:
        """Tool for verifying answers against context."""
        verify_prompt = PromptTemplate(
            input_variables=["query", "answer", "context"],
            template="""You are a fact checker who verifies if answers are supported by the context. Your job is to ensure answers make appropriate use of available information.

            Query: {query}
            Answer: {answer}
            Context: {context}
            
            Verification Rules:
            
            1. For Personal Information (names, gender, etc.):
               - If the answer states a fact that exactly matches the context, mark as TRUE
               - Example: If context shows "I'm a boy" and answer says "you are a boy", mark as TRUE
               - Names, dates, and personal details must match exactly
            
            2. For Food-Related Information:
               - If the answer mentions foods that appear in the context, mark as TRUE
               - Food suggestions based on past preferences are valid
               - Example: If they've had hot pot and barbecue, suggesting either is valid
            
            3. For Time/Date Information:
               - Timestamps must match exactly for specific events
               - Multiple occurrences of the same event (e.g., saying "I love you" twice) should all be mentioned
               - Example: If "I love you" was said at 21:10 and 21:12, mentioning both times is correct
            
            4. For Recommendations:
               - Suggestions based on explicitly mentioned preferences are valid
               - Combining multiple mentioned items into a new suggestion is valid
               - Example: If they like barbecue and turkey, suggesting a "barbecue turkey sandwich" is valid
            
            You MUST respond in this EXACT format:

            VERIFICATION PROCESS:
            [Your step-by-step verification explanation here]

            EVIDENCE:
            [Quote exact relevant parts from context]

            VERDICT: [True/False]

            Do not add any other text or explanation after the verdict line."""
        )
        
        chain = verify_prompt | self.llm
        result = chain.invoke(data)
        
        try:
            # Split the result into sections
            sections = result.strip().split('\n\n')
            
            # Find the verdict section (should be the last one)
            verdict_line = None
            for section in reversed(sections):
                if section.strip().startswith('VERDICT:'):
                    verdict_line = section.strip()
                    break
            
            if not verdict_line:
                logger.warning(f"No VERDICT section found in verification result: {result}")
                return False
                
            verdict = verdict_line.replace('VERDICT:', '').strip().lower()
            
            # Log the complete verification result for debugging
            logger.debug(f"Verification result for query: {data['query']}")
            logger.debug(f"Complete verification response:\n{result}")
            logger.debug(f"Final verdict: {verdict}")
            
            return verdict == 'true'
            
        except Exception as e:
            logger.error(f"Error parsing verification result: {str(e)}")
            logger.debug(f"Problematic verification response:\n{result}")
            return False
    
    async def process_query(self, query: str) -> ConversationResponse:
        """Process a user query using the agent framework with improved scalability."""
        try:
            logger.info("Starting process_query...")
            
            # Update message count
            self.message_count += 1
            logger.debug(f"Updated message count to {self.message_count}")
            
            # Get previous classification
            previous_classification = self.current_classification
            logger.debug(f"Previous classification: {previous_classification}")
            
            # Get classification using the agent's tool
            try:
                async with asyncio.timeout(ScalabilityConfig.CLASSIFICATION_TIMEOUT):
                    classification = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self._classify_conversation_tool,
                        query
                    )
                    logger.info(f"Classification completed: {classification}")
            except asyncio.TimeoutError:
                logger.error("Classification timed out")
                classification = "Unknown"
            except Exception as e:
                logger.error(f"Classification error: {str(e)}", exc_info=True)
                classification = "Unknown"
            
            # Search for context with caching
            context_key = f"context:{query}"
            context = self.cache_manager.get("search_results", context_key)
            logger.debug(f"Cache lookup for context: {'hit' if context else 'miss'}")
            
            if not context:
                logger.info("Searching for context...")
                try:
                    async with asyncio.timeout(ScalabilityConfig.CONTEXT_SEARCH_TIMEOUT):
                        context = await self._search_context_with_retry(query)
                        self.cache_manager.set("search_results", context_key, context)
                        logger.info("Context search completed")
                except asyncio.TimeoutError:
                    logger.error("Context search timed out")
                    context = "No context available due to timeout"
            
            # Generate answer
            logger.info("Generating answer...")
            try:
                async with asyncio.timeout(ScalabilityConfig.ANSWER_GENERATION_TIMEOUT):
                    answer = await self._generate_answer(query, context)
                    logger.info("Answer generated successfully")
                    logger.debug(f"Generated answer: {answer[:100]}...")
            except asyncio.TimeoutError:
                logger.error("Answer generation timed out")
                answer = "I apologize, but I couldn't generate an answer in time. Please try again."
            
            # Verify answer
            logger.info("Starting answer verification...")
            try:
                async with asyncio.timeout(ScalabilityConfig.VERIFICATION_TIMEOUT):
                    logger.debug("Preparing verification data...")
                    verification = await self._verify_answer_with_retry(query, answer, context)
                    logger.info(f"Answer verification completed: {verification}")
                    if not verification:
                        logger.warning(f"Answer verification failed for query: {query}")
                        logger.debug(f"Failed verification context length: {len(context)}")
                        logger.debug(f"Failed verification answer length: {len(answer)}")
            except asyncio.TimeoutError:
                logger.error(f"Answer verification timed out after {ScalabilityConfig.VERIFICATION_TIMEOUT} seconds for query: {query}")
                logger.debug(f"Verification timeout - Context length: {len(context)}")
                logger.debug(f"Verification timeout - Answer length: {len(answer)}")
                verification = False
            except Exception as e:
                logger.error(f"Unexpected error during verification: {str(e)}", exc_info=True)
                verification = False
            
            # Update state
            self.current_classification = classification
            logger.debug("Updating memory...")
            self._update_memory_with_cleanup(query, answer, classification)
            
            # Create and return response
            logger.info("Creating response object...")
            return ConversationResponse(
                question=query,
                answer=answer,
                context={"search_results": context, "prompt": str(self.agent.agent.llm_chain.prompt)},
                classification=classification,
                verification_result=verification,
                previous_classification=previous_classification
            )
        
        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}", exc_info=True)
            raise
    
    async def _search_context_with_retry(self, query: str, max_retries: int = 3) -> str:
        """Search for context with retry logic."""
        for attempt in range(max_retries):
            try:
                return await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._search_context_tool,
                    query
                )
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def _verify_answer_with_retry(
        self,
        query: str,
        answer: str,
        context: str,
        max_retries: int = 3
    ) -> bool:
        """Verify answer with retry logic."""
        for attempt in range(max_retries):
            try:
                return await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._verify_answer_tool,
                    {"query": query, "answer": answer, "context": context}
                )
            except Exception as e:
                if attempt == max_retries - 1:
                    return False
                await asyncio.sleep(2 ** attempt)
    
    def _update_memory_with_cleanup(self, query: str, answer: str, classification: str):
        """Update memory with automatic cleanup if needed."""
        current_size = len(self.memory.chat_memory.messages)
        
        if current_size >= self.memory.max_entries * self.memory.cleanup_threshold:
            # Remove oldest entries to make space
            num_to_remove = int(current_size * 0.2)  # Remove 20% of entries
            self.memory.chat_memory.messages = self.memory.chat_memory.messages[num_to_remove:]
        
        self.memory.save_context(
            {"input": query},
            {"output": answer, "classification": classification}
        )
    
    def _batch_classify_conversations(self, conversations: List[str]) -> List[str]:
        """Batch process multiple conversation classifications efficiently."""
        try:
            # Process all conversations in a single batch using the classifier
            results = []
            for conversation in conversations:
                result = self.classifier.classify_conversation(conversation)
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"Error in batch classification: {str(e)}", exc_info=True)
            return ["Unknown"] * len(conversations)  # Fallback classification
    
    async def load_conversations(self, json_file: str):
        """Load and vectorize conversations with improved batch processing."""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        batch_items = []
        for idx, message in enumerate(data.get("mess", [])):
            if not all(key in message for key in ["time", "user", "ai"]):
                continue
            
            combined_text = f"""
            Time: {message['time']}
            User message: {message['user']}
            AI response: {message['ai']}
            Keywords: {' '.join(set(message['user'].lower().split() + message['ai'].lower().split()))}
            """
            
            batch_items.append({
                "id": str(idx),
                "time": message["time"],
                "user": message["user"],
                "ai": message["ai"],
                "text": combined_text
            })
        
        # Process in optimized batches
        for i in range(0, len(batch_items), ScalabilityConfig.VECTOR_BATCH_SIZE):
            batch = batch_items[i:i + ScalabilityConfig.VECTOR_BATCH_SIZE]
            await self.vector_search.batch_upsert_async(batch)
            
            # Small delay to prevent overwhelming the service
            if i + ScalabilityConfig.VECTOR_BATCH_SIZE < len(batch_items):
                await asyncio.sleep(0.1)
    
    async def cleanup(self):
        """Cleanup resources before shutdown."""
        await self.batch_handler.cleanup()
        self.cache_manager.clear_expired()
    
    def get_conversation_analytics(self) -> Dict[str, Any]:
        """Get analytics about the current conversation."""
        history = self.memory.load_memory_variables({})["chat_history"]
        return {
            "message_count": self.message_count,
            "current_classification": self.current_classification,
            "conversation_length": len(history),
            "classification_changes": self._count_classification_changes(history)
        }
    
    def _count_classification_changes(self, history) -> int:
        """Count classification changes in conversation history."""
        changes = 0
        prev_classification = None
        
        for msg in history:
            if hasattr(msg, "classification"):
                if prev_classification and prev_classification != msg.classification:
                    changes += 1
                prev_classification = msg.classification
        
        return changes
    
    def _format_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results into readable context."""
        context_pieces = []
        for result in search_results:
            metadata = result['metadata']
            time_str = metadata.get('time', 'Unknown')
            user_msg = metadata.get('user', '')
            ai_msg = metadata.get('ai', '')
            
            context_pieces.append(f"[Conversation at {time_str}]")
            context_pieces.append(f"User said: {user_msg}")
            context_pieces.append(f"AI responded: {ai_msg}")
            context_pieces.append("-" * 40)
        
        return "\n".join(context_pieces) if context_pieces else "No relevant context found."

    async def _generate_answer(self, query: str, context: str) -> str:
        """Generate an answer based on the query type and context."""
        try:
            # Check cache first
            cache_key = f"answer:{query}"
            cached_answer = self.cache_manager.get("answers", cache_key)
            if cached_answer:
                return cached_answer

            # Generate answer based on query type
            if "what to eat" in query.lower():
                answer = await self._generate_food_recommendation(context)
            elif "movie" in query.lower() and "recommend" in query.lower():
                answer = await self._generate_movie_recommendation(context)
            elif "how long did it take" in query.lower():
                answer = await self._calculate_time_difference(context)
            elif any(keyword in query.lower() for keyword in ["remember", "what", "when", "who", "where"]):
                answer = await self._generate_factual_answer(query, context)
            else:
                # Run the agent to get the answer
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.agent.invoke({
                        "input": query,
                        "chat_history": self.memory.load_memory_variables({})["chat_history"]
                    })
                )
                answer = result.get("output", "I apologize, I couldn't process that request.")

            # Cache the answer
            self.cache_manager.set("answers", cache_key, answer)
            return answer

        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return "I apologize, but I encountered an error while processing your request."

    async def _generate_food_recommendation(self, context: str) -> str:
        """Generate food recommendations based on past preferences."""
        food_prompt = PromptTemplate(
            input_variables=["context"],
            template="""Based on the conversation history, suggest food options that align with the person's preferences.
            
            Guidelines:
            1. Look for ANY mentions of food in the context
            2. Include both foods they've eaten and foods they've planned to eat
            3. Consider the context of when/where they ate certain foods
            4. If you find ANY food mentions, use them to make suggestions
            5. Only say "I don't have enough information" if there are NO food mentions at all
            
            Examples of useful information:
            - Foods they've eaten recently
            - Foods they've mentioned wanting to eat
            - Types of meals they've had (breakfast, dinner, etc.)
            - Occasions where they've had specific foods
            
            Context: {context}
            
            Response format:
            - If ANY food mentions found: "Based on your history, I see you've enjoyed [foods]. How about trying [suggestion]?"
            - If NO food mentions: "I don't have enough information about your food preferences to make a specific recommendation."
            
            Response:"""
        )
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: (food_prompt | self.llm).invoke({"context": context})
        )
        return result

    async def _generate_movie_recommendation(self, context: str) -> str:
        """Generate movie recommendations based on stated preferences."""
        movie_prompt = PromptTemplate(
            input_variables=["context"],
            template="""You are having a friendly conversation about movies. Use the context to understand their movie preferences and provide personalized recommendations in a casual, friendly tone.

            Context Analysis Steps:
            1. First identify their movie preferences:
               - Explicitly liked genres (e.g., "I prefer romantic movies")
               - Specific movies they liked (e.g., "I also like 'The Pursuit of Happyness'")
               - Genres they dislike (e.g., "I don't like horror movies")
               - Emotional responses to movies (e.g., "made me cry")

            2. Create a preference profile:
               - List all positive preferences
               - List all negative preferences
               - Note any emotional responses
               - Identify pattern in movie choices

            3. Generate recommendations that:
               - Match their stated genre preferences
               - Are similar to movies they've enjoyed
               - Avoid genres they dislike
               - Match the emotional impact they appreciate
            
            Context: {context}

            Response Guidelines:
            - Start with a natural conversation opener
            - Acknowledge their preferences in a casual way
            - Make recommendations that feel personal
            - Add friendly commentary about each suggestion
            - Keep the tone warm and engaging

            Example Response Styles:
            "Oh, since you enjoyed [movie], I think you'd love..."
            "You know what? Given that [preference], I have the perfect movie for you..."
            "Hey, I noticed you're a fan of [genre]. Have you checked out..."

            Remember to be conversational, as if chatting with a friend about movies!
            
            Response:"""
        )
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: (movie_prompt | self.llm).invoke({"context": context})
        )
        return result

    async def _calculate_time_difference(self, context: str) -> str:
        """Calculate time differences between events accurately."""
        time_prompt = PromptTemplate(
            input_variables=["context"],
            template="""Find the exact timestamps for the events in question and calculate the precise time difference.
            Only use explicitly stated timestamps from the context.
            If you find the timestamps, calculate the exact difference.
            Format the response to include the exact times found and the calculated difference.
            
            Context: {context}
            
            Response:"""
        )
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: (time_prompt | self.llm).invoke({"context": context})
        )
        return result

    async def _generate_factual_answer(self, query: str, context: str) -> str:
        """Generate precise answers for factual queries."""
        factual_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""Generate a precise answer based on the conversation history.
            
            Guidelines:
            1. For personal information (names, gender, etc.):
               - State the exact information as found in context
               - Include when the information was last updated if relevant
            
            2. For event timestamps:
               - Include all relevant timestamps
               - For repeated events, mention all occurrences
            
            3. For preferences or choices:
               - State exactly what was mentioned in context
               - Include any relevant context about the choice
            
            Query: {query}
            Context: {context}
            
            Response:"""
        )
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: (factual_prompt | self.llm).invoke({"query": query, "context": context})
        )
        return result

def main():
    # Initialize environment variables
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENV")
    pinecone_index = os.getenv("PINECONE_INDEX")
    together_api_key = os.getenv("TOGETHER_API_KEY")
    
    if not all([pinecone_api_key, pinecone_index, together_api_key]):
        raise ValueError("Missing required environment variables")
    
    # Initialize agent
    agent = ConversationAgent(
        pinecone_api_key=pinecone_api_key,
        pinecone_env=pinecone_env,
        index_name=pinecone_index,
        together_api_key=together_api_key
    )
    
    # Load conversations
    agent.load_conversations('text_1.json')
    
    # Test conversation sequence
    conversation_sequence = [
        "Do you remember my name?",
        "I don't know what to eat",
        "What was my final exam score?",
        "When did I change my name?",
        "What did I eat for Christmas?",
        "Do you remember what color watch I ended up buying?",
        "I forgot the day I said I love you",
        "How long did it take from 'I love you' to breaking up?",
        "Do you remember my gender?",
        "Recommend a movie, preferably one of my favorite genres"
    ]
    
    # Store all Q&A pairs and their analytics
    conversation_history = []
    
    print("\n=== Processing Conversation Sequence ===\n")
    
    for message in conversation_sequence:
        print(f"Processing: {message}")
        
        response = agent.process_query(message)
        analytics = agent.get_conversation_analytics()
        
        # Store the Q&A pair and analytics
        conversation_history.append({
            "question": message,
            "answer": response.answer,
            "classification": response.classification,
            "verification": response.verification_result,
            "context": response.context
        })
        print("✓", end="", flush=True)
    
    print("\n\n=== Complete Conversation Analysis ===\n")
    
    # Print all Q&A pairs with their analysis
    for idx, qa in enumerate(conversation_history, 1):
        print(f"\n[Q{idx}] {qa['question']}")
        print(f"Category: {qa['classification'].strip()}")
        print(f"Answer: {qa['answer'].strip()}")
        print(f"Verified: {'✓' if qa['verification'] else '✗'}")
        
        # If verification failed, show the context that was used
        if not qa['verification']:
            print("\nContext Used:")
            if isinstance(qa['context'], dict) and 'search_results' in qa['context']:
                context_text = qa['context']['search_results']
                if isinstance(context_text, str):
                    print(context_text)
                else:
                    for match in context_text.matches[:3]:
                        print(f"Time: {match.metadata.get('time', 'Unknown')}")
                        print(f"User: {match.metadata.get('user', 'N/A')}")
                        print(f"AI: {match.metadata.get('ai', 'N/A')}")
                        print()
        print("-" * 80)
    
    # Print summary statistics
    print("\n=== Conversation Summary ===")
    
    # Count categories
    categories = {}
    for qa in conversation_history:
        cat = qa['classification'].strip()
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nCategory Distribution:")
    for cat, count in categories.items():
        print(f"- {cat}: {count}")
    
    print(f"\nTotal Questions: {len(conversation_history)}")
    verified_count = sum(1 for qa in conversation_history if qa['verification'])
    print(f"Verified Answers: {verified_count} ({verified_count/len(conversation_history)*100:.1f}%)")
    
    # List questions with potential hallucinations
    unverified = [(qa['question'], qa['answer']) for qa in conversation_history if not qa['verification']]
    if unverified:
        print("\nPotential Hallucinations:")
        for q, a in unverified:
            print(f"Q: {q}")
            print(f"A: {a}")
            print()

if __name__ == "__main__":
    main()
