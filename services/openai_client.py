import openai
import logging
from typing import List, Dict, Any, Optional
import json

from config.settings import get_settings

logger = logging.getLogger(__name__)

class OpenAIClient:
    """OpenAI API client for generating responses"""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = openai.OpenAI(api_key=self.settings.OPENAI_API_KEY)
    
    def check_connection(self) -> bool:
        """Check if OpenAI API is accessible"""
        try:
            # Try to list models to check connection
            models = self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"OpenAI connection check failed: {e}")
            return False

    async def generate_response(self, question: str, context_chunks: List[str], 
                              metadata: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Generate response using OpenAI API"""
        try:
            # Prepare context from chunks
            context = "\n\n".join([
                f"Section: {meta.get('heading', 'Unknown') if meta else 'Unknown'}\n{chunk}"
                for chunk, meta in zip(context_chunks, metadata or [{}] * len(context_chunks))
            ])
            
            # Create system message
            system_message = """You are an expert assistant that answers questions based on PDF document content. 
            
Guidelines:
1. Answer questions based ONLY on the provided context from the PDF documents
2. If the information is not in the context, clearly state that you don't have that information
3. Provide specific, detailed answers when possible
4. Reference relevant sections or headings when answering
5. If there are step-by-step instructions, format them clearly
6. Be concise but comprehensive

Context from PDF documents:
""" + context

            # Create user message
            user_message = f"Question: {question}\n\nPlease provide a detailed answer based on the PDF content provided above."

            # Generate response
            response = self.client.chat.completions.create(
                model=self.settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=self.settings.OPENAI_MAX_TOKENS,
                temperature=self.settings.OPENAI_TEMPERATURE
            )
            
            answer = response.choices[0].message.content
            
            # Calculate confidence based on response characteristics
            confidence = self._calculate_confidence(answer, context_chunks)
            
            return {
                "answer": answer,
                "confidence": confidence,
                "model_used": self.settings.OPENAI_MODEL,
                "tokens_used": response.usage.total_tokens if response.usage else 0
            }
            
        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            return {
                "answer": "I'm currently experiencing high demand. Please try again in a moment.",
                "confidence": 0.0,
                "error": "rate_limit"
            }
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI authentication error: {e}")
            return {
                "answer": "There's an issue with the API configuration. Please contact support.",
                "confidence": 0.0,
                "error": "authentication"
            }
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {e}")
            return {
                "answer": "I encountered an error while processing your question. Please try again.",
                "confidence": 0.0,
                "error": str(e)
            }

    def _calculate_confidence(self, answer: str, context_chunks: List[str]) -> float:
        """Calculate confidence score for the generated answer"""
        try:
            # Simple heuristic-based confidence calculation
            confidence = 0.5  # Base confidence
            
            # Check if answer indicates uncertainty
            uncertainty_phrases = [
                "i don't have", "not mentioned", "not provided", 
                "unclear", "cannot determine", "no information"
            ]
            
            answer_lower = answer.lower()
            
            # Reduce confidence if uncertainty is detected
            for phrase in uncertainty_phrases:
                if phrase in answer_lower:
                    confidence = max(0.1, confidence - 0.3)
                    break
            
            # Increase confidence if answer is detailed
            if len(answer) > 200:
                confidence = min(1.0, confidence + 0.2)
            
            # Increase confidence if answer references specific sections
            if any(word in answer_lower for word in ["section", "according to", "as stated"]):
                confidence = min(1.0, confidence + 0.1)
            
            return round(confidence, 2)
            
        except Exception:
            return 0.5

    async def summarize_document(self, chunks: List[str], filename: str) -> str:
        """Generate a summary of the document"""
        try:
            # Combine first few chunks for summary
            content = "\n\n".join(chunks[:5])  # Use first 5 chunks
            
            system_message = f"""You are tasked with creating a concise summary of a PDF document titled "{filename}". 
            
Please provide:
1. A brief overview of what the document is about
2. Main topics or sections covered
3. Key information or instructions provided

Keep the summary to 2-3 paragraphs."""

            response = self.client.chat.completions.create(
                model=self.settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Please summarize this document content:\n\n{content}"}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating document summary: {e}")
            return "Unable to generate summary for this document."

    async def extract_key_topics(self, chunks: List[str]) -> List[str]:
        """Extract key topics from document chunks"""
        try:
            content = "\n\n".join(chunks[:10])  # Use first 10 chunks
            
            system_message = """Extract 5-10 key topics or main subjects from the provided document content. 
            Return them as a simple list, one topic per line, without numbering or bullet points."""

            response = self.client.chat.completions.create(
                model=self.settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Content:\n\n{content}"}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            topics = response.choices[0].message.content.strip().split('\n')
            return [topic.strip() for topic in topics if topic.strip()]
            
        except Exception as e:
            logger.error(f"Error extracting key topics: {e}")
            return []