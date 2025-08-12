import openai
import logging
from typing import List, Dict, Any, Optional
import logging

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
                              pdf_filename: str) -> Dict[str, Any]:
        """Generate response using OpenAI API"""
        try:
            # Prepare context from chunks
            context = "\n\n".join([
                f"Section {i+1}:\n{chunk}"
                for i, chunk in enumerate(context_chunks)
            ])
            
            # Create system message
            system_message = f"""You are an expert assistant that answers questions based on PDF document content from "{pdf_filename}".

Guidelines:
1. Answer questions based ONLY on the provided context from the PDF document
2. If the information is not in the context, clearly state that you don't have that information
3. Provide specific, detailed answers when possible
4. Reference relevant sections when answering
5. If there are step-by-step instructions, format them clearly with numbers
6. Be concise but comprehensive
7. If images are mentioned in the context, acknowledge them in your response

Context from PDF "{pdf_filename}":
{context}"""

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
            if any(word in answer_lower for word in ["section", "according to", "as stated", "the document"]):
                confidence = min(1.0, confidence + 0.1)
            
            return round(confidence, 2)
            
        except Exception:
            return 0.5

    async def generate_response_with_chunk_identification(self, question: str, context_chunks: List[str], 
                                                        pdf_filename: str) -> Dict[str, Any]:
        """Generate response using OpenAI API and identify which chunks were used"""
        try:
            # Prepare context from chunks
            context = "\n\n".join([
                f"Section {i+1}:\n{chunk}"
                for i, chunk in enumerate(context_chunks)
            ])
            
            # Create system message
            system_message = f"""You are an expert assistant that answers questions based on PDF document content from "{pdf_filename}".

Guidelines:
1. Answer questions based ONLY on the provided context from the PDF document
2. If the information is not in the context, clearly state that you don't have that information
3. Provide specific, detailed answers when possible
4. Reference relevant sections when answering
5. If there are step-by-step instructions, format them clearly with numbers
6. Be concise but comprehensive
7. If images are mentioned in the context, acknowledge them in your response

IMPORTANT: After providing your answer, you MUST include a section titled "USED_CHUNKS:" followed by a comma-separated list of the section numbers (1, 2, 3, etc.) that you actually used to generate your response. Only include sections that were directly relevant to answering the question.

Context from PDF "{pdf_filename}":
{context}"""

            # Create user message
            user_message = f"Question: {question}\n\nPlease provide a detailed answer based on the PDF content provided above. Remember to include the USED_CHUNKS section at the end."

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
            
            full_response = response.choices[0].message.content
            
            # Extract answer and used chunk indices
            answer, used_chunk_indices = self._parse_response_with_chunks(full_response, len(context_chunks))
            
            # Calculate confidence based on response characteristics
            confidence = self._calculate_confidence(answer, context_chunks)
            
            return {
                "answer": answer,
                "confidence": confidence,
                "used_chunk_indices": used_chunk_indices,
                "model_used": self.settings.OPENAI_MODEL,
                "tokens_used": response.usage.total_tokens if response.usage else 0
            }
            
        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            return {
                "answer": "I'm currently experiencing high demand. Please try again in a moment.",
                "confidence": 0.0,
                "used_chunk_indices": [],
                "error": "rate_limit"
            }
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI authentication error: {e}")
            return {
                "answer": "There's an issue with the API configuration. Please contact support.",
                "confidence": 0.0,
                "used_chunk_indices": [],
                "error": "authentication"
            }
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {e}")
            return {
                "answer": "I encountered an error while processing your question. Please try again.",
                "confidence": 0.0,
                "used_chunk_indices": [],
                "error": str(e)
            }

    def _parse_response_with_chunks(self, full_response: str, total_chunks: int) -> tuple[str, List[int]]:
        """Parse the response to extract answer and used chunk indices"""
        try:
            # Split response by USED_CHUNKS marker
            parts = full_response.split("USED_CHUNKS:")
            
            if len(parts) != 2:
                # If no USED_CHUNKS section found, return full response and empty list
                return full_response.strip(), []
            
            answer = parts[0].strip()
            chunks_section = parts[1].strip()
            
            # Parse chunk indices
            used_chunk_indices = []
            try:
                # Extract numbers from the chunks section
                import re
                numbers = re.findall(r'\d+', chunks_section)
                for num_str in numbers:
                    chunk_index = int(num_str) - 1  # Convert to 0-based index
                    if 0 <= chunk_index < total_chunks:
                        used_chunk_indices.append(chunk_index)
                
                # Remove duplicates and sort
                used_chunk_indices = sorted(list(set(used_chunk_indices)))
                
            except Exception as e:
                logger.warning(f"Error parsing chunk indices: {e}")
                used_chunk_indices = []
            
            return answer, used_chunk_indices
            
        except Exception as e:
            logger.error(f"Error parsing response with chunks: {e}")
            return full_response.strip(), []