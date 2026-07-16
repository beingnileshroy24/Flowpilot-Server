import logging
import asyncio
from typing import AsyncGenerator
from app.config import settings

logger = logging.getLogger(__name__)

class LLMManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._is_loading = False

    def load_model(self):
        if settings.USE_MOCK_LLM:
            return
        if self.model is not None and self.tokenizer is not None:
            return
            
        try:
            import mlx_lm
            logger.info(f"Loading MLX model: {settings.MLX_MODEL}...")
            res = mlx_lm.load(settings.MLX_MODEL)
            self.model, self.tokenizer = res[0], res[1]
            logger.info("MLX model loaded successfully.")
        except ImportError:
            logger.warning("mlx_lm not installed. Falling back to Mock LLM.")
        except Exception as e:
            logger.error(f"Failed to load MLX model: {str(e)}. Falling back to Mock LLM.")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Synchronous/blocking generate for planning phase.
        """
        if settings.USE_MOCK_LLM or self.model is None:
            return self._generate_mock(prompt)
            
        try:
            import mlx_lm
            self.load_model()
            if self.model and self.tokenizer:
                try:
                    messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception as e:
                    logger.error(f"Failed to apply chat template: {str(e)}")
                    formatted_prompt = prompt
                response = mlx_lm.generate(self.model, self.tokenizer, formatted_prompt, max_tokens=max_tokens)
                return response
        except Exception as e:
            logger.error(f"Error in MLX generate: {str(e)}")
            
        return self._generate_mock(prompt)

    async def stream_generate(self, prompt: str, max_tokens: int = 2048) -> AsyncGenerator[tuple, None]:
        """
        Streaming generator for grounding and synthesis phase.
        Yields (chunk_type, text) tuples where chunk_type is either:
          - "thought": tokens inside <think>...</think> reasoning block
          - "chunk": tokens that form the final answer
        """
        if settings.USE_MOCK_LLM or self.model is None:
            async for item in self._stream_mock_synthesis(prompt):
                yield item
            return

        try:
            import mlx_lm
            self.load_model()
            if self.model and self.tokenizer:
                try:
                    messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception as e:
                    logger.error(f"Failed to apply chat template: {str(e)}")
                    formatted_prompt = prompt

                # DeepSeek-R1 chat templates append '<think>\n' to the formatted
                # prompt when add_generation_prompt=True. This means the model's
                # very first token is already INSIDE the think block — no <think>
                # tag is ever generated. We must start in_think=True in that case,
                # otherwise all reasoning tokens are wrongly emitted as "chunk".
                in_think = formatted_prompt.rstrip().endswith("<think>")
                think_buffer = ""

                for response in mlx_lm.stream_generate(self.model, self.tokenizer, formatted_prompt, max_tokens=max_tokens):
                    token_text = response if isinstance(response, str) else getattr(response, "text", str(response))

                    # Accumulate a small look-ahead buffer to detect <think>/<think> across token boundaries
                    think_buffer += token_text

                    # Detect opening <think> tag
                    while think_buffer:
                        if not in_think:
                            think_start = think_buffer.find("<think>")
                            if think_start != -1:
                                # Emit any text before <think> as answer chunk
                                before = think_buffer[:think_start]
                                if before:
                                    yield ("chunk", before)
                                in_think = True
                                think_buffer = think_buffer[think_start + len("<think>"):]
                            else:
                                # No <think> found - check if partial match at tail
                                safe_len = max(0, len(think_buffer) - len("<think>"))
                                if safe_len > 0:
                                    emit = think_buffer[:safe_len]
                                    yield ("chunk", emit)
                                    think_buffer = think_buffer[safe_len:]
                                break
                        else:
                            think_end = think_buffer.find("</think>")
                            if think_end != -1:
                                # Emit everything up to </think> as thought
                                thought_text = think_buffer[:think_end]
                                if thought_text:
                                    yield ("thought", thought_text)
                                in_think = False
                                think_buffer = think_buffer[think_end + len("</think>"):]
                            else:
                                # Still inside <think> - safely emit up to end minus partial tag
                                safe_len = max(0, len(think_buffer) - len("</think>"))
                                if safe_len > 0:
                                    emit = think_buffer[:safe_len]
                                    yield ("thought", emit)
                                    think_buffer = think_buffer[safe_len:]
                                break

                    await asyncio.sleep(0)

                # Flush remaining buffer
                if think_buffer:
                    chunk_type = "thought" if in_think else "chunk"
                    yield (chunk_type, think_buffer)

                return
        except Exception as e:
            logger.error(f"Error in MLX stream_generate: {str(e)}")

        async for item in self._stream_mock_synthesis(prompt):
            yield item

    def _generate_mock(self, prompt: str) -> str:
        """
        Generate mock ReAct planning outputs depending on query keywords.
        """
        prompt_lower = prompt.lower()
        
        # Scenario 1: Blockers or Sprint 8 query
        if "sprint 8" in prompt_lower or "blocker" in prompt_lower:
            return (
                "<think>\n"
                "The user is asking about blockers in Sprint 8.\n"
                "I should query MongoDB active collections to find tasks in Sprint 8 with status 'BLOCKED' or with 'blocker' in details.\n"
                "I will use the search_backlog tool.\n"
                "</think>\n"
                "Action: search_backlog(sprint=\"Sprint 8\", status=\"BLOCKED\")"
            )
            
        # Scenario 2: PRD/Requirements query
        if "prd" in prompt_lower or "requirements" in prompt_lower or "document" in prompt_lower:
            return (
                "<think>\n"
                "The user wants to inspect requirements or PRD documents.\n"
                "I will use the read_document_chunk tool to pull from requirements.\n"
                "</think>\n"
                "Action: read_document_chunk(doc_id=\"requirements\", chunk_index=0)"
            )
            
        # Scenario 3: Workload or engineer points query
        if "workload" in prompt_lower or "metrics" in prompt_lower or "engineer" in prompt_lower:
            return (
                "<think>\n"
                "The user wants to inspect team workload metrics.\n"
                "I will use the get_team_workload_metrics tool.\n"
                "</think>\n"
                "Action: get_team_workload_metrics(project_id=\"project_id_placeholder\")"
            )

        # Scenario 4: General query
        return (
            "<think>\n"
            "This is a general query about the workspace knowledge.\n"
            "I should query the database backlog.\n"
            "</think>\n"
            "Action: search_backlog(query=\"description_query\")"
        )

    async def _stream_mock_synthesis(self, prompt: str) -> AsyncGenerator[tuple, None]:
        """
        Streams a high-quality mock response representing the grounded synthesis phase with citations.
        Yields (chunk_type, text) tuples matching the signature of stream_generate().
        """
        # Parse what data was retrieved in the prompt to make it realistic
        prompt_lower = prompt.lower()
        
        # Prepare the response text
        think_text = (
            "Analyzing the query and the retrieved database documents.\n"
            "Retrieved database records show active items with status BLOCKED.\n"
            "Fusing task details and comments.\n"
            "Formulating output with Markdown structure and citation references.\n"
        )
        
        response_text = ""
        if "sprint 8" in prompt_lower:
            response_text = (
                "Based on the retrieved MongoDB collections and active sprint logs, there are **two critical blockers** active in Sprint 8:\n\n"
                "1. **Staging Database Setup** [cit_a7f92]: Blocked by missing credentials in `.env` setup. Assigned to developer John (Developer).\n"
                "2. **OAuth SSO Integration** [cit_b8e21]: Blocked awaiting security design approval from the admin. The associated activity log indicates a status update to `IN_REVIEW` but comments note a blocker on security policy clearance.\n\n"
            )
        elif "prd" in prompt_lower or "requirements" in prompt_lower:
            response_text = (
                "Based on the workspace document chunks retrieved from the PRD requirements [cit_c3d4e]:\n\n"
                "- **Authentication Requirements**: Users must sign in via JWT-based secure session keys [cit_c3d4e].\n"
                "- **Tech Stack**: Backend relies on FastAPI, Beanie ODM, and LanceDB for indexing workspace entities.\n\n"
            )
        else:
            response_text = (
                "Based on the semantic search results from your workspace knowledge base:\n\n"
                "- Several matching workspace components were found relating to your query [cit_d4e5f].\n"
                "- Active milestones and codebases are currently healthy [cit_e5f6a].\n\n"
            )
            
        # Stream the thought section as "thought" typed chunks
        for i in range(0, len(think_text), 10):
            yield ("thought", think_text[i:i+10])
            await asyncio.sleep(0.02)
            
        # Stream the response section as "chunk" typed chunks
        for i in range(0, len(response_text), 15):
            yield ("chunk", response_text[i:i+15])
            await asyncio.sleep(0.01)

llm_manager = LLMManager()
# Proactively load the model if not mocking
llm_manager.load_model()
