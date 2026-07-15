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
                response = mlx_lm.generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens)
                return response
        except Exception as e:
            logger.error(f"Error in MLX generate: {str(e)}")
            
        return self._generate_mock(prompt)

    async def stream_generate(self, prompt: str, max_tokens: int = 2048) -> AsyncGenerator[str, None]:
        """
        Streaming generator for grounding and synthesis phase.
        """
        if settings.USE_MOCK_LLM or self.model is None:
            async for chunk in self._stream_mock_synthesis(prompt):
                yield chunk
            return

        try:
            import mlx_lm
            self.load_model()
            if self.model and self.tokenizer:
                # stream_generate runs synchronously in generator; yield control periodically
                for response in mlx_lm.stream_generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens):
                    token_text = response if isinstance(response, str) else getattr(response, "text", str(response))
                    yield token_text
                    await asyncio.sleep(0)
                return
        except Exception as e:
            logger.error(f"Error in MLX stream_generate: {str(e)}")

        async for chunk in self._stream_mock_synthesis(prompt):
            yield chunk

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

    async def _stream_mock_synthesis(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Streams a high-quality mock response representing the grounded synthesis phase with citations.
        """
        # Parse what data was retrieved in the prompt to make it realistic
        prompt_lower = prompt.lower()
        
        # Prepare the response text
        think_section = (
            "<think>\n"
            "Analyzing the query and the retrieved database documents.\n"
            "Retrieved database records show active items with status BLOCKED.\n"
            "Fusing task details and comments.\n"
            "Formulating output with Markdown structure and citation references.\n"
            "</think>\n\n"
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
            
        # Stream the thought section
        for i in range(0, len(think_section), 10):
            yield think_section[i:i+10]
            await asyncio.sleep(0.02)
            
        # Stream the response section
        for i in range(0, len(response_text), 15):
            yield response_text[i:i+15]
            await asyncio.sleep(0.01)

llm_manager = LLMManager()
# Proactively load the model if not mocking
llm_manager.load_model()
