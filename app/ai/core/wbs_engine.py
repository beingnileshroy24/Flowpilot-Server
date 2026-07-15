import io
import json
import logging
import asyncio
from typing import AsyncGenerator
from pypdf import PdfReader
from docx import Document
from app.config import settings

logger = logging.getLogger(__name__)

class WbsEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._is_loading = False

    def extract_text(self, filename: str, content: bytes) -> str:
        text = ""
        ext = filename.split(".")[-1].lower() if "." in filename else ""
        
        try:
            if ext == "pdf":
                reader = PdfReader(io.BytesIO(content))
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            elif ext in ["docx", "doc"]:
                doc = Document(io.BytesIO(content))
                for para in doc.paragraphs:
                    text += para.text + "\n"
            else:
                # Fallback to plain text for .md, .txt, etc.
                text = content.decode("utf-8", errors="ignore")
        except Exception as e:
            logger.error(f"Error parsing document {filename}: {str(e)}")
            text = f"Error extracting text: {str(e)}"
            
        return text

    def sanitize_text(self, text: str, max_chars: int = 24000) -> str:
        """Simple truncation to roughly fit within 6000 tokens (4 chars/token)."""
        if len(text) > max_chars:
            return text[:max_chars] + "... [TRUNCATED]"
        return text

    async def stream_wbs_generation(self, text: str) -> AsyncGenerator[str, None]:
        """
        Generates structured WBS JSON based on document text.
        Streams SSE-compatible string chunks.
        """
        if settings.USE_MOCK_LLM:
            async for chunk in self._stream_mock_generation(text):
                yield chunk
            return

        try:
            import mlx_lm
            
            if self.model is None or self.tokenizer is None:
                yield "data: {\"status\": \"Loading MLX model (this may take a moment)...\"}\n\n"
                # Doing this synchronously blocks the event loop, but for a local desktop app it's acceptable.
                # In production, this would be in a thread pool.
                res = mlx_lm.load(settings.MLX_MODEL)
                self.model, self.tokenizer = res[0], res[1]
                
            prompt = self._build_prompt(text)
            
            # Use mlx_lm stream_generate
            # stream_generate yields text tokens one by one
            yield "data: {\"status\": \"Generating WBS...\"}\n\n"
            
            for response in mlx_lm.stream_generate(self.model, self.tokenizer, prompt, max_tokens=2048):
                token_text = response if isinstance(response, str) else getattr(response, "text", str(response))
                # We yield the raw text piece. We will format it as a JSON payload for SSE.
                payload = json.dumps({"chunk": token_text})
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0) # Yield control to event loop

        except ImportError:
            logger.warning("mlx_lm not installed or failed to import. Falling back to mock generator.")
            async for chunk in self._stream_mock_generation(text):
                yield chunk
        except Exception as e:
            logger.error(f"Error during MLX generation: {str(e)}")
            payload = json.dumps({"error": str(e)})
            yield f"data: {payload}\n\n"

    def _build_prompt(self, text: str) -> str:
        system_instruction = (
            "You are an expert technical project manager and software architect.\n"
            "Analyze the provided product requirements and break them down into a structured Work Breakdown Structure (WBS) containing actionable tasks.\n"
            "For each task, provide the following fields exactly: 'title', 'description', 'type' (EPIC, TASK, BUG, or SUBTASK), 'priority' (LOW, MEDIUM, HIGH, or CRITICAL), 'estimated_hours' (a number), and 'checklist_items' (a list of strings).\n"
            "Your output MUST be a valid JSON array of objects. Do not wrap the JSON in markdown blocks (e.g. no ```json). Do not add any conversational text after the JSON array."
        )
        
        # DeepSeek R1 often benefits from explicit think instructions or chat templates
        prompt = f"<|im_start|>system\n{system_instruction}<|im_end|>\n<|im_start|>user\nRequirements:\n{text}<|im_end|>\n<|im_start|>assistant\n"
        
        # If tokenizer has apply_chat_template, we could use that, but raw string formatting is fine as fallback.
        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Requirements:\n{text}"}
            ]
            try:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
                
        return prompt

    async def _stream_mock_generation(self, text: str) -> AsyncGenerator[str, None]:
        """Mock fallback generator that simulates MLX output."""
        yield "data: {\"status\": \"Using Mock MLX runtime...\"}\n\n"
        
        mock_think = "<think>\nAnalyzing requirements document...\nIdentifying core components.\nFound backend requirements.\nFound frontend requirements.\nConstructing WBS JSON array.\n</think>\n\n"
        for i in range(0, len(mock_think), 5):
            payload = json.dumps({"chunk": mock_think[i:i+5]})
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0.05)
            
        mock_json = [
            {
                "title": "Setup Project Infrastructure",
                "description": "Initialize repositories, configure environments, and setup CI/CD pipelines.",
                "type": "EPIC",
                "priority": "HIGH",
                "estimated_hours": 8.0,
                "checklist_items": ["Initialize backend repo", "Initialize frontend repo", "Setup Docker"]
            },
            {
                "title": "Implement User Authentication",
                "description": "Build login, signup, and JWT middleware.",
                "type": "TASK",
                "priority": "CRITICAL",
                "estimated_hours": 12.0,
                "checklist_items": ["Create User model", "Setup FastAPI auth routes", "Implement React Auth context"]
            }
        ]
        
        json_str = json.dumps(mock_json, indent=2)
        # Yield JSON in chunks
        chunk_size = 15
        for i in range(0, len(json_str), chunk_size):
            payload = json.dumps({"chunk": json_str[i:i+chunk_size]})
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0.02)

wbs_engine = WbsEngine()
