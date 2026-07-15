import re
from typing import List

class RecursiveParagraphSplitter:
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 1 token is roughly 4 characters on average for text. 
        self.char_size = chunk_size * 4
        self.char_overlap = chunk_overlap * 4

    def split_text(self, text: str) -> List[str]:
        """
        Splits text into chunks of maximum length chunk_size (in tokens, approx. 4 chars per token)
        with chunk_overlap overlap.
        """
        if not text:
            return []
            
        # Split recursively by: paragraph (\n\n), line (\n), sentence (. ), space ( )
        separators = ["\n\n", "\n", ". ", " ", ""]
        raw_chunks = self._recursive_split(text, separators)
        
        # Apply overlapping boundaries post-split
        overlapped_chunks = []
        for i, chunk in enumerate(raw_chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
                continue
            
            # Extract overlap from previous chunk
            prev_chunk = raw_chunks[i-1]
            overlap_start = max(0, len(prev_chunk) - self.char_overlap)
            overlap_text = prev_chunk[overlap_start:]
            
            # Combine overlap and current chunk
            overlapped_chunks.append(overlap_text + chunk)
            
        return overlapped_chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.char_size:
            return [text]
            
        if not separators:
            # Fallback: slice by character length
            chunks = []
            start = 0
            while start < len(text):
                end = start + self.char_size
                chunks.append(text[start:end])
                start += self.char_size - self.char_overlap
            return chunks

        separator = separators[0]
        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)
            
        chunks = []
        current_chunk = ""
        
        for part in splits:
            # Append separator if not empty
            candidate = current_chunk + (separator if current_chunk else "") + part
            
            if len(candidate) <= self.char_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Split large individual components recursively
                if len(part) > self.char_size:
                    sub_chunks = self._recursive_split(part, separators[1:])
                    if sub_chunks:
                        chunks.extend(sub_chunks[:-1])
                        current_chunk = sub_chunks[-1]
                    else:
                        current_chunk = ""
                else:
                    current_chunk = part
                    
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
