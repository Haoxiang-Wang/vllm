# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Optional, Union

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser, ReasoningParserManager

logger = init_logger(__name__)


@ReasoningParserManager.register_module("robobrain")
class RoboBrainReasoningParser(ReasoningParser):
    """
    Reasoning parser for RoboBrain2.0 models (7B and 32B).

    RoboBrain2.0 uses text-based markers (not special tokens) for reasoning:
    - <think>...</think> for reasoning content
    - <answer>...</answer> for answer content (optional)
    
    Output formats:
    - With thinking: <think>reasoning</think><answer>content</answer>
                 or: <think>reasoning</think>content
    - Without thinking: <answer>content</answer>
                    or: content
    """

    # Text markers (not special tokens)
    think_start: str = "<think>"
    think_end: str = "</think>"
    answer_start: str = "<answer>"
    answer_end: str = "</answer>"

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        
        # State tracking for streaming
        self.current_state = "initial"  # initial, thinking, content
        self.buffer = ""
        self.thinking_complete = False

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        """
        Check if reasoning has ended by looking for </think> in the decoded text.
        """
        text = self.model_tokenizer.decode(input_ids, skip_special_tokens=True)
        return self.think_end in text

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract the content token IDs after </think>.
        """
        text = self.model_tokenizer.decode(input_ids, skip_special_tokens=True)
        
        if self.think_end in text:
            # Find content after </think>
            parts = text.split(self.think_end, 1)
            if len(parts) > 1:
                content_text = parts[1]
                # Remove <answer> tags if present
                content_text = content_text.replace(self.answer_start, "").replace(self.answer_end, "")
                # Re-encode the content
                return self.model_tokenizer.encode(content_text, add_special_tokens=False)
        
        return []

    def extract_reasoning_content(
            self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning content and answer content from the model output.
        
        Handles formats:
        - <think>reasoning</think><answer>content</answer>
        - <think>reasoning</think>content
        - <answer>content</answer>
        - content
        """
        
        reasoning_content = None
        answer_content = None
        
        # Check if there's thinking content
        if self.think_start in model_output and self.think_end in model_output:
            # Extract reasoning between <think> and </think>
            start_idx = model_output.find(self.think_start) + len(self.think_start)
            end_idx = model_output.find(self.think_end)
            reasoning_content = model_output[start_idx:end_idx].strip()
            
            # Get content after </think>
            remaining = model_output[end_idx + len(self.think_end):].strip()
            
            # Check for <answer> tags
            if self.answer_start in remaining and self.answer_end in remaining:
                # Extract content between <answer> tags
                answer_start_idx = remaining.find(self.answer_start) + len(self.answer_start)
                answer_end_idx = remaining.find(self.answer_end)
                answer_content = remaining[answer_start_idx:answer_end_idx].strip()
            elif self.answer_start in remaining:
                # Answer tag started but not closed
                answer_start_idx = remaining.find(self.answer_start) + len(self.answer_start)
                answer_content = remaining[answer_start_idx:].strip()
            else:
                # No answer tags, everything after </think> is content
                answer_content = remaining
                
        elif self.think_start in model_output:
            # Thinking started but not ended
            start_idx = model_output.find(self.think_start) + len(self.think_start)
            reasoning_content = model_output[start_idx:].strip()
            
        elif self.answer_start in model_output and self.answer_end in model_output:
            # No thinking, just answer tags
            answer_start_idx = model_output.find(self.answer_start) + len(self.answer_start)
            answer_end_idx = model_output.find(self.answer_end)
            answer_content = model_output[answer_start_idx:answer_end_idx].strip()
            
        elif self.answer_start in model_output:
            # Answer tag started but not closed
            answer_start_idx = model_output.find(self.answer_start) + len(self.answer_start)
            answer_content = model_output[answer_start_idx:].strip()
            
        else:
            # No special formatting, treat everything as content
            answer_content = model_output.strip()
        
        # Return None for empty strings
        if reasoning_content == "":
            reasoning_content = None
        if answer_content == "":
            answer_content = None
            
        return reasoning_content, answer_content

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        """
        Extract reasoning content from streaming output.
        
        Tracks state transitions:
        - initial -> thinking (when <think> is detected)
        - thinking -> content (when </think> is detected)
        """
        
        # Update buffer with delta text
        self.buffer = current_text
        
        # State machine for streaming
        if self.current_state == "initial":
            if self.think_start in self.buffer:
                # Transition to thinking state
                self.current_state = "thinking"
                
                # Check if we have complete thinking in this delta
                if self.think_end in delta_text:
                    # Complete thinking cycle in one delta
                    think_start_idx = delta_text.find(self.think_start)
                    think_end_idx = delta_text.find(self.think_end)
                    
                    if think_start_idx < think_end_idx:
                        reasoning = delta_text[think_start_idx + len(self.think_start):think_end_idx]
                        remaining = delta_text[think_end_idx + len(self.think_end):]
                        
                        # Clean remaining content
                        remaining = remaining.replace(self.answer_start, "").replace(self.answer_end, "")
                        
                        self.current_state = "content"
                        self.thinking_complete = True
                        
                        return DeltaMessage(
                            reasoning_content=reasoning if reasoning else None,
                            content=remaining if remaining else None
                        )
                else:
                    # Just started thinking
                    if self.think_start in delta_text:
                        # Extract reasoning after <think>
                        idx = delta_text.find(self.think_start) + len(self.think_start)
                        reasoning = delta_text[idx:]
                        return DeltaMessage(reasoning_content=reasoning if reasoning else None)
                    else:
                        # <think> was in previous text
                        return DeltaMessage(reasoning_content=delta_text)
                        
            elif self.answer_start in delta_text:
                # Direct to answer without thinking
                self.current_state = "content"
                # Remove answer tags
                content = delta_text.replace(self.answer_start, "").replace(self.answer_end, "")
                return DeltaMessage(content=content if content else None)
            else:
                # No special markers, treat as content
                if delta_text:
                    return DeltaMessage(content=delta_text)
                    
        elif self.current_state == "thinking":
            if self.think_end in delta_text:
                # End of thinking
                self.current_state = "content"
                self.thinking_complete = True
                
                # Split at </think>
                idx = delta_text.find(self.think_end)
                reasoning = delta_text[:idx]
                remaining = delta_text[idx + len(self.think_end):]
                
                # Clean remaining content
                remaining = remaining.replace(self.answer_start, "").replace(self.answer_end, "")
                
                return DeltaMessage(
                    reasoning_content=reasoning if reasoning else None,
                    content=remaining if remaining else None
                )
            else:
                # Continue thinking
                return DeltaMessage(reasoning_content=delta_text)
                
        elif self.current_state == "content":
            # In content state, output everything as content
            # Remove any answer tags if present
            content = delta_text.replace(self.answer_start, "").replace(self.answer_end, "")
            if content:
                return DeltaMessage(content=content)
        
        return None