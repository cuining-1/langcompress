"""Summary prompt templates: the eight-segment structured summary (design §7).

``EIGHT_SEGMENT_TEMPLATE`` is this package's default (the design's enhancement
over Claude Code's eight-segment and Gemini CLI's five-segment formats). It
contains a ``{messages}`` placeholder consumed by the parent
``SummarizationMiddleware._create_summary`` via ``.format(messages=...)``.

``DEFAULT_SUMMARY_PROMPT`` re-exports the parent ``SummarizationMiddleware``'s
default prompt when ``langchain`` is importable (so hosts wanting plain parent
behaviour can pass ``summary_template=DEFAULT_SUMMARY_PROMPT``); otherwise a
fallback copy is used so the module stays importable without ``langchain``.
"""
from __future__ import annotations

EIGHT_SEGMENT_TEMPLATE = """<role>
Context Extraction Assistant
</role>

<primary_objective>
Your sole objective is to extract the highest-quality, most relevant context from the conversation history below. This context will replace the conversation history, so include only the information necessary to continue toward the overall goal.
</primary_objective>

<objective_information>
You are nearing the input token limit. Extract the most important context from the conversation history to replace it. Ensure you do not repeat completed actions.
</objective_information>

<instructions>
Structure your summary using the following eight sections. Each section is a checklist — populate it with relevant information or explicitly state "None" if there is nothing to report.

## 1. Primary Request and Intent
What is the user's primary goal or request? What overall task is being accomplished? Concise but complete enough to understand the purpose of the entire session. Never lose this.

## 2. Key Technical Concepts
Key technical decisions, constraints, and assumptions. Include the reasoning behind key decisions. Document rejected options and why they were not pursued.

## 3. Files and Code Sections
What artifacts, files, or resources were created, modified, or accessed? List specific file paths and briefly describe the changes. Prevents silent loss of artifact information.

## 4. Errors and Fixes
What errors were encountered and how were they fixed? Preserve error history to avoid repeating mistakes. Keep error codes/root causes; compress full stack frames.

## 5. Problem Solving
The key path of problem solving: what was tried, what worked, what was learned. Keep conclusions; compress reasoning chains.

## 6. All User Messages
All user messages from the session, preserved in compressed (paraphrased) form. User input is sacred — never drop a user message entirely.

## 7. Pending Tasks
What specific tasks remain to achieve the session intent? What should be done next?

## 8. Entity State
Key entities and their current state (e.g. user name, project name, selected options, environment). Preserves entity tracking across compaction to avoid re-asking.
</instructions>

Carefully read the entire conversation history and extract the most important, relevant context to replace it so you can free up space. Respond ONLY with the extracted context. Do not include any text before or after the extracted context.

<messages>
Messages to summarize:
{messages}
</messages>"""


# Plan-A fallback prompt (design §8.2): a simpler, lower-output prompt used to
# retry summarization when the primary (eight-segment) summary fails quality
# validation. Removes the strict eight-segment constraint to reduce LLM failure
# probability and output length. Consumed by CompressionMiddleware._create_summary
# via .format(messages=...) — same {messages} placeholder as EIGHT_SEGMENT_TEMPLATE.
FALLBACK_SUMMARY_PROMPT = """<role>
Context Extraction Assistant
</role>

<primary_objective>
Produce a concise, faithful summary of the conversation history below. The summary will replace the history, so preserve all information needed to continue the task.
</primary_objective>

<instructions>
Focus on, in this order:
1. The user's primary goal and intent.
2. Key decisions, constraints, and reasoning (including rejected options).
3. Files/artifacts created, modified, or accessed (with paths).
4. Errors encountered and how they were fixed.
5. Pending tasks and next steps.
6. Key entity state (names, selections, environment).
Keep it concise but complete. Do not include any text before or after the summary.
</instructions>

<messages>
Messages to summarize:
{messages}
</messages>"""


# Fallback copy of the parent SummarizationMiddleware default prompt, used when
# langchain is unavailable.
_FALLBACK_DEFAULT_SUMMARY_PROMPT = """<role>
Context Extraction Assistant
</role>

<primary_objective>
Extract the highest quality/most relevant context from the conversation history below.
</primary_objective>

<messages>
Messages to summarize:
{messages}
</messages>"""


def _load_default_summary_prompt() -> str:
    try:
        from langchain.agents.middleware.summarization import (
            DEFAULT_SUMMARY_PROMPT as _parent,
        )
    except ImportError:
        return _FALLBACK_DEFAULT_SUMMARY_PROMPT
    return _parent


DEFAULT_SUMMARY_PROMPT = _load_default_summary_prompt()

__all__ = ["DEFAULT_SUMMARY_PROMPT", "EIGHT_SEGMENT_TEMPLATE", "FALLBACK_SUMMARY_PROMPT"]
