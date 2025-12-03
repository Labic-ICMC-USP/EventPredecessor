
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Type, List, Mapping

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel

from .logging_utils import get_logger
from .news import Article
from .schemas import EventSchema

logger = get_logger(__name__)


class LLMExecutor:
    """Thin wrapper around LangChain's ChatOpenAI that enforces a JSON schema
    response using a Pydantic model.

    Typical usage:
        llm = LLMExecutor()
        llm.set_api_key("YOUR_KEY")
        llm.set_model("mistralai/mistral-nemo")  # or another model
        llm.set_schema(MyPydanticSchema)
        llm.set_system_prompt("You are ...")
        llm.set_user_prompt("Here is the task ...")
        obj = llm.run()  # returns an instance of MyPydanticSchema
    """

    def __init__(self) -> None:
        # Default model configuration (you can change via setters)
        self.model_name: str = "mistralai/mistral-nemo"
        self.api_key: Optional[str] = None
        self.base_url: str = "https://openrouter.ai/api/v1"

        # Prompts
        self.system_prompt: Optional[str] = None
        self.user_prompt: Optional[str] = None

        # JSON schema configuration
        self.schema_class: Optional[Type[BaseModel]] = None
        self.schema_name: str = "response_schema"
        self.strict_mode: bool = True

        # Decoding parameters
        self.temperature: float = 0.0
        self.max_tokens: Optional[int] = None

    # ------------------------------------------------------------------
    # Configuration setters
    # ------------------------------------------------------------------

    def set_model(self, model_name: str) -> None:
        """Set the model name to be used by ChatOpenAI."""
        self.model_name = model_name

    def set_api_key(self, api_key: str) -> None:
        """Set the API key for the provider (OpenAI, OpenRouter, etc.)."""
        self.api_key = api_key

    def set_base_url(self, base_url: str) -> None:
        """Set the base URL for the API endpoint (e.g., OpenRouter)."""
        self.base_url = base_url

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt (role/instructions)."""
        self.system_prompt = prompt

    def set_user_prompt(self, prompt: str) -> None:
        """Set the user prompt (task/content)."""
        self.user_prompt = prompt

    def set_schema(
        self,
        schema_class: Type[BaseModel],
        name: str = "response_schema",
        strict: bool = True,
    ) -> None:
        """Configure which Pydantic model describes the JSON output."""
        self.schema_class = schema_class
        self.schema_name = name
        self.strict_mode = strict

    def set_temperature(self, temperature: float) -> None:
        """Set the sampling temperature for the model."""
        self.temperature = temperature

    def set_max_tokens(self, max_tokens: int) -> None:
        """Optional maximum number of tokens for the response."""
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self) -> BaseModel:
        """Execute a single LLM call and parse the output as the configured
        Pydantic schema.

        Returns:
            An instance of `self.schema_class`.

        Raises:
            ValueError if configuration is missing or the response cannot
            be parsed/validated as the schema.
        """
        if not self.api_key:
            raise ValueError("API key is not configured. Call set_api_key(...) first.")
        if not self.schema_class:
            raise ValueError("Pydantic validation class (schema_class) is not set.")
        if not self.user_prompt:
            raise ValueError("User prompt is not set. Call set_user_prompt(...) first.")

        # Build JSON schema for response_format
        json_schema = self.schema_class.model_json_schema()
        # Enforce no extra fields beyond the schema (safety)
        json_schema.setdefault("additionalProperties", False)

        model_kwargs: Dict[str, Any] = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": self.schema_name,
                    "strict": self.strict_mode,
                    "schema": json_schema,
                },
            }
        }

        if self.max_tokens is not None:
            model_kwargs["max_tokens"] = self.max_tokens

        llm = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            model_kwargs=model_kwargs,
        )

        messages = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(HumanMessage(content=self.user_prompt))

        try:
            response = llm.invoke(messages)
            # response.content should be a JSON string
            parsed_json = json.loads(response.content)
            return self.schema_class(**parsed_json)  # type: ignore[arg-type]
        except Exception as e:
            logger.error(
                "Error while processing LLM response.",
                extra={"extra_data": {"error": str(e)}},
                exc_info=True,
            )
            raise ValueError(f"Error while processing LLM response: {e}")


class EventLLMExtractor:
    """Use an LLMExecutor to convert news articles into 5W1H events.

    This version is thread-safe:
    - it clones the base LLM configuration on each call
    - no shared mutable state per request

    Categories configuration:
    - The constructor receives a mapping {label: description}.
    - Descriptions are injected into the system prompt to help the model
      choose the best label.
    - The special label "OTHERS" is automatically added if it is absent.
    """

    def __init__(self, base_llm_executor: LLMExecutor, categories: Mapping[str, str]) -> None:
        # Keep only the configuration of the base LLM
        self.base_llm_executor = base_llm_executor

        # Copy categories (label -> description)
        self.categories: Dict[str, str] = dict(categories)
        if "OTHERS" not in self.categories:
            self.categories["OTHERS"] = (
                "Catch-all category for events that do not fit the other labels."
            )

        self.allowed_categories: List[str] = list(self.categories.keys())

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """System prompt describing exactly what we want from the LLM."""
        cats_str = ", ".join(f"'{c}'" for c in self.allowed_categories)
        cats_with_desc = "\n".join(
            f"- '{label}': {desc}" for label, desc in self.categories.items()
        )

        return (
            "You are an event generator that works with a 5W1H event structure.\n\n"
            "Each event has the following JSON structure:\n"
            "{\n"
            '  "event_id": "unique_string_identifier_based_on_content_or_date",\n'
            '  "parent_event_id": "id_of_parent_event_or_null",\n'
            '  "what": "short description of the event",\n'
            '  "when": "ISO date in the format YYYY-MM-DD",\n'
            '  "where": {\n'
            '      "text": "textual description of the spatial scope (e.g., local, national, global)",\n'
            '      "locations": [\n'
            "          {\n"
            '              "country": "country name (never use generic words like \'global\', \'world\', \'planet\', or \'earth\')",\n'
            '              "city": "city name or null",\n'
            '              "lat": latitude_or_null,\n'
            '              "long": longitude_or_null\n'
            "          },\n"
            '          "..." \n'
            "      ]\n"
            "  },\n"
            '  "who": "main actors involved",\n'
            '  "why": "motivation or cause",\n'
            '  "how": "how the event happens or is executed",\n'
            '  "category": "one of the allowed category labels"\n'
            "}\n\n"
            "Rules for WHERE:\n"
            '- The field "where.text" summarizes the spatial scope '
            '(e.g., "Global impact across several countries" or '
            '"Local impact in Brasília, Brazil").\n'
            '- The list "where.locations" MUST contain at least one concrete location.\n'
            "- For events that are global or affect many regions, you MUST select a SMALL LIST "
            "(2 to 5) of representative countries and/or cities as concrete locations.\n"
            '- Never use generic values such as "global", "world", "planet", or "earth" as the country name.\n'
            '- Each entry in "where.locations" MUST be a real or realistic country/city pair '
            "with lat/long whenever possible.\n\n"
            "Language:\n"
            "- The text fields (what, where.text, who, why, how) MUST be written in the same language "
            "as the news article.\n\n"
            "Category:\n"
            f'- You MUST choose exactly one of the following labels for the field "category": {cats_str}.\n'
            "- The intended meaning of each label is:\n"
            f"{cats_with_desc}\n"
            '- If the article is not clearly about any of these categories, use "OTHERS".\n\n'
            "Output:\n"
            "- You MUST return a single JSON object that strictly follows the JSON schema provided by the system.\n"
            "- Do not include any explanations, comments, or markdown fences in your output.\n"
            "- Return ONLY JSON.\n"
        )


    def _build_user_prompt(self, article: Article) -> str:
        """User prompt containing the raw article content."""
        return (
            "Given the following news article, extract EXACTLY ONE event in the 5W1H format.\n\n"
            "Use the schema described by the system "
            "(event_id, parent_event_id, what, when, where, who, why, how, category)\n"
            "and strictly follow the rules for the \"where\" field (spatial scope and list of locations).\n\n"
            f"Title: {article.title}\n"
            f"Description: {article.description}\n"
            f"URL: {article.url}\n"
            f"Publication date: {article.published.isoformat()}\n\n"
            "All textual fields of the event (what, where.text, who, why, how) MUST be written "
            "in the SAME LANGUAGE as the article above.\n"
            "Return only the JSON object for this single event, with no explanations or additional text."
        )

    def _clone_executor(self) -> LLMExecutor:
        """Create a fresh LLMExecutor instance copying config from base."""
        base = self.base_llm_executor
        exe = LLMExecutor()
        exe.set_api_key(base.api_key)       # type: ignore[arg-type]
        exe.set_model(base.model_name)
        exe.set_base_url(base.base_url)
        exe.set_temperature(base.temperature)
        if base.max_tokens is not None:
            exe.set_max_tokens(base.max_tokens)
        return exe

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_event(self, article: Article) -> Optional[EventSchema]:
        """Call the LLM and parse the response as an EventSchema.

        Returns:
            EventSchema if the extraction succeeds, or None if it fails.
        """
        exe = self._clone_executor()
        exe.set_schema(EventSchema, name="event_schema", strict=True)
        exe.set_system_prompt(self._build_system_prompt())
        exe.set_user_prompt(self._build_user_prompt(article))

        try:
            event_obj = exe.run()
        except Exception as e:
            logger.warning(
                "LLM extraction failed for article.",
                extra={
                    "extra_data": {
                        "title": article.title,
                        "error": str(e),
                    }
                },
            )
            return None

        # Normalize category
        if not event_obj.category:
            event_obj.category = "OTHERS"
        elif event_obj.category not in self.allowed_categories:
            event_obj.category = "OTHERS"

        # Ensure event_id is empty here; we can set it later
        if not event_obj.event_id:
            event_obj.event_id = ""
        # parent_event_id stays None for now (no graph linkage yet)
        event_obj.parent_event_id = None

        return event_obj
