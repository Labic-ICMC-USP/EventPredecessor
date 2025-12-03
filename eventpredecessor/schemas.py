
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class LocationSchema(BaseModel):
    """Single concrete location (country, city, lat, long)."""
    country: str = Field(
        description="Country name (never use generic values like 'global', 'world', 'planet', 'earth')."
    )
    city: Optional[str] = Field(
        default=None,
        description="City name or null.",
    )
    lat: Optional[float] = Field(
        default=None,
        description="Latitude as a float, or null if unknown.",
    )
    long: Optional[float] = Field(
        default=None,
        description="Longitude as a float, or null if unknown.",
    )


class WhereSchema(BaseModel):
    """Spatial scope description + list of concrete locations."""
    text: str = Field(
        description=(
            "Textual description of the spatial scope "
            "(e.g., 'Local impact in São Paulo, Brazil' or "
            "'Global impact across several countries')."
        )
    )
    locations: List[LocationSchema] = Field(
        description="List of 1 to N concrete locations related to the event."
    )


class EventSchema(BaseModel):
    """Structured 5W1H event representation.

    The LLM will fill this structure from a news article.
    """

    event_id: str = Field(
        description=(
            "Unique identifier for the event. The LLM can leave it empty; "
            "it will be filled later."
        )
    )
    parent_event_id: Optional[str] = Field(
        default=None,
        description=("Identifier of a precursor/parent event. "
                     "For now, it will always be null."),
    )

    what: str = Field(description="Short description of what happened.")
    when: str = Field(description="Date of the event, in ISO format YYYY-MM-DD when possible.")
    where: WhereSchema = Field(
        description="Location information for the event, including spatial scope and concrete locations."
    )
    who: str = Field(description="Main actors or groups involved.")
    why: str = Field(description="Main reasons or causes.")
    how: str = Field(description="How the event took place (methods, actions, dynamics).")

    category: str = Field(
        description=(
            "Semantic category of the event "
            "(prison, protest, election, corruption, etc.)."
        )
    )
