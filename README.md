
# EventPredecessor

`EventPredecessor` is a small but ambitious pipeline to **discover precursor
events from news**. It combines:

1. **News collection** in temporal windows with `GNews`;
2. **5W1H event extraction** powered by an LLM (JSON-constrained with Pydantic);
3. **Semantic graph construction**, where edges represent likely *precursor*
   relations between events, based on sentence embeddings and temporal order;
4. Optional **visualization on a world map** using `yfiles_jupyter_graphs`.

The goal is to explore chains of events like *"protests → repression → arrests →
political crisis"*, starting from a user-defined reference window and walking
backwards in time to search for possible precursors.

---

## 1. Installation

From a local clone of this project:

```bash
pip install .
```

This will install the package `eventpredecessor` and the console script
`eventpredecessor`.

> **Important:** the package expects you to have an API key for an LLM provider.
> By default, it is configured for OpenRouter (`ChatOpenAI` via
> `langchain-openai`), but you can adapt it to any compatible endpoint.

---

## 2. High-level architecture

The pipeline is composed of the following main components:

- `PrecursorNewsCollector` (`news.py`)
  - Receives a reference time window and a step (`d`, `w`, `m`, `y`).
  - Iteratively builds earlier windows (precursors) and queries `GNews`.
  - Normalizes results into `Article` objects.

- `EventLLMExtractor` (`llm.py`)
  - Wraps `LLMExecutor` and converts each news article into a 5W1H event
    (`EventSchema`).
  - Uses a **category dictionary** `{label: description}` provided in the
    YAML config to guide the LLM when choosing the event category.

- `PrecursorEventGraphBuilder` (`graph_builder.py`)
  - Receives `(article, event)` pairs and builds a directed graph.
  - First phase: for each event, chooses at most **one best parent** whose
    similarity is above `mean + std` of all candidate pairs.
  - Second phase: connects disconnected components while keeping at most
    one parent per node.

- `graph_io.py` and `plot.py`
  - Utilities to export/import the full graph as JSON.
  - `EventGraphPlotter` allows you to inspect the graph in Jupyter, either
    with a data sidebar or on top of a world map (using lat/long).

- `EventPredecessorPipeline` (`runner.py`)
  - Reads a YAML configuration file.
  - Orchestrates news collection, LLM extraction, and graph construction.
  - Saves the final graph to a JSON file.

All **parameters** (keywords, time windows, LLM model, categories, thresholds,
etc.) are stored in a single YAML file to make experiments fully reproducible.

---

## 3. Configuration via YAML

Below is an example `config.yaml`:

```yaml
news:
  keywords:
    - "prisão de bolsonaro"
  reference_start: "2025-11-20"
  reference_end: "2025-11-23"
  window: "m"              # d (day), w (week), m (month), y (year)
  max_iterations: 3
  stop_when_no_articles: false
  searcher:
    language: "pt"
    country: "BR"
    max_results: 100

categories:
  "prisão": "Arrests, detentions, warrants and similar events."
  "protestos": "Social demonstrations, marches, protests, strikes."
  "eleições": "Elections, campaigning, official voting processes."
  "corrupção": "Corruption scandals, misuse of public funds, bribery."
  "investigação": "Formal investigations, inquiries, commissions."
  "decisão judicial": "Court decisions, sentences, judicial rulings."

llm:
  # Prefer using an environment variable for the key:
  api_key_env_var: "OPENROUTER_API_KEY"
  # Or set it directly here (less secure):
  # api_key: "sk-..."
  model_name: "mistralai/mistral-nemo"
  base_url: "https://openrouter.ai/api/v1"
  temperature: 0.0
  max_tokens: 2048
  max_workers: 20

builder:
  embedding_model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  same_category_only: true
  connect_components: true

output:
  full_graph_json_path: "events_full_graph.json"
```

> Note: The `categories` section is a **dictionary**. The keys are labels that
> will appear in the `EventSchema.category` field, and the values are short
> descriptions injected in the system prompt. This helps the LLM to understand
> the intended meaning of each label.

---

## 4. Running the pipeline (CLI)

Once installed, create your `config.yaml` and simply run:

```bash
eventpredecessor -c config.yaml
```

This will:

1. Collect news for the reference window and precursor windows;
2. Extract 5W1H events for each article using the configured LLM;
3. Build a precursor graph with semantic and temporal constraints;
4. Save the final graph to `output.full_graph_json_path` (in the example:
   `events_full_graph.json`).

All steps log structured JSON to stdout using Python's `logging` with a
custom JSON formatter.

---

## 5. Using the pipeline from Python

```python
from eventpredecessor import EventPredecessorPipeline

pipeline = EventPredecessorPipeline.from_yaml("config.yaml")
result = pipeline.run()

G = result["graph"]
stats = result["stats"]
print(stats)
```

You can then export or visualize the graph as you prefer. The package already
offers JSON helpers and a yFiles-based plotter:

```python
from eventpredecessor.graph_io import save_full_graph_json
from eventpredecessor.plot import EventGraphPlotter

save_full_graph_json(G, "events_full_graph.json")
plotter = EventGraphPlotter("events_full_graph.json")
w_sidebar = plotter.plot_with_sidebar()
w_map = plotter.plot_on_map(use_heat_mapping=True)
```

The `tutorial.ipynb` included in this repository shows a small end-to-end
example.

---

## 6. Why this is interesting?

- It transforms *raw news streams* into **event graphs** that can be explored
  as chains of causality, escalation or diffusion.
- It is **LLM-agnostic**: any provider compatible with `ChatOpenAI` can be
  used (OpenAI, OpenRouter, local gateways, etc.).
- All hyperparameters and semantic choices (like event categories) live in a
  YAML file, which makes it a nice playground for students and researchers
  interested in event mining, graph analysis, and precursor detection.
