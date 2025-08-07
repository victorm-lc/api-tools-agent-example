# LangGraph API Tools Example

A simple example showing how to wrap REST APIs as tools in LangGraph using the modern `@tool` decorator approach.

## What This Does

Creates an AI agent that can call multiple public APIs to answer questions about:
- Random jokes
- Country information
- Currency exchange rates
- GitHub user profiles

## Quick Start

### 1. Install Dependencies

```bash
# using UV
uv sync langgraph langchain-openai langchain-core

# Using pip
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Duplicate .env.example and create .env file from it. Set OpenAI API Key (LangSmith optional)
```bash
OPENAI_API_KEY="<your_api_key>"
```

### 3. Run the Example with LangGraph Studio

```bash
source .venv/bin/activate
langgraph dev
```

## How It Works

Instead of complex API chains, just write simple Python functions:

```python
@tool
def get_country_info(country_name: str) -> str:
    """Get information about a country."""
    response = requests.get(f"https://restcountries.com/v3.1/name/{country_name}")
    # ... handle response
    return formatted_result
```

That's it! The `@tool` decorator turns any function into a tool the AI can use.

## Example Queries

Ask the agent questions like:
- "Tell me a joke"
- "What's the population of Japan?"
- "Convert 100 USD to EUR"
- "Get info about the GitHub user hwchase17"
- "Tell me about France and convert 50 EUR to USD" (uses multiple tools!)

## Key Benefits vs Old API Chain

- ✅ **Simple** - Just regular Python functions
- ✅ **Type-safe** - Automatic parameter validation
- ✅ **Parallel** - Multiple API calls run concurrently
- ✅ **Testable** - Tools are plain functions
- ✅ **Production-ready** - Built-in error handling and retries

## APIs Used

All APIs are free and require no authentication:
- [JokeAPI](https://official-joke-api.appspot.com/) - Random jokes
- [REST Countries](https://restcountries.com/) - Country data
- [ExchangeRate-API](https://www.exchangerate-api.com/) - Currency rates
- [GitHub API](https://docs.github.com/en/rest) - Public user info

## License

MIT