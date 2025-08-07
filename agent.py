"""
Simple examples of wrapping public APIs as tools in LangGraph using @tool decorator
This replaces the older API Chain approach with clean, modern code
"""

import requests
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition


# ============================================
# EXAMPLE 1: SIMPLE GET API - JOKE API
# ============================================

@tool
def get_random_joke() -> str:
    """Get a random joke from the JokeAPI.
    
    Returns a random joke to lighten the mood.
    """
    try:
        response = requests.get(
            "https://official-joke-api.appspot.com/random_joke",
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        setup = data.get("setup", "")
        punchline = data.get("punchline", "")
        
        return f"Setup: {setup}\nPunchline: {punchline}"
        
    except Exception as e:
        return f"Couldn't fetch a joke right now: {str(e)}"


# ============================================
# EXAMPLE 2: API WITH PARAMETERS - COUNTRY INFO
# ============================================

@tool
def get_country_info(country_name: str) -> str:
    """Get information about a country including capital, population, and region.
    
    Args:
        country_name: Name of the country to look up (e.g., 'France', 'Japan')
    """
    try:
        # The REST Countries API is free and doesn't require authentication
        response = requests.get(
            f"https://restcountries.com/v3.1/name/{country_name}",
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        if not data:
            return f"No information found for country: {country_name}"
        
        # Get the first matching country
        country = data[0]
        
        name = country.get("name", {}).get("common", "Unknown")
        capital = country.get("capital", ["Unknown"])[0] if country.get("capital") else "Unknown"
        population = country.get("population", "Unknown")
        region = country.get("region", "Unknown")
        subregion = country.get("subregion", "Unknown")
        
        # Format population with commas
        if isinstance(population, int):
            population = f"{population:,}"
        
        return (
            f"Country: {name}\n"
            f"Capital: {capital}\n"
            f"Population: {population}\n"
            f"Region: {region}\n"
            f"Subregion: {subregion}"
        )
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return f"Country '{country_name}' not found. Try checking the spelling."
        return f"Error fetching country info: {str(e)}"
    except Exception as e:
        return f"Error fetching country info: {str(e)}"


# ============================================
# EXAMPLE 3: API WITH MULTIPLE PARAMS - EXCHANGE RATES
# ============================================

@tool
def get_exchange_rate(from_currency: str, to_currency: str, amount: float = 1.0) -> str:
    """Convert currency using real-time exchange rates.
    
    Args:
        from_currency: Source currency code (e.g., 'USD', 'EUR')
        to_currency: Target currency code (e.g., 'GBP', 'JPY')
        amount: Amount to convert (default is 1.0)
    """
    try:
        # Using the free ExchangeRate-API
        response = requests.get(
            f"https://api.exchangerate-api.com/v4/latest/{from_currency.upper()}",
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        rates = data.get("rates", {})
        
        if to_currency.upper() not in rates:
            return f"Currency '{to_currency}' not found or not supported."
        
        rate = rates[to_currency.upper()]
        converted = amount * rate
        
        return (
            f"{amount} {from_currency.upper()} = {converted:.2f} {to_currency.upper()}\n"
            f"Exchange rate: 1 {from_currency.upper()} = {rate:.4f} {to_currency.upper()}"
        )
        
    except Exception as e:
        return f"Error fetching exchange rate: {str(e)}"


# ============================================
# EXAMPLE 4: API RETURNING STRUCTURED DATA - GITHUB USER
# ============================================

@tool
def get_github_user_info(username: str) -> str:
    """Get public information about a GitHub user.
    
    Args:
        username: GitHub username to look up
    """
    try:
        response = requests.get(
            f"https://api.github.com/users/{username}",
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        
        name = data.get("name", "Not provided")
        company = data.get("company", "Not provided")
        location = data.get("location", "Not provided")
        bio = data.get("bio", "No bio available")
        public_repos = data.get("public_repos", 0)
        followers = data.get("followers", 0)
        following = data.get("following", 0)
        
        return (
            f"GitHub User: {username}\n"
            f"Name: {name}\n"
            f"Company: {company}\n"
            f"Location: {location}\n"
            f"Bio: {bio}\n"
            f"Public Repos: {public_repos}\n"
            f"Followers: {followers}\n"
            f"Following: {following}"
        )
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return f"GitHub user '{username}' not found."
        return f"Error fetching GitHub user info: {str(e)}"
    except Exception as e:
        return f"Error fetching GitHub user info: {str(e)}"


# ============================================
# LANGGRAPH AGENT SETUP
# ============================================

def create_api_agent():
    """Create a LangGraph agent with our API tools"""
    
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # List all our tools
    tools = [
        get_random_joke,
        get_country_info,
        get_exchange_rate,
        get_github_user_info
    ]
    
    # Bind tools to the model
    llm_with_tools = llm.bind_tools(tools)
    
    # Create the tool node using prebuilt ToolNode
    # This handles tool execution, error handling, and parallel execution
    tool_node = ToolNode(tools)
    
    # Define the agent node
    def agent(state: MessagesState):
        """Agent that decides whether to use tools"""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # Build the graph
    workflow = StateGraph(MessagesState)
    
    # Add nodes
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge(START, "agent")
    
    # Use prebuilt tools_condition for routing
    # This automatically routes to "tools" if the agent wants to use a tool
    # or to END if the agent is done
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
    )
    
    # Always return to agent after tool execution
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    return workflow.compile()