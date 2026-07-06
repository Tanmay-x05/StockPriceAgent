#!/usr/bin/env python3
import os
import sys
import argparse
import yfinance as yf
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables from .env file
load_dotenv()

@tool
def get_stock_price(ticker: str) -> str:
    """Fetches the latest stock price and basic statistics for a given stock ticker symbol (e.g., AAPL, GOOGL, MSFT)."""
    ticker = ticker.strip().upper()
    try:
        stock = yf.Ticker(ticker)
        # Attempt to get fast info first
        info = stock.info
        
        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("navPrice")
        currency = info.get("currency", "USD")
        company_name = info.get("longName") or info.get("shortName") or ticker
        
        # Fallback to history if currentPrice is not available in info
        if current_price is None:
            hist = stock.history(period="5d")
            if not hist.empty:
                current_price = hist["Close"].iloc[-1]
            else:
                return f"Could not retrieve stock price for ticker '{ticker}'. Please verify if the ticker is correct."
        
        price_str = f"{current_price:.2f} {currency}"
        
        # Add day high, low, and open if available
        day_high = info.get("dayHigh")
        day_low = info.get("dayLow")
        day_range_str = ""
        if day_low and day_high:
            day_range_str = f" | Day Range: {day_low:.2f} - {day_high:.2f} {currency}"
            
        return f"Company: {company_name} ({ticker})\nCurrent Price: {price_str}{day_range_str}"
        
    except Exception as e:
        return f"Error retrieving data for ticker '{ticker}': {str(e)}"

@tool
def get_company_info(ticker: str) -> str:
    """Fetches general company information, business summary, and key financial metrics for a given stock ticker symbol."""
    ticker = ticker.strip().upper()
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        company_name = info.get("longName") or info.get("shortName") or ticker
        summary = info.get("longBusinessSummary", "No summary available.")
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        market_cap = info.get("marketCap")
        pe_ratio = info.get("trailingPE")
        
        market_cap_str = f"{market_cap:,}" if market_cap else "N/A"
        pe_str = f"{pe_ratio:.2f}" if pe_ratio else "N/A"
        
        return (
            f"Company: {company_name} ({ticker})\n"
            f"Sector: {sector} | Industry: {industry}\n"
            f"Market Cap: {market_cap_str} | P/E Ratio: {pe_str}\n\n"
            f"Business Summary:\n{summary}"
        )
    except Exception as e:
        return f"Error retrieving company info for ticker '{ticker}': {str(e)}"

def run_agent(query: str):
    # Verify API key is present
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable is not set.", file=sys.stderr)
        print("Please create a '.env' file in this directory and add your key, e.g.:", file=sys.stderr)
        print("GROQ_API_KEY=gsk_xxxxxxxxxxxx", file=sys.stderr)
        print("\nAlternatively, set it in your terminal environment:", file=sys.stderr)
        print("export GROQ_API_KEY=gsk_xxxxxxxxxxxx", file=sys.stderr)
        sys.exit(1)

    # Initialize the LLM (using Llama 3.3 70B for strong tool calling and logic)
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0
        )
    except Exception as e:
        print(f"Error initializing ChatGroq: {e}", file=sys.stderr)
        sys.exit(1)

    # Define tools
    tools = [get_stock_price, get_company_info]

    # Create the agent prompt
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful and professional financial assistant. "
            "You have access to tools that fetch real-time stock prices and detailed company information. "
            "Always identify the correct stock ticker symbol before invoking tools. If a user asks about "
            "a company without giving the ticker, map the company to the correct ticker symbol "
            "(e.g., Apple -> AAPL, Google -> GOOGL). "
            "Explain your answers clearly and cite the ticker symbol used."
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Construct the agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Run the query
    try:
        response = agent_executor.invoke({"input": query})
        return response.get("output")
    except Exception as e:
        return f"An error occurred while executing the agent: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Groq-powered Stock Price Agent")
    parser.add_argument("--query", "-q", type=str, help="Single query to ask the agent")
    
    # Check if run inside a Jupyter/IPython notebook environment
    # Jupyter runs with arguments like "-f /Users/.../kernel-xxxx.json"
    is_jupyter = any("jupyter" in arg or "ipykernel" in arg or "-f" == arg for arg in sys.argv)
    if is_jupyter:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    if args.query:
        print(f"Query: {args.query}\n")
        result = run_agent(args.query)
        print("\nAgent Response:")
        print(result)
    else:
        print("=== Groq & LangChain Stock Price Agent ===")
        print("Type 'quit' or 'exit' to stop.\n")
        while True:
            try:
                user_input = input("\nAsk about a stock (e.g., 'What is Google's share price?'): ")
                if user_input.strip().lower() in ["quit", "exit"]:
                    print("Goodbye!")
                    break
                if not user_input.strip():
                    continue
                
                result = run_agent(user_input)
                print("\nAgent Response:")
                print(result)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break

if __name__ == "__main__":
    main()
