# Groq Stock Price Agent using LangChain

An intelligent financial agent built with the **LangChain** framework and powered by **Groq Cloud API** (`llama-3.3-70b-versatile`). It retrieves real-time stock prices, company summaries, and key financial statistics using the Yahoo Finance API (`yfinance`).

## Features
- **Real-Time Data**: Fetches latest prices, daily high/low range, market cap, and business summaries.
- **Smart Ticker Matching**: Automatically maps company names to ticker symbols (e.g. "Apple" to `AAPL`, "Microsoft" to `MSFT`).
- **Interactive Mode**: Provides a conversational terminal interface to chat with the agent.
- **Single-Query Mode**: Execute single commands directly from the terminal via command-line arguments.

## Prerequisites
- Python 3.8 or higher.
- A Groq API key from the [Groq Console](https://console.groq.com/).

## Installation

1. Clone or navigate to the project directory:
   ```bash
   cd /Users/tanmaysingh/.gemini/antigravity/scratch/groq_share_price_agent
   ```

2. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and add your Groq API key:
   ```env
   GROQ_API_KEY=your_actual_groq_api_key
   ```

## Usage

You can run the agent in two ways:

### 1. Interactive CLI Mode
Run the script without arguments to enter an interactive conversation:
```bash
python agent.py
```
Example interaction:
```
=== Groq & LangChain Stock Price Agent ===
Type 'quit' or 'exit' to stop.

Ask about a stock (e.g., 'What is Google's share price?'): What is Apple's current share price and what does the company do?
```

### 2. Single-Query Mode
Provide a query directly using the `--query` or `-q` flag:
```bash
python agent.py --query "What is the current share price of MSFT and Nvidia?"
```
