# DataGem — AI Conversational Data Analyst

> "Talk to your data like never before."

DataGem is an **AI-powered conversational data analyst** that enables users to explore, visualize, and gain insights from datasets using natural language. Built with **FastAPI**, **React (Vite)**, and **Google Gemini 2.0**, DataGem intelligently interprets user queries, generates and executes Python code, and visualizes insights in real time.

---

## Overview

DataGem combines **Natural Language Processing, Data Science, and Full-Stack Engineering** to make data analytics interactive and intuitive. Users can ask questions such as:

- "Show me top 5 products by revenue."
- "Visualize monthly sales trends."
- "Summarize customer churn rate."

The system automatically:
1. Understands the question via **Gemini 2.0 Flash**.
2. Dynamically generates Python code.
3. Executes the code safely in a sandboxed environment.
4. Instantly returns both results and visualizations.

---

## Architecture

This architecture supports:
- Real-time AI streaming
- Modular tool execution
- Automated visualization
- Scalable backend architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Browser                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         React Frontend (Port 5188)                   │   │
│  │  - Chat Interface                                    │   │
│  │  - Dataset Upload & Preview                         │   │
│  │  - Visualizations & Tables                          │   │
│  │  - Real-time Streaming                              │   │
│  └──────────────────┬───────────────────────────────────┘   │
└─────────────────────┼───────────────────────────────────────┘
                      │ HTTP/WebSocket
                      │ (CORS-enabled)
┌─────────────────────┼───────────────────────────────────────┐
│  ┌──────────────────▼───────────────────────────────────┐   │
│  │      FastAPI Backend (Port 8000)                     │   │
│  │  ┌──────────────────────────────────────────────┐    │   │
│  │  │  Chat Router (/chat/)                        │    │   │
│  │  │  - Receives user messages                    │    │   │
│  │  │  - Streams responses                         │    │   │
│  │  └──────────────┬───────────────────────────────┘    │   │
│  │                 │                                     │   │
│  │  ┌──────────────▼───────────────────────────────┐    │   │
│  │  │  DataAnalystAgent                            │    │   │
│  │  │  - Manages Gemini AI interaction             │    │   │
│  │  │  - Handles tool calls                        │    │   │
│  │  │  - Generates summaries                       │    │   │
│  │  └──────────────┬───────────────────────────────┘    │   │
│  │                 │                                     │   │
│  │  ┌──────────────▼───────────────────────────────┐    │   │
│  │  │  Tools Module                                │    │   │
│  │  │  - run_python_code: Executes Python          │    │   │
│  │  │  - google_search: Web search (mock)          │    │   │
│  │  └──────────────────────────────────────────────┘    │   │
│  └──────────────────┬───────────────────────────────────┘   │
└─────────────────────┼───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│  ┌──────────────────▼───────────────────────────────────┐   │
│  │  Google Gemini API                                   │   │
│  │  - Natural language understanding                    │   │
│  │  - Code generation                                   │   │
│  │  - Text summarization                                │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│  ┌──────────────────▼───────────────────────────────────┐   │
│  │  SQLite Database                                     │   │
│  │  - Users table                                       │   │
│  │  - Chat history table                                │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer         | Technology                  |
|---------------|-----------------------------|
| **Frontend**  | React, Vite, TailwindCSS    |
| **Backend**   | FastAPI (Python 3.12)       |
| **AI Engine** | Google Gemini 2.0 Flash     |
| **Database**  | SQLite                      |
| **Deployment**| Docker, Render (Backend), Vercel (Frontend) |

---

