# SQL Chatbot with Google Gemini

This project implements a **SQL Chatbot** that allows users to interact with a PostgreSQL database using natural language. The system automatically generates SQL queries from user questions, executes them, and provides natural language answers.

The system uses:

- Google Gemini models (`gemini-2.5-flash`) via LangChain  
- Streamlit for web interface  
- SQLAlchemy for database connections  
- Pandas for handling query results  

## Setup

1. Create virtual environment:
   python -m venv venv

2. Activate:
   venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Create .env file:
   GOOGLE_API_KEY=your_api_key_here

5. Run:
   Streamlit run basic.py

