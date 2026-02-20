import re
import os
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_URL = os.getenv("DATABASE_URL")


st.set_page_config(page_title="SQL Chatbot", layout="wide")
st.title("Chat with DB")

# ----------------------------
# 1) Gemini Models
# ----------------------------
sql_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0,
    api_key=GOOGLE_API_KEY
)

answer_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    api_key=GOOGLE_API_KEY
)

# ----------------------------
# 2) DB Engine
# ----------------------------
@st.cache_resource
def get_engine():
    return create_engine(DB_URL)

# ----------------------------
# 3) Get Schema
# ----------------------------
@st.cache_data(show_spinner=False)
def get_schema():
    engine = get_engine()
    schema_str = ""
    inspector_query = text("""
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(inspector_query)
            current_table = None
            for row in result:
                table_name, column_name = row[0], row[1]
                table_fmt = f'"{table_name}"'
                col_fmt = f'"{column_name}"'

                if table_name != current_table:
                    schema_str += f"\nTable: {table_fmt}\nColumns: "
                    current_table = table_name

                schema_str += f"{col_fmt}, "
    except Exception as e:
        return f"ERROR_READING_SCHEMA: {e}"

    return schema_str.strip()

# ----------------------------
# 4) SQL Prompt & Logic
# ----------------------------
sql_prompt_template = """
You are a PostgreSQL expert. 

RULES:
1) Use double quotes for all identifiers: "Table", "Column".
2) For dates stored as text: Always cast them like "InvoiceDate"::timestamp.
3) For USA: Use "Country" IN ('USA', 'United States', 'US') to avoid matching 'Austria'.

EXAMPLES:
Question: invoices in 2013
SQL: SELECT * FROM "Invoice" WHERE EXTRACT(YEAR FROM "InvoiceDate"::timestamp) = 2013;

Schema:
{schema}

Question: {question}
Return ONLY SQL.
"""

sql_prompt = ChatPromptTemplate.from_template(sql_prompt_template)

def clean_sql(sql: str) -> str:
    sql = sql.strip()
    sql = re.sub(r"```sql?", "", sql, flags=re.IGNORECASE)
    sql = sql.replace("```", "").replace("`", "").strip()
    parts = [p.strip() for p in sql.split(";") if p.strip()]
    return parts[0] if parts else sql

def get_sql_from_llm(question, schema_str):
    chain = sql_prompt | sql_model
    result = chain.invoke({"schema": schema_str, "question": question})
    return clean_sql(result.content)

# ----------------------------
# 5) Run SQL
# ----------------------------
def run_sql(sql_query):
    engine = get_engine()
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql_query), conn)
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)

# ----------------------------
# 6) Natural Response
# ----------------------------
answer_prompt_template = """
User Question: {question}
SQL Result: {data}
Answer the user's question clearly. If no data, say "البيانات لا تعطي إجابة واضحة".
"""
answer_prompt = ChatPromptTemplate.from_template(answer_prompt_template)

def get_natural_answer(question, sql_query, df):
    chain = answer_prompt | answer_model
    data_text = df.to_string(index=False) if not df.empty else "EMPTY"
    result = chain.invoke({"question": question, "data": data_text})
    return result.content.strip()

# ----------------------------
# 7) UI
# ----------------------------
schema_str = get_schema()

question = st.text_input("Ask question for database:")

if st.button("Ask"):
    if question.strip():
        sql_query = get_sql_from_llm(question, schema_str)
        st.subheader("1) Generated SQL")
        st.code(sql_query, language="sql")

        df, err = run_sql(sql_query)
        st.subheader("2) SQL Result")
        if err:
            st.error(f"SQL Error: {err}")
        else:
            st.dataframe(df)
            st.subheader("3) Natural Response")
            st.write(get_natural_answer(question, sql_query, df))