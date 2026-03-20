"""SQL Synth Agentic Playground — Streamlit UI."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

from sql_synth.nl2sql import translate
from sql_synth.benchmark import run_benchmark, SPIDER_SUBSET
from sql_synth.db import create_demo_db, execute_sql

# Page config
st.set_page_config(
    page_title="SQL Synth Agentic Playground",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 SQL Synth Agentic Playground")
st.markdown("*Natural language to SQL translation with benchmark evaluation*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    show_confidence = st.checkbox("Show confidence scores", value=True)
    st.markdown("---")
    st.markdown("**Demo Tables:**")
    st.code("employee, product, customer")

# Main tabs
tab1, tab2, tab3 = st.tabs(["🔤 Translate", "📊 Benchmark", "🗄️ Execute"])

# ─── Tab 1: Translate ───────────────────────────────────────────────────────
with tab1:
    st.header("Natural Language → SQL")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Enter your query in plain English:",
            placeholder="e.g. Show all employees where salary > 80000",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        translate_btn = st.button("Translate", type="primary")

    # Quick examples
    st.markdown("**Quick examples:**")
    examples = [
        "List all employees",
        "Show employees where salary > 80000",
        "How many customers are there",
        "Get the top 5 products",
        "Show employees ordered by salary desc",
        "What is the average salary of employees",
    ]
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        if cols[i % 3].button(ex, key=f"ex_{i}"):
            query = ex
            translate_btn = True

    if query and translate_btn:
        result = translate(query)
        
        st.markdown("### Result")
        st.code(result.sql, language="sql")
        
        if show_confidence:
            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence", f"{result.confidence:.0%}")
            col2.metric("Method", result.method)
            col3.metric("Query Length", f"{len(query)} chars")

# ─── Tab 2: Benchmark ────────────────────────────────────────────────────────
with tab2:
    st.header("Benchmark Evaluation")
    st.markdown(f"Testing against {len(SPIDER_SUBSET)} Spider-style examples")
    
    if st.button("Run Benchmark", type="primary"):
        with st.spinner("Running benchmark..."):
            report = run_benchmark()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Examples", report.total)
        col2.metric("Exact Match", f"{report.exact_match_rate:.0%}")
        col3.metric("Structural Match", f"{report.structural_match_rate:.0%}")
        col4.metric("Avg Confidence", f"{report.avg_confidence:.0%}")
        
        st.markdown("### Detailed Results")
        for r in report.results:
            status = "✅" if r.structural_match else "❌"
            with st.expander(f"{status} {r.example.question}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Gold SQL:**")
                    st.code(r.example.gold_sql, language="sql")
                with col2:
                    st.markdown("**Predicted SQL:**")
                    st.code(r.predicted_sql, language="sql")
                st.markdown(
                    f"Confidence: `{r.confidence:.0%}` | Method: `{r.method}` | "
                    f"Exact: `{r.exact_match}` | Structural: `{r.structural_match}`"
                )

# ─── Tab 3: Execute ──────────────────────────────────────────────────────────
with tab3:
    st.header("Execute SQL on Demo Database")
    st.info("A demo SQLite database with employee, product, and customer tables is pre-loaded.")
    
    demo_conn = create_demo_db()
    
    # NL to SQL + execute
    nl_query = st.text_input(
        "Natural language query:",
        placeholder="e.g. List all employees",
        key="exec_nl"
    )
    
    if nl_query:
        result = translate(nl_query)
        sql_input = st.text_area("SQL (auto-generated, edit if needed):", value=result.sql, height=80)
    else:
        sql_input = st.text_area(
            "Or enter SQL directly:",
            value="SELECT * FROM employee",
            height=80,
        )
    
    if st.button("Execute", type="primary"):
        if sql_input and not sql_input.startswith("--"):
            try:
                cursor = demo_conn.execute(sql_input)
                rows = cursor.fetchall()
                if rows:
                    import pandas as pd
                    cols = [d[0] for d in cursor.description]
                    df = pd.DataFrame([dict(zip(cols, r)) for r in rows])
                    st.success(f"✅ {len(rows)} row(s) returned")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.success("✅ Query executed successfully (no rows returned)")
            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.warning("Please enter a valid SQL query")


if __name__ == "__main__":
    pass  # Run with: streamlit run app.py
