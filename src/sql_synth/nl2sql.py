"""Natural language to SQL translator using rule-based patterns."""
import re
import sqlite3
from dataclasses import dataclass
from typing import Optional


@dataclass
class TranslationResult:
    natural_query: str
    sql: str
    confidence: float  # 0-1
    method: str  # "exact", "pattern", "fallback"


# Rule patterns: (regex pattern, SQL template)
# Template uses named groups from the regex
PATTERNS = [
    # COUNT queries
    (
        r"^(how many|count(?: the)?) (\w+)s?(?:\s+(?:are there|exist|do we have|in the (?:database|system)))?$",
        "SELECT COUNT(*) FROM {table}",
        "count_simple",
    ),
    (
        r"^(how many|count) (\w+)s? where (\w+) (?:is |=\s*)(.+)$",
        "SELECT COUNT(*) FROM {table} WHERE {col} = '{val}'",
        "count_where",
    ),
    # SELECT all
    (
        r"^(?:show|list|get|select|find) (?:all )?(?:the )?(\w+)s?$",
        "SELECT * FROM {table}",
        "select_all",
    ),
    # SELECT with WHERE
    (
        r"^(?:show|list|get|find|select) (?:all )?(?:the )?(\w+)s? where (\w+) (?:is |= ?|equals ?)(.+)$",
        "SELECT * FROM {table} WHERE {col} = '{val}'",
        "select_where_eq",
    ),
    (
        r"^(?:show|list|get|find|select) (?:all )?(?:the )?(\w+)s? where (\w+) (?:>|is greater than) ?(\d+)$",
        "SELECT * FROM {table} WHERE {col} > {val}",
        "select_where_gt",
    ),
    (
        r"^(?:show|list|get|find|select) (?:all )?(?:the )?(\w+)s? where (\w+) (?:<|is less than) ?(\d+)$",
        "SELECT * FROM {table} WHERE {col} < {val}",
        "select_where_lt",
    ),
    # ORDER BY
    (
        r"^(?:show|list|get|find) (?:all )?(?:the )?(\w+)s? ordered by (\w+)(?: (asc|desc))?$",
        "SELECT * FROM {table} ORDER BY {col} {dir}",
        "select_ordered",
    ),
    # LIMIT
    (
        r"^(?:show|list|get|find) (?:the )?(?:top |first )?(\d+) (\w+)s?$",
        "SELECT * FROM {table} LIMIT {n}",
        "select_limit",
    ),
    # DISTINCT
    (
        r"^(?:show|list|get) unique (\w+) (\w+)s?$",
        "SELECT DISTINCT {col} FROM {table}",
        "select_distinct",
    ),
    # SUM / AVG / MAX / MIN
    (
        r"^(?:what is the )?(?:total |sum of )?(\w+) of (\w+)s?$",
        "SELECT SUM({col}) FROM {table}",
        "aggregate_sum",
    ),
    (
        r"^(?:what is the )?average (\w+) of (\w+)s?$",
        "SELECT AVG({col}) FROM {table}",
        "aggregate_avg",
    ),
    (
        r"^(?:what is the )?(?:maximum|max) (\w+) (?:of|from) (\w+)s?$",
        "SELECT MAX({col}) FROM {table}",
        "aggregate_max",
    ),
    (
        r"^(?:what is the )?(?:minimum|min) (\w+) (?:of|from) (\w+)s?$",
        "SELECT MIN({col}) FROM {table}",
        "aggregate_min",
    ),
    # GROUP BY
    (
        r"^(?:count|how many) (\w+)s? per (\w+)$",
        "SELECT {col}, COUNT(*) FROM {table} GROUP BY {col}",
        "count_group_by",
    ),
    # DELETE
    (
        r"^delete (?:all )?(\w+)s? where (\w+) (?:is |= ?)(.+)$",
        "DELETE FROM {table} WHERE {col} = '{val}'",
        "delete_where",
    ),
]

_COMPILED = [
    (re.compile(p, re.IGNORECASE), tmpl, name)
    for p, tmpl, name in PATTERNS
]


def translate(query: str, schema: Optional[dict] = None) -> TranslationResult:
    """
    Translate a natural language query to SQL.
    
    Args:
        query: Natural language query string
        schema: Optional dict of {table_name: [col_name, ...]} for validation
    
    Returns:
        TranslationResult with generated SQL
    """
    query = query.strip().rstrip("?").rstrip(".")

    # Try pattern matching
    for pattern, template, method in _COMPILED:
        m = pattern.match(query)
        if m:
            groups = m.groups()
            sql = _apply_template(template, groups, method)
            if sql:
                # Validate table/column against schema if provided
                confidence = 0.85
                if schema:
                    confidence = _validate_against_schema(sql, schema)
                return TranslationResult(
                    natural_query=query,
                    sql=sql,
                    confidence=confidence,
                    method=method,
                )

    # Fallback: generic SELECT
    words = query.lower().split()
    # Try to find a table name from the last noun
    table_candidate = _find_table_name(words)
    if table_candidate:
        return TranslationResult(
            natural_query=query,
            sql=f"SELECT * FROM {table_candidate}",
            confidence=0.3,
            method="fallback",
        )

    return TranslationResult(
        natural_query=query,
        sql="-- Could not translate query",
        confidence=0.0,
        method="failed",
    )


def _apply_template(template: str, groups: tuple, method: str) -> Optional[str]:
    """Fill template with captured groups."""
    try:
        if method == "count_simple":
            # Groups: (how many|count), (table)
            table = groups[-1]
            return template.format(table=table.lower())
        elif method == "count_where":
            _, table, col, val = groups
            return template.format(table=table.lower(), col=col.lower(), val=val.strip("'\""))
        elif method == "select_all":
            table = groups[0]
            return template.format(table=table.lower())
        elif method in ("select_where_eq",):
            table, col, val = groups[0], groups[1], groups[2]
            return template.format(table=table.lower(), col=col.lower(), val=val.strip("'\""))
        elif method in ("select_where_gt", "select_where_lt"):
            table, col, val = groups[0], groups[1], groups[2]
            return template.format(table=table.lower(), col=col.lower(), val=val)
        elif method == "select_ordered":
            table, col, direction = groups[0], groups[1], groups[2] or "ASC"
            return template.format(table=table.lower(), col=col.lower(), dir=direction.upper())
        elif method == "select_limit":
            n, table = groups[0], groups[1]
            return template.format(table=table.lower(), n=n)
        elif method == "select_distinct":
            col, table = groups[0], groups[1]
            return template.format(table=table.lower(), col=col.lower())
        elif method in ("aggregate_sum", "aggregate_avg", "aggregate_max", "aggregate_min"):
            col, table = groups[0], groups[1]
            return template.format(table=table.lower(), col=col.lower())
        elif method == "count_group_by":
            table, col = groups[0], groups[1]
            return template.format(table=table.lower(), col=col.lower())
        elif method == "delete_where":
            table, col, val = groups[0], groups[1], groups[2]
            return template.format(table=table.lower(), col=col.lower(), val=val.strip("'\""))
    except Exception:
        pass
    return None


def _find_table_name(words: list) -> Optional[str]:
    """Try to extract a table name from query words."""
    # Remove common stop words
    stop_words = {"show", "list", "get", "find", "select", "all", "the", "a", "an",
                  "where", "what", "how", "many", "is", "are", "there", "in", "from"}
    candidates = [w for w in words if w not in stop_words and w.isalpha() and len(w) > 2]
    return candidates[-1] if candidates else None


def _validate_against_schema(sql: str, schema: dict) -> float:
    """Check if tables/columns in SQL exist in schema. Returns confidence."""
    confidence = 0.85
    sql_upper = sql.upper()
    for table, cols in schema.items():
        if table.upper() in sql_upper:
            return 0.95  # Table found in schema
    return confidence * 0.5  # Table not in schema, lower confidence
