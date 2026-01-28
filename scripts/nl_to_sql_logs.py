import re
import argparse
import sys

# Schema Mapping for Log Analytics SQL
# Maps friendly keywords/regex to SQL expressions
SQL_FIELD_MAPPING = {
    # Core IDs
    r"\btrace\s?id\b": "trace",
    r"\bspan\s?id\b": "span_id",
    r"\bsession\s?id\b": "JSON_VALUE(json_payload.attributes['gcp.vertex.agent.session_id'])",
    r"\bproject\s?id\b": "resource.labels.project_id",
    r"\bissue\s?id\b": "JSON_VALUE(json_payload.issue_id)",
    r"\bmessage\s?id\b": "JSON_VALUE(json_payload.message_id)",
    r"\bmessages?\b": "JSON_VALUE(json_payload.user_message)",
    
    # Gen AI Attributes
    r"\bagent\b": "JSON_VALUE(json_payload.attributes['gen_ai.agent.name'])",
    r"\bmodel\b": "JSON_VALUE(json_payload.attributes['gen_ai.request.model'])",
    r"\binput\s?tokens?\b": "CAST(JSON_VALUE(json_payload.attributes['gen_ai.usage.input_tokens']) AS INT64)",
    r"\boutput\s?tokens?\b": "CAST(JSON_VALUE(json_payload.attributes['gen_ai.usage.output_tokens']) AS INT64)",
    
    # HTTP Attributes
    r"\burl\b": "JSON_VALUE(json_payload.attributes['/http/url'])",
    r"\bmethod\b": "JSON_VALUE(json_payload.attributes['/http/method'])",
    r"\bstatus(\s?code)?\b": "JSON_VALUE(json_payload.attributes['/http/status_code'])",

    # Common Fields
    r"\btimestamp\b": "timestamp",
    r"\bseverity\b": "severity",
}

SQL_OPERATORS = {
    "=": "=",
    "is": "=",
    "equals?": "=",
    "contains?": "LIKE", # Will need % wrapper
    "has": "LIKE",
    ">": ">",
    "greater than": ">",
    "<": "<",
    "less than": "<",
    "!=": "!=",
    "not": "!="
}

def translate_nl_to_sql(query: str, table_name: str = "`my-project.global._Trace._AllSpans`") -> str:
    """
    Translates a natural language query string into a Google Cloud Log Analytics SQL query.
    """
    query = query.lower()
    
    # Default SELECT
    select_clause = "SELECT timestamp, severity, json_payload, resource"
    where_parts = []
    
    # 1. Handle Simple Global Filters
    if "error" in query and "show" in query:
        where_parts.append('severity = "ERROR"')
    
    # 2. Extract Comparisons
    # Capture: <field> <operator> <value>
    # Value can be quoted string OR non-whitespace sequence
    comparison_regex = r"(.+?)\s+(is|equals?|contains?|has|>|<|!=)\s+(['\"]?[\S]+['\"]?)"
    
    matches = re.finditer(comparison_regex, query)
    found_match = False
    
    for match in matches:
        found_match = True
        raw_field, operator_word, value = match.groups()
        raw_field = raw_field.strip()
        
        # Resolve Field
        sql_field = None
        for pattern, field_expr in SQL_FIELD_MAPPING.items():
            if re.search(pattern, raw_field):
                sql_field = field_expr
                break
        
        # Resolve Operator
        sql_op = SQL_OPERATORS.get(operator_word, "=")
        
        # Handle LIKE wildcards
        if sql_op == "LIKE":
            value = f"'%{value.strip('\"').strip('\'')}%'"
        elif sql_op == "=" and not value.replace('.', '').isdigit():
             # Quote string values if not already quoted
             if not (value.startswith("'") or value.startswith('"')):
                 value = f"'{value}'"

        if sql_field:
            where_parts.append(f'{sql_field} {sql_op} {value}')
    
    # 3. Handle standalone ID searches
    if not found_match:
         id_match = re.search(r"\b([a-f0-9]{32})\b", query)
         if id_match:
             where_parts.append(f'trace = "{id_match.group(1)}"')

    where_clause = " WHERE " + " AND ".join(where_parts) if where_parts else ""
    
    # Construct the final query with CTE
    resource_id = 'locations/us-central1/reasoningEngines/3122210601628073984'
    
    cte = f"""WITH TurnsPerSession AS (
  SELECT
    *
  FROM
    {table_name}
  WHERE
    IF(
      '{resource_id}' = '*',
      TRUE,
      ENDS_WITH(JSON_VALUE(resource.attributes, '$."cloud.resource_id"'), '{resource_id}')
    )
)"""

    return f"{cte}\n{select_clause} FROM TurnsPerSession{where_clause} LIMIT 100"

def main():
    parser = argparse.ArgumentParser(description="Translate Natural Language to Log Analytics SQL")
    parser.add_argument("query", nargs="*", help="The natural language query string")
    parser.add_argument("--table", default="`product-research-460317.global._Trace._AllSpans`", help="The BigQuery table name")
    
    args = parser.parse_args()
    
    full_query = " ".join(args.query)
    if not full_query:
        print("Please provide a query")
        sys.exit(1)
        
    print(translate_nl_to_sql(full_query, args.table))

if __name__ == "__main__":
    main()
