
import re
import argparse
import sys

# Schema Mapping
# Maps friendly keywords/regex to the actual log field names
FIELD_MAPPING = {
    # Core IDs
    r"\btrace\s?id\b": "traceId",
    r"\bspan\s?id\b": "spanId",
    r"\bsession\s?id\b": 'jsonPayload.attributes."gcp.vertex.agent.session_id"',
    r"\bproject\s?id\b": "resource.labels.project_id",
    r"\bissue\s?id\b": "jsonPayload.issue_id",
    r"\bmessage\s?id\b": "jsonPayload.message_id",
    r"\bmessages?\b": "jsonPayload.user_message", # Heuristic for "message"

    # Performance / Time
    r"\bduration\b": "duration",  # Often needs special handling like duration > 1s
    r"\blatency\b": "duration",
    r"\bstart\s?time\b": "timestamp",
    
    # Gen AI Attributes
    r"\bmodel\b": 'jsonPayload.attributes."gen_ai.request.model"',
    r"\btemperature\b": 'jsonPayload.attributes."gen_ai.request.temperature"',
    r"\btop\s?p\b": 'jsonPayload.attributes."gen_ai.request.top_p"',
    r"\binput\s?tokens?\b": 'jsonPayload.attributes."gen_ai.usage.input_tokens"',
    r"\boutput\s?tokens?\b": 'jsonPayload.attributes."gen_ai.usage.output_tokens"',
    r"\bfinish\s?reasons?\b": 'jsonPayload.attributes."gen_ai.response.finish_reasons"',
    
    # Agent / Tool Specifics
    r"\bagent\b": 'jsonPayload.attributes."gen_ai.agent.name"',
    r"\btool\b": 'jsonPayload.attributes."gen_ai.tool.name"',
    r"\boperation\b": 'jsonPayload.attributes."gen_ai.operation.name"',
    
    # HTTP / Network
    r"\bstatus\b": 'jsonPayload.attributes."/http/status_code"',
    r"\bmethod\b": 'jsonPayload.attributes."/http/method"',
    r"\burl\b": 'jsonPayload.attributes."/http/url"',
    r"\bhost\b": 'jsonPayload.attributes."/http/host"',
    
    # Errors
    r"\berror\b": "severity=\"ERROR\"", # Heuristic for "show me errors"
}

OPERATORS = {
    "=": "=",
    "is": "=",
    "equals?": "=",
    "contains?": ":",
    "has": ":",
    ">": ">",
    "greater than": ">",
    "<": "<",
    "less than": "<",
    "!=": "!=",
    "not": "!="
}

def translate_nl_to_lql(query: str) -> str:
    """
    Translates a natural language query string into a Google Cloud Logging LQL string.
    """
    query = query.lower()
    
    # 1. Handle Simple Global Filters
    if "error" in query and "show" in query:
        return 'severity="ERROR"'
    
    lql_parts = []
    
    # 2. Extract Comparisons (e.g., "latency > 500ms", "agent is GuardAgent")
    # Regex to capture: <field> <operator> <value>
    # We look for value patterns like quotes, numbers, or simple words
    comparison_regex = r"(.+?)\s+(is|equals?|contains?|has|>|<|!=)\s+(['\"]?[\w\.-]+['\"]?)"
    
    matches = re.finditer(comparison_regex, query)
    found_match = False
    
    for match in matches:
        found_match = True
        raw_field, operator_word, value = match.groups()
        raw_field = raw_field.strip()
        
        # Resolve Field
        lql_field = None
        for pattern, field_name in FIELD_MAPPING.items():
            if re.search(pattern, raw_field):
                lql_field = field_name
                break
        
        # Resolve Operator
        lql_op = OPERATORS.get(operator_word, "=")
        
        if lql_field and lql_field != 'severity="ERROR"':
            # Handle Duration Special Case
            if lql_field == "duration":
                # Ensure value has unit 's' or 'ms' if just number provided, default to ms implies mapping logic, 
                # but LQL requires units.
                if value.replace('"','').isdigit():
                    value = f'{value}ms' 
            
            lql_parts.append(f'{lql_field}{lql_op}{value}')
    
    # 3. Handle standalone keyword searches if no strict comparisons found
    # (e.g., "Find trace_id 12345")
    if not found_match:
        # Check for ID patterns
        id_match = re.search(r"\b([a-f0-9]{32})\b", query) # Trace ID regex
        if id_match:
             lql_parts.append(f'traceId="{id_match.group(1)}"')

    if not lql_parts:
        return f'# Could not confidently translate: "{query}"'
        
    return " AND ".join(lql_parts)

def main():
    parser = argparse.ArgumentParser(description="Translate Natural Language to LQL")
    parser.add_argument("query", nargs="*", help="The natural language query string")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive:
        print("Enter your NL query (or 'exit'):")
        while True:
            try:
                user_input = input("> ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                print(translate_nl_to_lql(user_input))
            except KeyboardInterrupt:
                break
    else:
        full_query = " ".join(args.query)
        if not full_query:
            print("Please provide a query or use --interactive")
            sys.exit(1)
        print(translate_nl_to_lql(full_query))

if __name__ == "__main__":
    main()
