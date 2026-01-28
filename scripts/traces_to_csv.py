import json
import csv
import sys
import os

def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            # For lists, we might want to join them or keep them as stringified lists
            # For spans, we handle them separately in the main loop, but for other lists:
            out[name[:-1]] = json.dumps(x)
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

def main():
    input_file = 'sample_traces.json'
    output_file = 'traces_export.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        sys.exit(1)

    print(f"Reading {input_file}...")
    try:
        with open(input_file, 'r') as f:
            traces = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        sys.exit(1)
        
    print(f"Found {len(traces)} traces.")
    
    rows = []
    all_keys = set()
    
    for trace in traces:
        # Common trace attributes
        trace_common = {k: v for k, v in trace.items() if k != 'spans'}
        
        spans = trace.get('spans', [])
        if not spans:
            # If no spans, just add trace info
            rows.append(trace_common)
            all_keys.update(trace_common.keys())
        else:
            for span in spans:
                row = trace_common.copy()
                # Flatten the span
                flat_span = flatten_json(span)
                
                # Prefix span keys to avoid collision with trace keys if necessary
                # But typically traceId is in trace, spanId in span.
                for k, v in flat_span.items():
                    row[k] = v
                
                rows.append(row)
                all_keys.update(row.keys())
    
    print(f"Extracted {len(rows)} rows (spans).")
    
    # Sort keys for consistent column order
    # Prioritize common ID fields
    priority_keys = ['traceId', 'spanId', 'projectId', 'name', 'startTime', 'endTime']
    sorted_keys = [k for k in priority_keys if k in all_keys] + sorted([k for k in all_keys if k not in priority_keys])
    
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=sorted_keys)
        writer.writeheader()
        writer.writerows(rows)
        
    print("Done.")

if __name__ == "__main__":
    main()
