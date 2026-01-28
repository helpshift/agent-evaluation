
from google.cloud import trace_v1
from google.protobuf.json_format import MessageToDict
import json
import datetime
import os

def list_traces(project_id):
    client = trace_v1.TraceServiceClient()
    
    # Calculate time range: last 12 hours (user mentioned P14D but let's look at recent first to see schema)
    # Actually, user linked to a view with duration P14D.
    # We'll just fetch the latest 5 traces.
    
    # view=2 corresponds to COMPLETE (0=VIEW_TYPE_UNSPECIFIED, 1=MINIMAL, 2=ROOTSPAN, 3=COMPLETE usually, but let's check or just default)
    # The API definition usually has ViewType. 
    # Let's try to access types directly if possible, or just not specify view (defaults to something). 
    # Actually, default is MINIMAL. We generally want COMPLETE.
    # In v1, view is an Enum.
    
    # Try using trace_v1.ListTracesRequest.ViewType if it exists, else just 2.
    # But wait, google-cloud-trace v1 is for the Stackdriver Trace API v1.
    # The ListTracesRequest message has a field 'view'.
    
    try:
         view_option = trace_v1.ListTracesRequest.ViewType.COMPLETE
    except AttributeError:
         view_option = 2 # Fallback
         
    request = trace_v1.ListTracesRequest(
        project_id=project_id,
        view=view_option,
        page_size=5
    )
    
    print(f"Fetching traces for project: {project_id}...")
    # Iterating the pager
    page_result = client.list_traces(request=request)
    
    traces_found = []
    for trace in page_result:
        # trace is a Trace object
        # We can convert to dict for easier inspection
        trace_dict = MessageToDict(trace._pb)
        traces_found.append(trace_dict)
        if len(traces_found) >= 5:
            break
            
    print(f"Found {len(traces_found)} traces.")
    
    if traces_found:
        # Save to file for inspection
        with open('sample_traces.json', 'w') as f:
            json.dump(traces_found, f, indent=2)
        print("Saved sample_traces.json")
        
        # Print a summary of the first trace's spans and attributes keys to console
        first_trace = traces_found[0]
        print(f"Trace ID: {first_trace.get('traceId')}")
        spans = first_trace.get('spans', [])
        print(f"Number of spans: {len(spans)}")
        for i, span in enumerate(spans[:3]): # Show first 3 spans
            print(f"Span {i}: {span.get('name')}")
            attrs = span.get('attributes', {}).get('attributeMap', {})
            print(f"  Attributes: {list(attrs.keys())}")

if __name__ == "__main__":
    PROJECT_ID = "product-research-460317"
    try:
        list_traces(PROJECT_ID)
    except Exception as e:
        print(f"Error: {e}")
