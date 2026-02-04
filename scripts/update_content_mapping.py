
import json
import os

TRACE_FILE = '/Users/rinit.lulla/Documents/GitHub/agent-evaluation/gcs_data/all_agents_traces.jsonl'
CONTENT_MAPPING_FILE = '/Users/rinit.lulla/Documents/GitHub/agent-evaluation/gcs_data/content_mapping.json'

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {"instructions": {}, "faqs": {}}

def save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def extract_content():
    content_mapping = load_json(CONTENT_MAPPING_FILE)
    if "instructions" not in content_mapping:
        content_mapping["instructions"] = {}
    if "faqs" not in content_mapping:
        content_mapping["faqs"] = {}

    instructions_count = 0
    faqs_count = 0

    if not os.path.exists(TRACE_FILE):
        print(f"Trace file not found: {TRACE_FILE}")
        return

    with open(TRACE_FILE, 'r') as f:
        for line in f:
            try:
                trace = json.loads(line)
            except json.JSONDecodeError:
                continue

            agent_infos = trace.get('agent_info', [])
            for agent in agent_infos:
                tool_infos = agent.get('tool_info', [])
                for tool in tool_infos:
                    tool_name = tool.get('tool_name')
                    tool_args = tool.get('tool_args', {})
                    tool_output = tool.get('tool_output', {})

                    if tool_name == 'get_usecase_instruction':
                        instruction_id = tool_args.get('instruction_id')
                        # Check output for content
                        content_list = tool_output.get('content', [])
                        for item in content_list:
                            if item.get('type') == 'text':
                                text_data = item.get('text', {})
                                if isinstance(text_data, dict):
                                    instruction_text = text_data.get('instruction')
                                    if instruction_id and instruction_text:
                                        if instruction_id not in content_mapping["instructions"]:
                                            content_mapping["instructions"][instruction_id] = instruction_text
                                            instructions_count += 1
                                        # You could also verify if content changed, but we assume ID is unique version

                    if tool_name == 'get_faqs':
                        content_list = tool_output.get('content', [])
                        for item in content_list:
                            if item.get('type') == 'text':
                                text_data = item.get('text', {})
                                if isinstance(text_data, dict):
                                    faqs_list = text_data.get('faqs_list', [])
                                    for faq in faqs_list:
                                        faq_id = faq.get('id')
                                        if faq_id:
                                            # Clean up FAQ object to match content_mapping structure
                                            # Expected: title, body (or faq_body -> body), faq_slug
                                            faq_entry = {
                                                "title": faq.get('title'),
                                                "body": faq.get('body') or faq.get('faq_body'),
                                            }
                                            if faq.get('faq_slug'):
                                                faq_entry["faq_slug"] = faq.get('faq_slug')
                                            
                                            if faq_id not in content_mapping["faqs"]:
                                                content_mapping["faqs"][faq_id] = faq_entry
                                                faqs_count += 1
                                            elif content_mapping["faqs"][faq_id].get('body') != faq_entry['body']:
                                                # Optional: Update if changed
                                                content_mapping["faqs"][faq_id] = faq_entry

    print(f"Added {instructions_count} new instructions.")
    print(f"Added {faqs_count} new FAQs.")
    
    save_json(CONTENT_MAPPING_FILE, content_mapping)
    print(f"Updated content mapping saved to {CONTENT_MAPPING_FILE}")

if __name__ == "__main__":
    extract_content()
