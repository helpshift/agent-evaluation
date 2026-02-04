
import json
import os

INPUT_FILE = '/Users/rinit.lulla/Documents/GitHub/agent-evaluation/gcs_data/all_agents_traces.jsonl'
OUTPUT_FILE_1 = 'test1.jsonl'
OUTPUT_FILE_2 = 'test2.jsonl'
MAPPING_FILE = 'test_content_mapping.json'

def main():
    print(f"Reading from {INPUT_FILE}...")
    traces = []
    if os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, 'r') as f:
            for line in f:
                try:
                    traces.append(json.loads(line))
                except:
                    pass
    else:
        print("Input file not found!")
        return

    # 1. Generate test1.jsonl (Copy)
    print(f"Generating {OUTPUT_FILE_1}...")
    with open(OUTPUT_FILE_1, 'w') as f:
        for t in traces:
            f.write(json.dumps(t) + '\n')

    # 2. Generate test2.jsonl (Optimized) & Mapping
    print(f"Generating {OUTPUT_FILE_2} and {MAPPING_FILE}...")
    content_mapping = {"instructions": {}, "faqs": {}}
    optimized_traces = []

    for trace in traces:
        # Deep copy to ensure we don't modify the objects for test1 if we were doing this in memory differently
        # (Though we already wrote test1, so modification here is fine for test2)
        new_trace = json.loads(json.dumps(trace))
        
        agent_infos = new_trace.get('agent_info', [])
        for agent in agent_infos:
            tool_infos = agent.get('tool_info', [])
            for tool in tool_infos:
                tool_name = tool.get('tool_name')
                tool_args = tool.get('tool_args', {})
                tool_output = tool.get('tool_output', {})
                
                # Instruction Logic
                if tool_name == "get_usecase_instruction" and isinstance(tool_args, dict):
                    instruction_id = tool_args.get("instruction_id")
                    if instruction_id and isinstance(tool_output, dict):
                        content_list = tool_output.get("content", [])
                        structured = tool_output.get("structuredContent", {})
                        
                        instruction_text = None
                        
                        # Check structuredContent
                        if isinstance(structured, dict) and structured.get("instruction"):
                            instruction_text = structured["instruction"]
                            del structured["instruction"]
                            structured["instruction_ref"] = instruction_id
                        
                        # Check content list
                        for item in content_list:
                            if item.get("type") == "text":
                                txt_data = item.get("text", {})
                                if isinstance(txt_data, dict) and txt_data.get("instruction"):
                                    instruction_text = txt_data["instruction"]
                                    del txt_data["instruction"]
                                    txt_data["instruction_ref"] = instruction_id
                        
                        if instruction_text:
                            content_mapping["instructions"][instruction_id] = instruction_text

                # FAQ Logic
                if tool_name == "get_faqs" and isinstance(tool_output, dict):
                    content_list = tool_output.get("content", [])
                    for item in content_list:
                        if item.get("type") == "text":
                            txt_data = item.get("text", {})
                            if isinstance(txt_data, dict):
                                faqs_list = txt_data.get("faqs_list", [])
                                for faq in faqs_list:
                                    faq_id = faq.get("id")
                                    if faq_id:
                                        faq_entry = {
                                            "title": faq.get("title"),
                                            "body": faq.get("body") or faq.get("faq_body"),
                                        }
                                        if faq.get("faq_slug"):
                                            faq_entry["faq_slug"] = faq.get("faq_slug")
                                        
                                        content_mapping["faqs"][faq_id] = faq_entry
                                        
                                        if "body" in faq: del faq["body"]
                                        if "faq_body" in faq: del faq["faq_body"]
                                        faq["faq_ref"] = faq_id
        
        optimized_traces.append(new_trace)

    with open(OUTPUT_FILE_2, 'w') as f:
        for t in optimized_traces:
            f.write(json.dumps(t) + '\n')

    with open(MAPPING_FILE, 'w') as f:
        json.dump(content_mapping, f, indent=2)

    print("Done.")

if __name__ == "__main__":
    main()
