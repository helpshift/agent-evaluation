
import json
import os
from collections import defaultdict

TRACE_FILE = '/Users/rinit.lulla/Documents/GitHub/agent-evaluation/gcs_data/all_agents_traces.jsonl'

def analyze_usage():
    if not os.path.exists(TRACE_FILE):
        print(f"File not found: {TRACE_FILE}")
        return

    # Map (profile_id, domain, instruction_id, version) -> count
    usage_stats = defaultdict(int)
    
    # Also track what instruction IDs exist
    instruction_ids_found = set()

    with open(TRACE_FILE, 'r') as f:
        for line in f:
            try:
                trace = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Try to find profile_id and domain from trace context if possible
            trace_profile = trace.get('profile_id')
            trace_domain = trace.get('domain')

            agent_infos = trace.get('agent_info', [])
            for agent in agent_infos:
                tool_infos = agent.get('tool_info', [])
                for tool in tool_infos:
                    tool_name = tool.get('tool_name')
                    tool_args = tool.get('tool_args', {})
                    
                    if tool_name == 'get_usecase_instruction':
                        inst_id = tool_args.get('instruction_id')
                        inst_ver = tool_args.get('instruction_version')
                        domain = tool_args.get('domain') or trace_domain
                        # profile_id is usually not in get_usecase_instruction args, but implied by session
                        # We use trace_profile if available
                        profile = trace_profile
                        
                        if inst_id:
                            key = (profile, domain, inst_id, inst_ver)
                            usage_stats[key] += 1
                            instruction_ids_found.add(inst_id)

    print(f"Found {len(usage_stats)} unique combinations of usage.")
    print(f"Total unique instruction IDs: {len(instruction_ids_found)}")
    print("\nTop 20 Usages:")
    sorted_usage = sorted(usage_stats.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Profile ID':<60} | {'Domain':<30} | {'Instruction ID':<60} | {'Ver':<5} | {'Count'}")
    print("-" * 170)
    
    for (prof, dom, iid, ver), count in sorted_usage[:20]:
        prof_str = str(prof) if prof else "None"
        dom_str = str(dom) if dom else "None"
        iid_str = str(iid)
        ver_str = str(ver)
        print(f"{prof_str:<60} | {dom_str:<30} | {iid_str:<60} | {ver_str:<5} | {count}")

if __name__ == "__main__":
    analyze_usage()
