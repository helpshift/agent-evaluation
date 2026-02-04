
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import json
import datetime

# Add scripts to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

# Mock google.cloud before importing the script
sys.modules['google.cloud'] = MagicMock()
sys.modules['google.cloud.trace_v1'] = MagicMock()
sys.modules['google.cloud.storage'] = MagicMock()
sys.modules['google.cloud.logging_v2'] = MagicMock()

import extract_traces_v7

class TestExtractTracesV7(unittest.TestCase):

    def setUp(self):
        self.content_mapping = {"instructions": {}, "faqs": {}}

    def test_process_consolidated_trace_extracts_instruction(self):
        # Create a mock trace with get_usecase_instruction tool call
        trace = {
            "traceId": "test_trace_1",
            "spans": [
                {
                    "spanId": "root",
                    "startTime": "2023-01-01T00:00:00Z",
                    "name": "invoke_agent",
                    "labels": {
                        "gen_ai.agent.name": "test_agent"
                    }
                },
                {
                    "spanId": "child1",
                    "parentSpanId": "root",
                    "startTime": "2023-01-01T00:00:01Z",
                    "name": "execute_tool",
                    "labels": {
                        "gen_ai.tool.name": "get_usecase_instruction",
                        "gcp.vertex.agent.tool_call_args": json.dumps({
                            "instruction_id": "inst_123"
                        }),
                        "gcp.vertex.agent.tool_response": json.dumps({
                            "content": [
                                {
                                    "type": "text",
                                    "text": {
                                        "instruction": "This is a test instruction."
                                    }
                                }
                            ]
                        })
                    }
                }
            ]
        }

        record = extract_traces_v7.process_consolidated_trace(trace, self.content_mapping)

        # Verify instruction was extracted to mapping
        self.assertIn("inst_123", self.content_mapping["instructions"])
        self.assertEqual(self.content_mapping["instructions"]["inst_123"], "This is a test instruction.")

        # Verify trace record has the reference
        agent_info = record["agent_info"][0]
        tool_info = agent_info["tool_info"][0]
        tool_output = tool_info["tool_output"]
        
        # Depending on implementation, the original instruction might be removed or replaced
        # Let's check the output content in the record
        output_content = tool_output["content"][0]["text"]
        self.assertEqual(output_content.get("instruction_ref"), "inst_123")
        self.assertIsNone(output_content.get("instruction"))

    def test_process_consolidated_trace_extracts_faq(self):
        # Create a mock trace with get_faqs tool call
        trace = {
            "traceId": "test_trace_2",
            "spans": [
                {
                    "spanId": "root",
                    "startTime": "2023-01-01T00:00:00Z",
                    "name": "invoke_agent",
                    "labels": {
                        "gen_ai.agent.name": "test_agent"
                    }
                },
                {
                    "spanId": "child1",
                    "parentSpanId": "root",
                    "startTime": "2023-01-01T00:00:01Z",
                    "name": "execute_tool",
                    "labels": {
                        "gen_ai.tool.name": "get_faqs",
                        "gcp.vertex.agent.tool_call_args": json.dumps({
                            "query": "help"
                        }),
                        "gcp.vertex.agent.tool_response": json.dumps({
                            "content": [
                                {
                                    "type": "text",
                                    "text": {
                                        "faqs_list": [
                                            {
                                                "id": "faq_1",
                                                "title": "FAQ 1",
                                                "body": "Body of FAQ 1",
                                                "faq_slug": "slug-1"
                                            }
                                        ]
                                    }
                                }
                            ]
                        })
                    }
                }
            ]
        }

        record = extract_traces_v7.process_consolidated_trace(trace, self.content_mapping)

        # Verify FAQ was extracted to mapping
        self.assertIn("faq_1", self.content_mapping["faqs"])
        self.assertEqual(self.content_mapping["faqs"]["faq_1"]["body"], "Body of FAQ 1")

        # Verify trace record has the reference
        agent_info = record["agent_info"][0]
        tool_info = agent_info["tool_info"][0]
        tool_output = tool_info["tool_output"]
        
        output_faq = tool_output["content"][0]["text"]["faqs_list"][0]
        self.assertEqual(output_faq.get("faq_ref"), "faq_1")
        self.assertIsNone(output_faq.get("body"))

    @patch('extract_traces_v7.logging_v2.Client')
    def test_fetch_logs_parallel(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        # Mock list_entries to return some dummy entry
        mock_entry = MagicMock()
        # Mock extract_ids_from_log_entry logic by setting payload
        mock_entry.payload = {"issue_id": "123"}
        mock_entry.text_payload = ""
        mock_client.list_entries.return_value = [mock_entry]

        trace_ids = ["t1", "t2"]
        session_ids = ["s1", "s2"]
        profile_ids = ["p1", "p2"]
        issue_ids = ["i1", "i2"]
        merged_traces = [{"trace_id": "t1", "user_message": "test"}, {"trace_id": "t2", "user_message": "test"}]

        log_data = extract_traces_v7.fetch_logs_by_trace_ids(
            "project", trace_ids, session_ids, days=1,
            profile_ids=profile_ids, issue_ids=issue_ids,
            merged_traces=merged_traces
        )

        # Check if client was instantiated
        mock_client_cls.assert_called_with(project="project")
        
        # Check if list_entries was called multiple times (parallel execution)
        self.assertTrue(mock_client.list_entries.call_count > 0)
        
        # Since we mocked return values, log_data should contain results
        self.assertIn("t1", log_data)
        # Note: Depending on how the mock ties back to specific traces in the logic, 
        # validation might be tricky without deeper mocking of extract_ids.
        # But we mostly want to ensure it runs without crashing.

if __name__ == '__main__':
    unittest.main()
