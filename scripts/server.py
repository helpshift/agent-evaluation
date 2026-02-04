#!/usr/bin/env python3
"""
HTTP server wrapper for Cloud Run to trigger trace extraction job.
"""

import os
import subprocess
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime, timedelta, timezone
from google.cloud import trace_v1, logging_v2, storage
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.json_format import MessageToDict
from google.api_core.exceptions import ResourceExhausted
import concurrent.futures


class JobHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/poc" or self.path == "/poc/":
            self.run_poc_test()
        elif self.path == "/poc-full" or self.path == "/poc-full/":
            self.run_full_poc_test()
        elif self.path == "/health":
            self.health_check()
        else:
            self.run_extraction_job()

    def health_check(self):
        """Health check endpoint."""
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "healthy"}).encode())

    def run_full_poc_test(self):
        """
        FULL POC: Tests the complete extraction pipeline including:
        1. Cloud Trace API (listing traces)
        2. Cloud Logging API (fetching logs)
        3. Data processing
        4. GCS write operations

        This gives the TRUE production cost estimate.
        """
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

        print("="*70)
        print("🔴 FULL PRODUCTION POC TEST")
        print("="*70)

        PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "product-research-460317")
        AGENT_ENGINE_ID = os.environ.get("AGENT_ENGINE_ID", "3122210601628073984")
        GCS_BUCKET = os.environ.get("GCS_BUCKET", "evaluation-research")

        trace_client = trace_v1.TraceServiceClient()
        logging_client = logging_v2.Client(project=PROJECT_ID)
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(GCS_BUCKET)

        past_date = datetime.now(timezone.utc) - timedelta(days=1)
        start_time_proto = Timestamp()
        start_time_proto.FromDatetime(past_date)

        results = {
            "test_started": datetime.now().isoformat(),
            "environment": "Cloud Run - FULL PIPELINE",
            "project_id": PROJECT_ID,
            "agent_engine_id": AGENT_ENGINE_ID,
            "phases": {}
        }

        total_start = time.time()

        # ===== PHASE 1: List Traces =====
        print("\n📊 PHASE 1: Listing Traces (Cloud Trace API)")
        phase1_start = time.time()
        traces_fetched = 0
        trace_quota_hits = 0
        trace_data = []

        for i in range(5):  # 5 batches of 100 = 500 traces
            try:
                request = trace_v1.ListTracesRequest(
                    project_id=PROJECT_ID,
                    view=trace_v1.ListTracesRequest.ViewType.COMPLETE,
                    page_size=100,
                    start_time=start_time_proto,
                    filter=f"+service.name:{AGENT_ENGINE_ID}"
                )

                pager = trace_client.list_traces(request=request, timeout=60.0)
                batch_traces = []
                for trace in pager:
                    trace_dict = MessageToDict(trace._pb)
                    batch_traces.append(trace_dict)
                    if len(batch_traces) >= 100:
                        break

                traces_fetched += len(batch_traces)
                trace_data.extend(batch_traces)
                print(f"  Batch {i+1}: +{len(batch_traces)} traces | Total: {traces_fetched}")

            except ResourceExhausted:
                trace_quota_hits += 1
                print(f"  Batch {i+1}: QUOTA EXHAUSTED")
                time.sleep(5)

        phase1_time = time.time() - phase1_start
        results["phases"]["trace_listing"] = {
            "traces_fetched": traces_fetched,
            "quota_hits": trace_quota_hits,
            "duration_seconds": round(phase1_time, 1),
            "rate_per_minute": round((traces_fetched / phase1_time) * 60, 0) if phase1_time > 0 else 0
        }

        # ===== PHASE 2: Fetch Logs (Cloud Logging API) =====
        print("\n📊 PHASE 2: Fetching Logs (Cloud Logging API)")
        phase2_start = time.time()
        logs_fetched = 0
        log_queries = 0
        log_quota_hits = 0

        # Get unique session IDs from traces
        session_ids = set()
        for trace in trace_data[:50]:  # Test with first 50 traces
            spans = trace.get("spans", [])
            for span in spans:
                labels = span.get("labels", {})
                sid = labels.get("gcp.vertex.agent.session_id")
                if sid:
                    session_ids.add(sid)

        print(f"  Found {len(session_ids)} unique session IDs")

        time_filter = f'timestamp >= "{past_date.isoformat()}"'

        for sid in list(session_ids)[:10]:  # Test 10 log queries
            try:
                log_queries += 1
                filter_str = f'jsonPayload.adk_session_id="{sid}" AND {time_filter}'
                entries = list(logging_client.list_entries(filter_=filter_str, max_results=10))
                logs_fetched += len(entries)
                print(f"  Query {log_queries}: {len(entries)} log entries")
                time.sleep(1)  # Respect rate limit (60/min)
            except Exception as e:
                log_quota_hits += 1
                print(f"  Query {log_queries}: ERROR - {str(e)[:50]}")

        phase2_time = time.time() - phase2_start
        results["phases"]["log_fetching"] = {
            "log_queries": log_queries,
            "logs_fetched": logs_fetched,
            "quota_hits": log_quota_hits,
            "duration_seconds": round(phase2_time, 1),
            "queries_per_minute": round((log_queries / phase2_time) * 60, 0) if phase2_time > 0 else 0
        }

        # ===== PHASE 3: Data Processing =====
        print("\n📊 PHASE 3: Processing Data")
        phase3_start = time.time()
        processed_records = []

        for trace in trace_data[:100]:  # Process 100 traces
            record = {
                "trace_id": trace.get("traceId"),
                "timestamp": None,
                "session_id": None,
                "processed_at": datetime.now().isoformat()
            }

            spans = trace.get("spans", [])
            if spans:
                record["timestamp"] = spans[0].get("startTime")
                for span in spans:
                    labels = span.get("labels", {})
                    if not record["session_id"]:
                        record["session_id"] = labels.get("gcp.vertex.agent.session_id")

            processed_records.append(record)

        phase3_time = time.time() - phase3_start
        results["phases"]["data_processing"] = {
            "records_processed": len(processed_records),
            "duration_seconds": round(phase3_time, 1),
            "records_per_minute": round((len(processed_records) / phase3_time) * 60, 0) if phase3_time > 0 else 0
        }

        # ===== PHASE 4: GCS Write =====
        print("\n📊 PHASE 4: Writing to GCS")
        phase4_start = time.time()

        try:
            output_data = json.dumps(processed_records, indent=2)
            blob = bucket.blob(f"poc_test/poc_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            blob.upload_from_string(output_data)
            gcs_write_success = True
            gcs_bytes_written = len(output_data)
            print(f"  Written {gcs_bytes_written} bytes to GCS")
        except Exception as e:
            gcs_write_success = False
            gcs_bytes_written = 0
            print(f"  GCS write failed: {e}")

        phase4_time = time.time() - phase4_start
        results["phases"]["gcs_write"] = {
            "success": gcs_write_success,
            "bytes_written": gcs_bytes_written,
            "duration_seconds": round(phase4_time, 1)
        }

        # ===== FINAL SUMMARY =====
        total_time = time.time() - total_start

        # Calculate REAL throughput (end-to-end)
        traces_per_minute = (traces_fetched / total_time) * 60 if total_time > 0 else 0

        # Cost calculation for 6M issues
        TOTAL_ISSUES = 6_000_000
        hours_needed = (TOTAL_ISSUES / traces_per_minute) / 60 if traces_per_minute > 0 else 0
        cloud_run_cost = hours_needed * 0.1908
        gcs_storage_gb = (TOTAL_ISSUES * 7.7) / (1024 * 1024)
        gcs_cost = gcs_storage_gb * 0.023

        results["summary"] = {
            "total_duration_seconds": round(total_time, 1),
            "traces_processed": traces_fetched,
            "end_to_end_throughput_per_minute": round(traces_per_minute, 0),
            "bottleneck": "Cloud Logging API (60 queries/min)" if log_queries > 5 else "Cloud Trace API",
            "cost_estimate_6m_issues": {
                "hours_needed": round(hours_needed, 1),
                "days_needed": round(hours_needed / 24, 1),
                "cloud_run_cost_usd": round(cloud_run_cost, 2),
                "gcs_storage_gb": round(gcs_storage_gb, 1),
                "gcs_cost_usd": round(gcs_cost, 2),
                "total_cost_usd": round(cloud_run_cost + gcs_cost, 2)
            }
        }
        results["test_completed"] = datetime.now().isoformat()

        print()
        print("="*70)
        print("🔴 FULL POC RESULTS")
        print("="*70)
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Traces processed: {traces_fetched}")
        print(f"  ⭐ END-TO-END THROUGHPUT: {traces_per_minute:.0f} traces/minute")
        print(f"  💰 6M ISSUES: {hours_needed:.1f} hours = ${cloud_run_cost + gcs_cost:.2f}")
        print("="*70)

        self.wfile.write(json.dumps(results, indent=2).encode())

    def run_poc_test(self):
        """Run POC throughput test (trace listing only)."""
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

        print("="*70)
        print("🚀 POC THROUGHPUT TEST ON CLOUD RUN")
        print("="*70)

        PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "product-research-460317")
        AGENT_ENGINE_ID = os.environ.get("AGENT_ENGINE_ID", "3122210601628073984")

        client = trace_v1.TraceServiceClient()
        past_date = datetime.now(timezone.utc) - timedelta(days=1)
        start_time_proto = Timestamp()
        start_time_proto.FromDatetime(past_date)

        results = {
            "test_started": datetime.now().isoformat(),
            "environment": "Cloud Run",
            "project_id": PROJECT_ID,
            "agent_engine_id": AGENT_ENGINE_ID,
            "calls": []
        }

        total_traces = 0
        quota_hits = 0
        test_start = time.time()

        for i in range(12):
            call_start = time.time()
            try:
                request = trace_v1.ListTracesRequest(
                    project_id=PROJECT_ID,
                    view=trace_v1.ListTracesRequest.ViewType.COMPLETE,
                    page_size=100,
                    start_time=start_time_proto,
                    filter=f"+service.name:{AGENT_ENGINE_ID}"
                )

                pager = client.list_traces(request=request, timeout=60.0)
                count = 0
                for trace in pager:
                    count += 1
                    if count >= 100:
                        break

                latency = time.time() - call_start
                total_traces += count
                elapsed = time.time() - test_start
                rate = (total_traces / elapsed) * 60

                call_result = {
                    "call": i + 1,
                    "status": "success",
                    "traces": count,
                    "latency_seconds": round(latency, 2),
                    "total_traces": total_traces,
                    "rate_per_minute": round(rate, 0)
                }
                results["calls"].append(call_result)

                print(f"[{elapsed:5.1f}s] Call {i+1:2d}: ✓ +{count} traces | Total: {total_traces} | Rate: {rate:.0f}/min")

            except ResourceExhausted:
                quota_hits += 1
                call_result = {
                    "call": i + 1,
                    "status": "quota_exhausted",
                    "traces": 0
                }
                results["calls"].append(call_result)
                print(f"[{time.time()-test_start:5.1f}s] Call {i+1:2d}: ✗ QUOTA EXHAUSTED")
                time.sleep(5)

            except Exception as e:
                call_result = {
                    "call": i + 1,
                    "status": "error",
                    "error": str(e)
                }
                results["calls"].append(call_result)
                print(f"[{time.time()-test_start:5.1f}s] Call {i+1:2d}: ✗ Error: {e}")

        elapsed_total = time.time() - test_start
        final_rate = (total_traces / elapsed_total) * 60 if elapsed_total > 0 else 0

        TOTAL_ISSUES = 6_000_000
        hours_needed = (TOTAL_ISSUES / final_rate) / 60 if final_rate > 0 else 0
        cloud_run_cost = hours_needed * 0.1908
        gcs_cost = 1.01
        total_cost = cloud_run_cost + gcs_cost

        results["summary"] = {
            "test_duration_seconds": round(elapsed_total, 1),
            "total_traces": total_traces,
            "quota_hits": quota_hits,
            "throughput_per_minute": round(final_rate, 0),
            "cost_estimate_6m_issues": {
                "hours_needed": round(hours_needed, 1),
                "days_needed": round(hours_needed / 24, 1),
                "cloud_run_cost_usd": round(cloud_run_cost, 2),
                "gcs_cost_usd": round(gcs_cost, 2),
                "total_cost_usd": round(total_cost, 2)
            }
        }
        results["test_completed"] = datetime.now().isoformat()

        print()
        print("="*70)
        print(f"⭐ THROUGHPUT: {final_rate:.0f} traces/minute")
        print(f"💰 6M ISSUES: {hours_needed:.1f} hours = ${total_cost:.2f}")
        print("="*70)

        self.wfile.write(json.dumps(results, indent=2).encode())

    def run_extraction_job(self):
        """Handle GET requests to trigger the trace extraction job."""
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

        response = {
            "status": "running",
            "job": "trace_extraction",
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Trace extraction job started..."
        }

        self.wfile.write(json.dumps(response, indent=2).encode() + b"\n\n")
        self.wfile.flush()

        try:
            print("Starting trace extraction job...")
            result = subprocess.run(
                ["python", "extract_traces_v7.py",
                 "--incremental",
                 "--limit", "1000",
                 "--output", "incremental_traces.jsonl"],
                capture_output=True,
                text=True,
                timeout=3600
            )

            output_response = {
                "status": "completed" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }

            self.wfile.write(b"\n=== Job Results ===\n")
            self.wfile.write(json.dumps(output_response, indent=2).encode())
            print(f"Job completed with return code: {result.returncode}")

        except subprocess.TimeoutExpired:
            error_response = {"status": "timeout", "error": "Job exceeded 1 hour timeout"}
            self.wfile.write(json.dumps(error_response, indent=2).encode())
            print("Job timed out after 1 hour")

        except Exception as e:
            error_response = {"status": "error", "error": str(e)}
            self.wfile.write(json.dumps(error_response, indent=2).encode())
            print(f"Job failed with error: {e}")

    def do_POST(self):
        """Handle POST requests (for Cloud Scheduler)."""
        self.do_GET()

    def log_message(self, format, *args):
        """Custom logging to stdout for Cloud Run."""
        print(f"{self.address_string()} - {format % args}")


def main():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("", port), JobHandler)
    print(f"Trace extraction server starting on port {port}...")
    print(f"Endpoints:")
    print(f"  /poc      - Quick trace listing test")
    print(f"  /poc-full - FULL pipeline test (traces + logs + GCS)")
    print(f"  /health   - Health check")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()


if __name__ == "__main__":
    main()
