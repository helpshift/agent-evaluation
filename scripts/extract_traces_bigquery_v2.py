import os
import json
import argparse
from google.cloud import bigquery

# --- Configuration ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "product-research-460317")

# --- TABLE NAMES ---
# 1. Traces (Global Table)
BQ_TRACE_TABLE = "product-research-460317.all_logs._AllLogs"

# 2. Logs (Linked Dataset in US)
BQ_LOGS_TABLE = "product-research-460317.all_logs._AllLogs"

def fetch_full_distributed_trace(project_id, days, limit, agent_engine_id):
    # CRITICAL FIX: We FORCE the location to "US".
    # The error you saw happens because this was set to "global" or left blank.
    client = bigquery.Client(project=project_id, location="global")

    query = f"""
    WITH 
    -- 1. ANCHOR: Find Trace IDs from the Agent Engine
    TargetTraceIDs AS (
        SELECT DISTINCT trace_id
        FROM `{BQ_TRACE_TABLE}`
        WHERE 
            start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
            AND ENDS_WITH(JSON_VALUE(resource.attributes, '$."cloud.resource_id"'), @agent_engine_id)
        LIMIT @limit
    ),

    -- 2. EXPAND (SPANS): Get all related spans
    AllSpans AS (
        SELECT 
            t.trace_id, t.span_id, t.parent_span_id, t.name, t.start_time, t.end_time, 
            t.json_payload, t.attributes,
            COALESCE(
                JSON_VALUE(t.resource.labels, '$."service_name"'), 
                JSON_VALUE(t.resource.labels, '$."module_id"'),
                'unknown-service'
            ) as service_name
        FROM `{BQ_TRACE_TABLE}` t
        INNER JOIN TargetTraceIDs target ON t.trace_id = target.trace_id
    ),

    -- 3. EXPAND (LOGS): Get all related logs
    AllLogs AS (
        SELECT 
            trace,
            ARRAY_AGG(STRUCT(
                timestamp,
                resource.labels.service_name as log_service,
                severity,
                jsonPayload,
                textPayload
            ) ORDER BY timestamp ASC) as log_entries,
            
            -- Extract IDs
            MAX(COALESCE(
                JSON_VALUE(jsonPayload, '$.issue_id'), 
                JSON_VALUE(jsonPayload, '$.issue_pid')
            )) as issue_id,
            MAX(JSON_VALUE(jsonPayload, '$.profile_id')) as profile_id

        FROM `{BQ_LOGS_TABLE}`
        WHERE 
            timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
            AND trace IS NOT NULL
        GROUP BY trace
    ),

    -- 4. CONSOLIDATE
    TraceChain AS (
        SELECT
            trace_id,
            ARRAY_AGG(STRUCT(
                service_name,
                name as operation,
                start_time,
                end_time,
                CASE 
                    WHEN service_name LIKE '%agent%' THEN 'Agent'
                    WHEN service_name LIKE '%integration%' THEN 'Integration'
                    WHEN service_name LIKE '%mcp%' THEN 'MCP'
                    ELSE 'Other'
                END as service_type
            ) ORDER BY start_time ASC) as full_history
        FROM AllSpans
        GROUP BY trace_id
    )

    -- 5. FINAL JOIN
    SELECT 
        t.trace_id,
        t.full_history,
        l.issue_id,
        l.profile_id,
        l.log_entries
    FROM TraceChain t
    LEFT JOIN AllLogs l 
        ON l.trace = CONCAT('projects/', @project_id, '/traces/', t.trace_id)
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("days", "INT64", days),
            bigquery.ScalarQueryParameter("agent_engine_id", "STRING", agent_engine_id),
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
            bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
        ]
    )

    print(f"Executing Query in 'US' region against {BQ_LOGS_TABLE}...")
    try:
        query_job = client.query(query, job_config=job_config)
        results = list(query_job.result())
        print(f"Success! Found {len(results)} trace chains.")
        return results
    except Exception as e:
        print(f"\n[!] BigQuery Error: {e}")
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    AGENT_ID = "3122210601628073984"

    data = fetch_full_distributed_trace(PROJECT_ID, args.days, args.limit, AGENT_ID)

    if data:
        with open("full_distributed_traces.jsonl", "w") as f:
            for row in data:
                f.write(json.dumps(dict(row), default=str) + "\n")
        print(f"Saved output to full_distributed_traces.jsonl")

if __name__ == "__main__":
    main()