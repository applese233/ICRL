#!/usr/bin/env bash
# Launch the online search server (SerpAPI/Serper proxy).
#
# This server provides web search capabilities for ICRL training.
# It listens on port 8000 and provides a /retrieve endpoint.
#
# Requirements:
#   - SERPAPI_KEY must be set (get from https://serpapi.com or https://serper.dev)
#   - Python deps: fastapi, uvicorn, requests
#
# Usage:
#   # Using SerpAPI (default):
#   SERPAPI_KEY=your_key bash scripts/search/run_search_server.sh
#
#   # Using Serper.dev:
#   PROVIDER=serper SERPAPI_KEY=your_key bash scripts/search/run_search_server.sh

set -euo pipefail

# --- Configurable vars ---
PROVIDER=${PROVIDER:-serpapi}   # serpapi | serper
TOPK=${TOPK:-3}
ENGINE=${SERP_ENGINE:-google}
SEARCH_URL=${SEARCH_URL:-}
SERPAPI_KEY=${SERPAPI_KEY:-}

if [[ -z "$SERPAPI_KEY" ]]; then
    echo "[ERROR] SERPAPI_KEY is required. Export SERPAPI_KEY before running." >&2
    echo "  Get your API key from:" >&2
    echo "    - SerpAPI: https://serpapi.com" >&2
    echo "    - Serper: https://serper.dev" >&2
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_PY="$SCRIPT_DIR/serp_search_server.py"

if [[ ! -f "$SERVER_PY" ]]; then
    echo "[ERROR] serp_search_server.py not found at $SERVER_PY" >&2
    exit 1
fi

echo "========================================"
echo "Starting Search Server"
echo "  Provider: $PROVIDER"
echo "  TopK: $TOPK"
echo "  Engine: $ENGINE"
echo "  Endpoint: http://0.0.0.0:8000/retrieve"
echo "========================================"

exec python "$SERVER_PY" \
    --provider "$PROVIDER" \
    ${SEARCH_URL:+--search_url "$SEARCH_URL"} \
    --serp_api_key "$SERPAPI_KEY" \
    --serp_engine "$ENGINE" \
    --topk "$TOPK" \
    --host 0.0.0.0 \
    --port 8000
