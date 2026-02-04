"""
Online Search Server using SerpAPI or Serper.dev

This server provides a REST API endpoint for batch web search queries.
It supports two providers:
- SerpAPI (https://serpapi.com)
- Serper.dev (https://serper.dev)

Usage:
    python serp_search_server.py --provider serpapi --serp_api_key YOUR_KEY --topk 3
"""

import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor
import argparse
import uvicorn

parser = argparse.ArgumentParser(description="Launch online search server (SerpAPI proxy).")
parser.add_argument('--provider', type=str, default="serpapi", choices=["serpapi", "serper"],
                    help="Search provider: serpapi or serper.dev")
parser.add_argument('--search_url', type=str, default=None,
                    help="Override search URL (optional, otherwise provider default)")
parser.add_argument('--topk', type=int, default=3,
                    help="Number of results to return per query")
parser.add_argument('--serp_api_key', type=str, default=None,
                    help="SerpAPI key for online search")
parser.add_argument('--serp_engine', type=str, default="google",
                    help="SerpAPI engine for online search")
parser.add_argument('--host', type=str, default="0.0.0.0",
                    help="Host to bind FastAPI server")
parser.add_argument('--port', type=int, default=8000,
                    help="Port to bind FastAPI server")
args = parser.parse_args()


class OnlineSearchConfig:
    def __init__(
        self,
        provider: str = "serpapi",
        search_url: Optional[str] = None,
        topk: int = 3,
        serp_api_key: Optional[str] = None,
        serp_engine: Optional[str] = None,
    ):
        self.provider = provider
        # defaults per provider
        if search_url:
            self.search_url = search_url
        elif provider == "serper":
            self.search_url = "https://google.serper.dev/search"
        else:
            self.search_url = "https://serpapi.com/search"
        self.topk = topk
        self.serp_api_key = serp_api_key
        self.serp_engine = serp_engine


class OnlineSearchEngine:
    def __init__(self, config: OnlineSearchConfig):
        self.config = config

    def _search_query(self, query: str):
        if self.config.provider == "serper":
            headers = {"X-API-KEY": self.config.serp_api_key}
            payload = {"q": query}
            response = requests.post(self.config.search_url, json=payload, headers=headers)
            return response.json()
        # default: serpapi
        params = {
            "engine": self.config.serp_engine,
            "q": query,
            "api_key": self.config.serp_api_key,
        }
        response = requests.get(self.config.search_url, params=params)
        return response.json()

    def batch_search(self, queries: List[str]):
        results = []
        with ThreadPoolExecutor() as executor:
            for result in executor.map(self._search_query, queries):
                results.append(self._process_result(result))
        return results

    def _process_result(self, search_result: Dict):
        results = []

        if self.config.provider == "serper":
            organic = search_result.get("organic", []) or []
            for item in organic[: self.config.topk]:
                title = item.get("title", "No title.")
                snippet = item.get("snippet") or item.get("description") or "No snippet available."
                results.append({
                    'document': {"contents": f'"{title}"\n{snippet}'},
                })
            return results

        # serpapi format
        answer_box = search_result.get('answer_box', {})
        if answer_box:
            title = answer_box.get('title', 'No title.')
            snippet = answer_box.get('snippet', 'No snippet available.')
            results.append({
                'document': {"contents": f'"{title}"\n{snippet}'},
            })

        organic_results = search_result.get('organic_results', [])
        for _, result in enumerate(organic_results[:self.config.topk]):
            title = result.get('title', 'No title.')
            snippet = result.get('snippet', 'No snippet available.')
            results.append({
                'document': {"contents": f'"{title}"\n{snippet}'},
            })

        related_results = search_result.get('related_questions', [])
        for _, result in enumerate(related_results[:self.config.topk]):
            title = result.get('question', 'No title.')
            snippet = result.get('snippet', 'No snippet available.')
            results.append({
                'document': {"contents": f'"{title}"\n{snippet}'},
            })

        return results


app = FastAPI(title="Online Search Proxy Server")


class SearchRequest(BaseModel):
    queries: List[str]


def create_app():
    return app


config = OnlineSearchConfig(
    provider=args.provider,
    search_url=args.search_url,
    topk=args.topk,
    serp_api_key=args.serp_api_key,
    serp_engine=args.serp_engine,
)
engine = OnlineSearchEngine(config)


@app.post("/retrieve")
def search_endpoint(request: SearchRequest):
    results = engine.batch_search(request.queries)
    return {"result": results}


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
