#!/usr/bin/env python3
"""
FastMCP Server for Dataiku
Server with text processing + utility tools
Compatible with Dataiku MCP Tool (streamable-http)
"""

import sys
import json
import re
from typing import Dict, Any

# ==========================================================
# IMPORT FASTMCP (WAJIB)
# ==========================================================
try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:
    print("ERROR: fastmcp belum terinstall. Install dengan: pip install fastmcp", file=sys.stderr)
    print(f"Detail error: {e}", file=sys.stderr)
    sys.exit(1)

# ==========================================================
# KONFIGURASI SESUAI TEMPLATE DATAIKU
# ==========================================================
MCP_SERVER_NAME = "Dataiku MCP Server"
MCP_SERVER_INSTRUCTIONS = "Server ini menyediakan tools untuk text processing & formatting"
HOST = "0.0.0.0"
PORT = 58000     # bisa Anda ubah sesuai kebutuhan

# ==========================================================
# INISIALISASI MCP SERVER (SESUAI TEMPLATE)
# ==========================================================
mcp = FastMCP(
    name=MCP_SERVER_NAME,
    instructions=MCP_SERVER_INSTRUCTIONS,
    host=HOST,
    port=PORT
)

# ==========================================================
# TEXT PROCESSING TOOLS
# ==========================================================

@mcp.tool()
async def process_text(text: str, operation: str) -> str:
    """
    Memproses teks berdasarkan operasi tertentu.
    """

    result = {
        "original": text,
        "operation": operation,
        "result": None
    }

    if operation == "lowercase":
        result["result"] = text.lower()

    elif operation == "uppercase":
        result["result"] = text.upper()

    elif operation == "clean":
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        result["result"] = ' '.join(cleaned.split())

    elif operation == "tokenize":
        tokens = text.split()
        result["result"] = tokens
        result["token_count"] = len(tokens)

    else:
        raise ValueError(f"Unknown operation: {operation}")

    return json.dumps(result, indent=2)


@mcp.tool()
async def analyze_sentiment(text: str) -> str:
    """
    Analisis sentimen sederhana.
    """

    positive_words = [
        'bagus', 'baik', 'senang', 'suka', 'hebat',
        'mantap', 'keren', 'excellent', 'good', 'great',
        'amazing', 'wonderful', 'happy', 'love'
    ]

    negative_words = [
        'buruk', 'jelek', 'sedih', 'benci', 'kecewa',
        'mengecewakan', 'bad', 'terrible', 'awful',
        'horrible', 'hate', 'sad', 'angry', 'disappointed'
    ]

    text_lower = text.lower()
    words = text_lower.split()

    positive_count = sum(1 for w in words if w in positive_words)
    negative_count = sum(1 for w in words if w in negative_words)

    if positive_count + negative_count == 0:
        sentiment = "neutral"
        score = 0.0
    elif positive_count > negative_count:
        sentiment = "positive"
        score = positive_count / (positive_count + negative_count)
    elif negative_count > positive_count:
        sentiment = "negative"
        score = -(negative_count / (positive_count + negative_count))
    else:
        sentiment = "neutral"
        score = 0.0

    result = {
        "text": text,
        "sentiment": sentiment,
        "score": round(score, 2),
        "positive_words_found": positive_count,
        "negative_words_found": negative_count,
        "word_count": len(words)
    }

    return json.dumps(result, indent=2)


# ==========================================================
# UTILITY TOOLS
# ==========================================================

@mcp.tool()
async def format_data(data: str, format_type: str) -> str:
    """
    Format data ke JSON/CSV/Table
    """

    result = {
        "original": data,
        "format": format_type,
        "result": None
    }

    try:
        parsed = json.loads(data)

        if format_type == "json":
            result["result"] = json.dumps(parsed, indent=2)

        elif format_type == "csv":
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                headers = list(parsed[0].keys())
                csv_lines = [",".join(headers)]
                for item in parsed:
                    csv_lines.append(",".join(str(item.get(h, "")) for h in headers))
                result["result"] = "\n".join(csv_lines)
            else:
                result["result"] = str(parsed)

        elif format_type == "table":
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                headers = list(parsed[0].keys())
                table = [" | ".join(headers)]
                table.append("-" * len(table[0]))
                for item in parsed:
                    table.append(" | ".join(str(item.get(h, "")) for h in headers))
                result["result"] = "\n".join(table)
            else:
                result["result"] = str(parsed)

        else:
            raise ValueError(f"Unknown format: {format_type}")

    except json.JSONDecodeError:
        result["result"] = data

    return json.dumps(result, indent=2)


@mcp.tool()
async def validate_input(data: str, schema_type: str) -> str:
    """
    Validasi input: email, URL, phone, date
    """

    result = {
        "data": data,
        "schema_type": schema_type,
        "is_valid": False,
        "message": ""
    }

    if schema_type == "email":
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        result["is_valid"] = bool(re.match(pattern, data))
        result["message"] = "Valid email" if result["is_valid"] else "Invalid email"

    elif schema_type == "url":
        pattern = r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$'
        result["is_valid"] = bool(re.match(pattern, data))
        result["message"] = "Valid URL" if result["is_valid"] else "Invalid URL"

    elif schema_type == "phone":
        pattern = r'^(\+\d{1,3}[- ]?)?\d{8,15}$'
        cleaned = re.sub(r'[\s\-\(\)]', '', data)
        result["is_valid"] = bool(re.match(pattern, cleaned))
        result["message"] = "Valid phone" if result["is_valid"] else "Invalid phone"

    elif schema_type == "date":
        patterns = [
            r'^\d{4}-\d{2}-\d{2}$',
            r'^\d{2}/\d{2}/\d{4}$',
            r'^\d{2}-\d{2}-\d{4}$'
        ]
        result["is_valid"] = any(re.match(p, data) for p in patterns)
        result["message"] = "Valid date" if result["is_valid"] else "Invalid date"

    else:
        result["message"] = f"Unknown schema type: {schema_type}"

    return json.dumps(result, indent=2)


# ==========================================================
# RESOURCE EXAMPLE
# ==========================================================

@mcp.resource("config://settings")
def get_settings() -> str:
    config = {
        "server_name": MCP_SERVER_NAME,
        "version": "1.0.0",
        "tools": ["process_text", "analyze_sentiment", "format_data", "validate_input"],
        "description": "Custom MCP Server untuk Dataiku"
    }
    return json.dumps(config, indent=2)


# ==========================================================
# RUN MCP SERVER (SESUAI TEMPLATE: STREAMABLE-HTTP)
# ==========================================================
if __name__ == "__main__":
    mcp.run(transport="stdio")                   