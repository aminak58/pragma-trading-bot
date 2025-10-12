"""
Test script to verify GitHub access via MCP
"""

import os

token = os.getenv("GITHUB_TOKEN")

if token:
    print(f"✅ GITHUB_TOKEN found: {token[:10]}...{token[-4:]}")
    print("✅ Length:", len(token))
else:
    print("❌ GITHUB_TOKEN not found in environment")
