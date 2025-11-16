"""
MCP Test Script
----------------
Tests MCP client connection and lists all available tools from the MCP server.

Usage:
    python mcp_test.py
    # or
    uv run mcp_test.py

Environment variables:
    MCP_TOOL_BASE_URL  (default: http://localhost:9100)
    MCP_TOOL_KEY       (default: mcp-test-key)
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add parent directory to path to import mcp_client
sys.path.insert(0, str(Path(__file__).parent))

from clients.mcp_client import MCPClient

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mcp_test")


async def main():
    """Main test function"""
    # Get configuration from environment
    base_url = os.getenv("MCP_TOOL_BASE_URL", "http://localhost:9100")
    tool_key = os.getenv("MCP_TOOL_KEY", "mcp-test-key")
    
    print("=" * 60)
    print("MCP Client Test")
    print("=" * 60)
    print(f"Connecting to MCP server at: {base_url}")
    print(f"Using tool key: {tool_key}")
    print()
    
    client = None
    try:
        # Initialize client
        client = MCPClient(base_url=base_url, tool_key=tool_key, timeout=10.0)
        print(f"✓ Client initialized (transport: {client.transport_mode})")
        
        # Connect to server
        print("Connecting to server...")
        await client.initialize()
        print("✓ Connected successfully")
        print()
        
        # List all available tools
        print("Fetching available tools...")
        tools_list = []
        
        # Try multiple methods to get tools
        try:
            # Method 1: Use FastMCP client's built-in tool listing if available
            if client.transport_mode == "fastmcp" and hasattr(client._impl, 'client'):
                fastmcp_client = client._impl.client
                # FastMCP client should have tools accessible
                if hasattr(fastmcp_client, 'list_tools'):
                    tools_result = await fastmcp_client.list_tools()
                    if isinstance(tools_result, dict) and "tools" in tools_result:
                        tools_list = tools_result["tools"]
                    elif isinstance(tools_result, list):
                        tools_list = tools_result
                elif hasattr(fastmcp_client, '_tools') or hasattr(fastmcp_client, 'tools'):
                    # Try to access tools directly
                    tools_attr = getattr(fastmcp_client, 'tools', None) or getattr(fastmcp_client, '_tools', None)
                    if tools_attr:
                        if isinstance(tools_attr, dict):
                            tools_list = list(tools_attr.keys())
                        elif isinstance(tools_attr, list):
                            tools_list = tools_attr
        except Exception as e:
            logger.debug(f"Method 1 failed: {e}")
        
        # Method 2: Try calling list_tools tool
        if not tools_list:
            try:
                result = await client.call_tool("list_tools", {})
                if "tools" in result:
                    tools_list = result["tools"]
            except Exception as e:
                logger.debug(f"Method 2 (list_tools tool) failed: {e}")
        
        # Method 3: Fetch manifest via HTTP
        if not tools_list:
            try:
                import httpx
                async with httpx.AsyncClient() as http_client:
                    # Try different manifest endpoints
                    for manifest_path in ["/mcp/manifest", "/manifest", "/mcp"]:
                        manifest_url = f"{base_url}{manifest_path}"
                        try:
                            resp = await http_client.get(manifest_url, headers={"X-MCP-TOOL-KEY": tool_key})
                            if resp.status_code == 200:
                                manifest = resp.json()
                                # Extract tools from manifest
                                if "tools" in manifest:
                                    tools_list = [t.get("name") if isinstance(t, dict) else t for t in manifest["tools"]]
                                elif "resources" in manifest:
                                    # Some MCP servers return resources instead
                                    pass
                                if tools_list:
                                    break
                        except Exception:
                            continue
            except Exception as manifest_error:
                logger.debug(f"Method 3 (manifest) failed: {manifest_error}")
        
        # Method 4: Try to discover tools by calling known tool names
        if not tools_list:
            known_tools = ["balance", "transactions", "transfer", "list_tools"]
            print("Attempting to discover tools by probing known names...")
            discovered = []
            for tool_name in known_tools:
                try:
                    # Try calling with minimal payload to see if tool exists
                    # Use a safe call that won't execute
                    await client.call_tool(tool_name, {})
                    discovered.append(tool_name)
                except Exception as e:
                    # Check if error is "unknown tool" vs other error
                    error_str = str(e).lower()
                    if "unknown tool" not in error_str and "not found" not in error_str:
                        # Tool exists but call failed for other reason
                        discovered.append(tool_name)
            if discovered:
                tools_list = discovered
        
        # Display results
        if tools_list:
            print(f"✓ Found {len(tools_list)} tool(s):")
            print()
            for i, tool_name in enumerate(tools_list, 1):
                print(f"  {i}. {tool_name}")
            print()
        else:
            print("⚠ Could not retrieve tool list using any method")
            print()
            print("Troubleshooting:")
            print("  1. Make sure the MCP server is running")
            print("  2. Check that MCP_TOOL_BASE_URL is correct")
            print("  3. Verify the server exposes tools correctly")
            print()
        
    except Exception as e:
        logger.exception("Test failed")
        print(f"✗ Error: {e}")
        print()
        print("Make sure the MCP server is running:")
        print("  cd mcp-tools/src")
        print("  python mcp_server.py")
        sys.exit(1)
        
    finally:
        # Clean up
        if client:
            try:
                await client.close()
                print()
                print("✓ Client closed")
            except Exception as e:
                logger.warning(f"Error closing client: {e}")
    
    print()
    print("=" * 60)
    print("Test completed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

