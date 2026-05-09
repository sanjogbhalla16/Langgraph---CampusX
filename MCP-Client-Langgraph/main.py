# We will generate the code for mcp server here
from __future__ import annotations
from fastmcp import FastMCP

# Initialize the server with a name
mcp = FastMCP("arith")

def _as_number(x):
    # Accept ints/floats or numeric strings; raise clear errors otherwise
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        return float(x.strip())
    raise TypeError(f"Expected a number (int/float) or numeric string)")


# Define a tool using the @mcp.tool() decorator
@mcp.tool()
async def add(a: float, b: float) -> float:
    """Return a + b."""
    return _as_number(a) + _as_number(b)

@mcp.tool()
async def subtract(a: float, b: float) -> float:
    """Return a - b."""
    return _as_number(a) - _as_number(b)

@mcp.tool()
async def multiply(a: float, b: float) -> float:
    """Return a * b."""
    return _as_number(a) * _as_number(b)

@mcp.tool()
async def divide(a: float, b: float) -> float:
    """Return a / b. Raises on division by zero"""
    a = _as_number(a)
    b = _as_number(b)
    if b == 0:
        raise ZeroDivisionError("Division by zero is not allowed")
    return a / b

@mcp.tool()
async def power(a: float, b: float) -> float:
    """Return a ** b."""
    return _as_number(a) ** _as_number(b)

@mcp.tool()
async def modulus(a: float, b: float) -> float:
    """Return a % b. Raises on division by zero"""
    a = _as_number(a)
    b = _as_number(b)
    if b == 0:
        raise ZeroDivisionError("Division by zero is not allowed")
    return a % b

# Define a resource (read-only data)
@mcp.resource("config://app-info")
def get_info() -> str:
    return "Server Version: 1.0.0 | Status: Online"

# Run the server using stdio transport (standard for local clients like Claude Desktop)
if __name__ == "__main__":
    mcp.run(transport="stdio")
