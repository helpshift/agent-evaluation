def add(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b

def subtract(a: float, b: float) -> float:
    """Subtracts the second number from the first."""
    return a - b

def multiply(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b

def divide(a: float, b: float) -> str:
    """Divides the first number by the second. Returns result as string, or error message."""
    if b == 0:
        return "Error: Division by zero"
    return str(a / b)
