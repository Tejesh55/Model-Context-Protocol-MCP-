from typing import Dict, Any, Tuple, Type
from langchain_core.pydantic_v1 import BaseModel, create_model

# Mapping from JSON Schema types to Python types
TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}

def create_pydantic_model_from_schema(
    model_name: str,
    input_schema: Dict[str, Any]
) -> Type[BaseModel]:
    """Create a Pydantic model from JSON Schema input"""
    properties = input_schema.get("properties", {})
    required_fields = set(input_schema.get("required", []))

    fields: Dict[str, Tuple[Any, Any]] = {}

    for field_name, field_info in properties.items():
        json_type = field_info.get("type", "string")
        py_type = TYPE_MAP.get(json_type, str)  # default to str
        default = ... if field_name in required_fields else None
        fields[field_name] = (py_type, default)

    return create_model(model_name, **fields)
