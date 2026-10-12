import json

from storymem_policy.contract import get_tool_schemas


if __name__ == "__main__":
    print(json.dumps(get_tool_schemas(), indent=2))
