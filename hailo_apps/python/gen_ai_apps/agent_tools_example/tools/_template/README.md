# Tool Template

This is a template for creating new tools. Copy this entire directory to create a new tool.

## Quick Start

```bash
# Copy the template
cp -r tools/_template tools/my_new_tool

# Edit the tool implementation
# Update tools/my_new_tool/tool.py with your logic

# Update the configuration
# Edit tools/my_new_tool/config.yaml with prompts and examples
```

## Directory Structure

```
my_new_tool/
├── __init__.py       # Re-exports tool interface
├── tool.py           # Tool implementation
├── config.yaml       # Prompts, examples, test cases
├── contexts/         # Saved context states (auto-generated)
└── README.md         # Tool documentation (optional)
```

## Required Exports

Your tool module must export these:

| Export | Type | Description |
|--------|------|-------------|
| `name` | `str` | Unique tool identifier |
| `description` | `str` | LLM-facing description with usage instructions |
| `display_description` | `str` | User-facing description for CLI |
| `schema` | `dict` | JSON schema for parameters |
| `TOOLS_SCHEMA` | `list` | OpenAI function calling format |
| `run` | `function` | Execution function: `(dict) -> dict` |

## Optional Exports

| Export | Type | Description |
|--------|------|-------------|
| `initialize_tool` | `function` | Called once when tool is loaded |
| `cleanup_tool` | `function` | Called when agent shuts down |
| `CONFIG_PATH` | `Path` | Path to config.yaml |

## run() Function

The `run()` function must:

1. Accept a dictionary of parameters
2. Return a dictionary with:
   - `ok: bool` - Success status
   - `result: Any` - Result value (if ok=True)
   - `error: str` - Error message (if ok=False)

Example:

```python
def run(input_data: dict[str, Any]) -> dict[str, Any]:
    param = input_data.get("my_param", "")

    if not param:
        return {"ok": False, "error": "Missing required 'my_param'"}

    result = process(param)
    return {"ok": True, "result": result}
```

## config.yaml Sections

| Section | Purpose |
|---------|---------|
| `persona` | Defines assistant role and constraints |
| `capabilities` | List of what the tool can do |
| `output_format` | Response formatting rules |
| `tool_instructions` | Detailed LLM instructions |
| `few_shot_examples` | Examples for context priming |
| `test_cases` | Benchmarking test cases |
| `metadata` | Author, date, description |

## Best Practices

1. **Description**: Include CRITICAL warnings and exact usage format
2. **Schema**: Follow OpenAI format, avoid `default`/`minimum` fields
3. **Examples**: Include 2-3 few-shot examples covering common cases
4. **Tests**: Cover success, edge cases, and no-tool scenarios
5. **Validation**: Validate all inputs in `run()` function

