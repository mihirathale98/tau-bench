# Add near the top
ALLOWED_MESSAGE_KEYS = {"role", "content", "name", "tool_call_id", "tool_calls"}

def coerce_content_to_str(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # Some 4o variants can return content as a list of parts; pull any text
    if isinstance(content, list):
        parts = []
        for p in content:
            # common shapes: {"type":"text","text":"..."} or {"text":"..."}
            if isinstance(p, dict) and "text" in p:
                parts.append(str(p["text"]))
            else:
                parts.append(str(p))
        return "\n".join(parts)
    # last resort
    return str(content)

def sanitize_tool_calls(tool_calls):
    if not tool_calls:
        return None
    clean = []
    for tc in tool_calls:
        # Expected minimal structure
        clean.append({
            "id": tc.get("id", ""),
            "type": "function",
            "function": {
                "name": tc.get("function", {}).get("name", ""),
                "arguments": tc.get("function", {}).get("arguments", "{}"),
            },
        })
    return clean

def sanitize_message(msg: dict) -> dict:
    role = msg.get("role", "assistant")
    content = coerce_content_to_str(msg.get("content"))
    tool_calls = sanitize_tool_calls(msg.get("tool_calls"))

    clean = {"role": role, "content": content}
    if tool_calls:
        clean["tool_calls"] = tool_calls
    # Only include name/tool_call_id for tool messages if present
    if role == "tool":
        if "name" in msg:
            clean["name"] = msg["name"]
        if "tool_call_id" in msg:
            clean["tool_call_id"] = msg["tool_call_id"]
    return clean
