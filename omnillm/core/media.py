import base64
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from omnillm.core.types import ChatMessage


def encode_image_to_base64(image: str | bytes | Path) -> str:
    """Encodes an image input (file path, Path, bytes, data URI, or raw base64) into a pure base64 string."""
    if isinstance(image, bytes):
        return base64.b64encode(image).decode("ascii")
    if isinstance(image, Path):
        return base64.b64encode(image.read_bytes()).decode("ascii")
    if isinstance(image, str):
        if image.startswith("data:image/") and ";base64," in image:
            return image.split(";base64,", 1)[1]
        path = Path(image)
        if path.is_file():
            return base64.b64encode(path.read_bytes()).decode("ascii")
        # Already a base64 string
        return image
    raise TypeError(f"Unsupported image type: {type(image)}")


def encode_image_to_data_uri(image: str | bytes | Path, default_mime: str = "image/jpeg") -> str:
    """Converts an image input into an RFC 2397 data URI string (e.g. data:image/jpeg;base64,...)."""
    if isinstance(image, str) and image.startswith("data:image/"):
        return image

    mime = default_mime
    if isinstance(image, Path) or (isinstance(image, str) and Path(image).is_file()):
        p = Path(image)
        suffix = p.suffix.lower().lstrip(".")
        if suffix in {"png", "webp", "gif"}:
            mime = f"image/{suffix}"
        elif suffix in {"jpg", "jpeg"}:
            mime = "image/jpeg"

    b64 = encode_image_to_base64(image)
    return f"data:{mime};base64,{b64}"


def has_images(messages: Sequence[ChatMessage]) -> bool:
    """Checks whether any message in the sequence contains image payloads or image_url blocks."""
    for msg in messages:
        if msg.get("images"):
            return True
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def normalize_messages_for_ollama(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Normalizes messages for Ollama, which expects string content and an optional 'images' list of base64 strings."""
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        msg_dict = dict(msg)
        content = msg_dict.get("content")
        raw_images = msg_dict.pop("images", None) or []
        images: list[str] = [encode_image_to_base64(img) for img in raw_images]

        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type")
                    if part_type == "text":
                        text_parts.append(part.get("text") or "")
                    elif part_type == "image_url":
                        img_info = part.get("image_url")
                        url = img_info.get("url") if isinstance(img_info, dict) else img_info
                        if url:
                            images.append(encode_image_to_base64(url))
            msg_dict["content"] = "\n".join(text_parts)
        elif content is None:
            msg_dict["content"] = ""

        if images:
            msg_dict["images"] = images

        normalized.append(msg_dict)
    return normalized


def normalize_messages_for_openai(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Normalizes messages for OpenAI-compatible backends (like llama.cpp), which expect content parts with image_url."""
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        msg_dict = dict(msg)
        content = msg_dict.get("content")
        raw_images = msg_dict.pop("images", None) or []

        if raw_images:
            parts: list[dict[str, Any]] = []
            if isinstance(content, str) and content:
                parts.append({"type": "text", "text": content})
            elif isinstance(content, list):
                parts.extend(content)
            for img in raw_images:
                parts.append({"type": "image_url", "image_url": {"url": encode_image_to_data_uri(img)}})
            msg_dict["content"] = parts
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    img_info = part.get("image_url")
                    url = img_info.get("url") if isinstance(img_info, dict) else img_info
                    if url:
                        parts.append({"type": "image_url", "image_url": {"url": encode_image_to_data_uri(url)}})
                else:
                    parts.append(part)
            msg_dict["content"] = parts

        normalized.append(msg_dict)
    return normalized
