import base64
from pathlib import Path

import pytest

from omnillm.core.media import (
    encode_image_to_base64,
    encode_image_to_data_uri,
    has_images,
    normalize_messages_for_ollama,
    normalize_messages_for_openai,
)


def test_encode_image_to_base64(tmp_path: Path):
    raw = b"sample_pixels"
    expected = base64.b64encode(raw).decode("ascii")

    # From raw bytes
    assert encode_image_to_base64(raw) == expected

    # From Path
    file_path = tmp_path / "test.png"
    file_path.write_bytes(raw)
    assert encode_image_to_base64(file_path) == expected

    # From str path
    assert encode_image_to_base64(str(file_path)) == expected

    # From data URI
    data_uri = f"data:image/png;base64,{expected}"
    assert encode_image_to_base64(data_uri) == expected

    # Already base64 string
    assert encode_image_to_base64(expected) == expected

    # Unsupported type
    with pytest.raises(TypeError, match="Unsupported image type"):
        encode_image_to_base64(12345)  # type: ignore[arg-type]


def test_encode_image_to_data_uri(tmp_path: Path):
    raw = b"png_data"
    b64 = base64.b64encode(raw).decode("ascii")

    # Already data uri
    uri = f"data:image/webp;base64,{b64}"
    assert encode_image_to_data_uri(uri) == uri

    # Path detection
    png_file = tmp_path / "img.png"
    png_file.write_bytes(raw)
    assert encode_image_to_data_uri(png_file) == f"data:image/png;base64,{b64}"

    # Default fallback
    assert encode_image_to_data_uri(raw) == f"data:image/jpeg;base64,{b64}"


def test_has_images():
    assert not has_images([{"role": "user", "content": "Just text"}])
    assert has_images([{"role": "user", "content": "Prompt", "images": ["data:image/png;base64,abc"]}])
    assert has_images(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Prompt"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ]
    )


def test_normalize_messages_for_ollama(tmp_path: Path):
    img_file = tmp_path / "test.jpg"
    img_file.write_bytes(b"image_content")
    b64 = base64.b64encode(b"image_content").decode("ascii")

    # OpenAI format converted to Ollama
    openai_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": str(img_file)}},
            ],
        }
    ]
    ollama_msgs = normalize_messages_for_ollama(openai_messages)
    assert ollama_msgs == [{"role": "user", "content": "What is this?", "images": [b64]}]

    # Native Ollama format with images field
    native_messages = [{"role": "user", "content": "Describe", "images": [img_file]}]
    assert normalize_messages_for_ollama(native_messages) == [{"role": "user", "content": "Describe", "images": [b64]}]


def test_normalize_messages_for_openai(tmp_path: Path):
    img_file = tmp_path / "photo.png"
    img_file.write_bytes(b"pixel_data")
    b64 = base64.b64encode(b"pixel_data").decode("ascii")

    # Native Ollama images converted to OpenAI format
    ollama_messages = [{"role": "user", "content": "Describe", "images": [img_file]}]
    openai_msgs = normalize_messages_for_openai(ollama_messages)

    assert openai_msgs == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }
    ]
