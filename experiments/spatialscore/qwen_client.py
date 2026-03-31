"""Custom AutoGen ModelClient for local Qwen2.5-VL inference.

Implements the ModelClient protocol from autogen.oai.client so that
the SpatialAgent conversation loop can use a locally-loaded Qwen model
instead of an API call.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class _Message:
    content: Optional[str]


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: List[_Choice]
    model: str
    usage: Dict[str, Any] = field(default_factory=lambda: {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0,
    })


class QwenLocalClient:
    """AutoGen ModelClient protocol implementation for local Qwen2.5-VL.

    Usage:
        client = QwenLocalClient(config, model=model, processor=processor)
        # Set image paths before each agent turn:
        client.image_paths = ["/path/to/img.jpg"]
        assistant.register_model_client(QwenLocalClient, model=model, processor=processor)
    """

    def __init__(self, config: Dict[str, Any], model=None, processor=None):
        self.model = model
        self.processor = processor
        self.model_name = config.get("model", "qwen2_5vl-3b")
        self.image_paths: List[str] = []
        self.max_new_tokens = config.get("max_new_tokens", 1024)

    def create(self, params: Dict[str, Any]) -> _Response:
        """Generate a response from the local Qwen model.

        Args:
            params: Dict with "messages" key containing the conversation history.
                    Messages follow OpenAI format: [{"role": "...", "content": "..."}]
        """
        messages = params.get("messages", [])
        qwen_messages = self._convert_messages(messages)

        if self.model is None:
            # Dry-run mode for testing
            return _Response(
                choices=[_Choice(message=_Message(content=""))],
                model=self.model_name,
            )

        import torch
        device = next(self.model.parameters()).device

        with torch.no_grad():
            input_text = self.processor.apply_chat_template(
                qwen_messages, tokenize=False, add_generation_prompt=True
            )

            # Process vision info if qwen_vl_utils is available
            image_inputs = None
            video_inputs = None
            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(qwen_messages)
            except ImportError:
                pass

            inputs = self.processor(
                text=[input_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            output_ids = self.model.generate(
                **inputs,
                num_beams=1,
                temperature=0.0,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                do_sample=False,
            )

            generated_ids = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            response_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return _Response(
            choices=[_Choice(message=_Message(content=response_text))],
            model=self.model_name,
        )

    def _convert_messages(self, messages: List[Dict]) -> List[Dict]:
        """Convert AutoGen message format to Qwen's multimodal format.

        AutoGen sends: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        Qwen expects: [{"role": "system", "content": [{"type": "text", "text": "..."}]},
                        {"role": "user", "content": [{"type": "image", "image": "..."}, {"type": "text", "text": "..."}]}]
        """
        qwen_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content is None:
                content = ""

            if role == "system":
                qwen_messages.append({"role": "system", "content": content})
                continue

            # Check if content references images (image-0, image-1, etc.)
            # and if we have image_paths set on the client
            image_refs = set(re.findall(r'image-(\d+)', content))

            if image_refs and self.image_paths:
                content_parts = []
                for ref_id in sorted(image_refs, key=int):
                    idx = int(ref_id)
                    if idx < len(self.image_paths):
                        content_parts.append({
                            "type": "image",
                            "image": self.image_paths[idx],
                        })
                content_parts.append({"type": "text", "text": content})
                qwen_messages.append({"role": role, "content": content_parts})
            else:
                # Also check for <img path> patterns from FeedbackPrompt observations
                img_pattern = r'<img\s+([^>]+)>'
                img_matches = re.findall(img_pattern, content)

                if img_matches:
                    content_parts = []
                    for img_path in img_matches:
                        content_parts.append({"type": "image", "image": img_path.strip()})
                    # Remove <img ...> tags from text
                    clean_text = re.sub(img_pattern, '', content).strip()
                    content_parts.append({"type": "text", "text": clean_text})
                    qwen_messages.append({"role": role, "content": content_parts})
                else:
                    qwen_messages.append({"role": role, "content": content})

        return qwen_messages

    def message_retrieval(self, response: _Response) -> List[str]:
        return [choice.message.content for choice in response.choices]

    def cost(self, response: _Response) -> float:
        return 0.0

    @staticmethod
    def get_usage(response: _Response) -> Dict:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0,
            "model": getattr(response, "model", "qwen2_5vl-3b"),
        }
