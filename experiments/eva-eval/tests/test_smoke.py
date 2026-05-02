import pytest

from eva_eval.llm.client import (
    DEFAULT_CONFIG_PATH,
    ChatModel,
    load_default_planner,
    load_default_vlm,
    load_model,
)


def test_default_config_exists():
    assert DEFAULT_CONFIG_PATH.exists(), DEFAULT_CONFIG_PATH


def test_default_planner_is_text_qwen():
    model = load_default_planner()
    assert model.backend == "openai_compatible"
    assert model.model == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert model.multimodal is False


def test_default_vlm_is_paper_internvl2_8b():
    model = load_default_vlm()
    assert model.backend == "openai_compatible"
    assert model.model == "OpenGVLab/InternVL2-8B-AWQ"
    assert model.multimodal is True


def test_load_named_qwen_vl():
    model = load_model("qwen2.5-vl-7b")
    assert model.model == "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    assert model.multimodal is True


def test_load_internvl_2_5():
    model = load_model("internvl2.5-8b")
    assert model.model == "OpenGVLab/InternVL2_5-8B"


def test_load_azure_gpt4o():
    model = load_model("gpt-4o")
    assert model.backend == "azure_openai"
    assert model.model == "gpt-4o-2024-08-06"


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        load_model("does-not-exist")


def test_invalid_backend_rejected():
    with pytest.raises(ValueError):
        ChatModel({"backend": "fictional"})


def test_vqa_rejects_text_only_model():
    model = load_default_planner()
    with pytest.raises(ValueError):
        model.vqa(image=None, question="hi")
