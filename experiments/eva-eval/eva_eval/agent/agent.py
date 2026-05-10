from __future__ import annotations

from pathlib import Path

from eva_eval.agent.context import AgentContext
from eva_eval.agent.tools import make_tools

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "react_vqa.txt"


def _build_planner_llm(planner_name: str | None):
    from langchain_openai import AzureChatOpenAI, ChatOpenAI

    from eva_eval.llm.client import load_config

    cfg = load_config()
    name = planner_name or cfg["default_planner"]
    if name not in cfg["models"]:
        raise KeyError(f"Planner {name!r} not in config")
    m = cfg["models"][name]
    gen = m.get("generation", {}) or {}
    if m["backend"] == "azure_openai":
        return AzureChatOpenAI(
            azure_deployment=m["azure_deployment"],
            api_version=m["api_version"],
            temperature=gen.get("temperature", 0.0),
        )
    return ChatOpenAI(
        base_url=m["base_url"],
        api_key=m.get("api_key") or "EMPTY",
        model=m["model_name"],
        temperature=gen.get("temperature", 0.0),
        max_tokens=gen.get("max_tokens"),
    )


def _build_vlm():
    from eva_eval.llm.client import load_default_vlm

    return load_default_vlm()


def _load_classes(classes_file: Path) -> list[str]:
    return [line.strip() for line in classes_file.read_text().splitlines() if line.strip() and not line.startswith("#")]


def build_agent(
    video_cache_dir: str | Path,
    paper_code_dir: str | Path,
    classes_file: str | Path,
    text_encoder,
    planner_name: str | None = None,
    max_iterations: int = 30,
):
    """Assemble a ReAct agent over the six paper-spec tools, bound to one
    video's persistent memory.

    Args:
        text_encoder: callable text -> np.ndarray, used by the *_appearance,
                      *_environment and frame_localization tools. Inject the
                      paper's CLIP text encoder here at the call site so this
                      module stays free of torch/clip imports.
    """
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.prompts import PromptTemplate

    vlm = _build_vlm()
    ctx = AgentContext.load(
        video_cache_dir=video_cache_dir,
        paper_code_dir=paper_code_dir,
        vlm=vlm,
        text_encoder=text_encoder,
    )
    tools = make_tools(ctx)
    llm = _build_planner_llm(planner_name)

    classes = _load_classes(Path(classes_file))
    template_text = PROMPT_PATH.read_text().replace("{categories_list}", str(classes))
    prompt = PromptTemplate.from_template(template_text)

    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
    )
    return executor, ctx
