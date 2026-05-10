from __future__ import annotations

from pathlib import Path

from eva_eval.agent.context import AgentContext
from eva_eval.agent.tools import make_tools

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "react_vqa.txt"
PROMPT_PATH_EXTENDED = Path(__file__).resolve().parents[1] / "prompts" / "react_vqa_extended.txt"


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
    return_intermediate_steps: bool = False,
    extended_schema: bool = False,
):
    """Assemble a ReAct agent over the paper-spec tools, bound to one video's
    persistent memory.

    Args:
        text_encoder: callable text -> np.ndarray, used by the *_appearance,
                      *_environment and frame_localization tools. Inject the
                      paper's CLIP text encoder here at the call site so this
                      module stays free of torch/clip imports.
        return_intermediate_steps: when True, executor.invoke(...) returns the
                      ReAct trace under the "intermediate_steps" key. Used by
                      OpenEQA harness for the trace inspector.
        extended_schema: when True, the SQL Objects table includes bbox extents
                      and centers, AND three computed-answer tools are added:
                      get_object_dimensions, get_distance, estimate_room_size.
                      Required for VSI-Bench's strict numeric questions.
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
    tools = make_tools(ctx, extended_schema=extended_schema)
    llm = _build_planner_llm(planner_name)

    prompt_path = PROMPT_PATH_EXTENDED if extended_schema else PROMPT_PATH
    classes = _load_classes(Path(classes_file))
    template_text = prompt_path.read_text().replace("{categories_list}", str(classes))
    prompt = PromptTemplate.from_template(template_text)

    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
        return_intermediate_steps=return_intermediate_steps,
    )
    return executor, ctx
