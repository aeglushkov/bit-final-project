"""Run SpatialAgent inference on SpatialScore subsets.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_agent.py \
        --model_path ~/models/Qwen2.5-VL-3B-Instruct \
        --model_name qwen2_5vl-3b \
        --dataset_json_path ../../literature/spatialscore/code/dataset/SpatialScore_test50.json \
        --output_dir ../../literature/spatialscore/code/eval_results_test_agent \
        --checkpoints_dir ~/checkpoints \
        --max_steps 5
"""

import os
import sys
import json
import argparse
import traceback
from tqdm import tqdm

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SPATIALSCORE_CODE = os.path.join(PROJECT_ROOT, "literature", "spatialscore", "code")
SPATIAL_AGENT_DIR = os.path.join(SPATIALSCORE_CODE, "SpatialAgent")

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, SPATIALSCORE_CODE)
sys.path.insert(0, SPATIAL_AGENT_DIR)

from actions import ALL_ACTIONS
from action_wrappers import make_action_registry
from model_registry import ModelRegistry
from qwen_client import QwenLocalClient


def build_result_entry(item, pred_answer, is_correct, score):
    """Build a result entry matching test_qwen.py output format."""
    return {
        "id": item.get("id", 0),
        "category": item.get("category", "unknown"),
        "subcategory": item.get("subcategory", "unknown"),
        "input_modality": item.get("input_modality", "image"),
        "question_type": item.get("question_type", ""),
        "source": item.get("source", "unknown"),
        "question": item.get("question", ""),
        "gt_answer": item.get("answer", ""),
        "pred_answer": pred_answer,
        "img_paths": item.get("img_paths", []),
        "is_correct": is_correct,
        "score": score,
    }


def evaluate_answer(pred_answer, item):
    """Evaluate predicted answer against ground truth. Reuses logic from test_qwen.py."""
    from utils.util import extract_option, extract_yes_no, extract_number, extract_numeric_with_unit

    question_type = item.get("question_type", "")
    ground_truth = item.get("answer", "")

    is_correct = False
    score = 0.0

    if question_type.lower() == "multi-choice":
        pred = extract_option(str(pred_answer))
        gt = extract_option(ground_truth)
        is_correct = pred.upper() == gt.upper()
    elif question_type.lower() == "judgment":
        pred = extract_yes_no(str(pred_answer))
        gt = extract_yes_no(ground_truth)
        is_correct = pred.lower() == gt.lower()
    else:  # open-ended
        if any(unit in ground_truth.lower() for unit in [
            "meter", "meters", "m", "cm", "centimeter", "centimeters",
            "km", "kilometer", "kilometers", "inch", "inches", "ft", "foot", "feet",
        ]):
            is_correct = extract_numeric_with_unit(str(pred_answer), ground_truth)["is_correct"]
        else:
            try:
                pred_value = float(extract_number(str(pred_answer)))
            except (ValueError, TypeError):
                pred_value = 0.0
            try:
                gt_value = float(extract_number(ground_truth))
            except (ValueError, TypeError):
                gt_value = 0.0
            is_correct = pred_value == gt_value

    if is_correct:
        score = 1.0

    return is_correct, score


def save_results(all_results, output_dir):
    """Save results in the same structure as test_qwen.py."""
    os.makedirs(output_dir, exist_ok=True)

    # Save all results
    with open(os.path.join(output_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Group by source
    source_results = {}
    category_results = {}
    for r in all_results:
        source = r.get("source", "unknown")
        source_results.setdefault(source, []).append(r)
        category = r.get("category", "unknown")
        category_results.setdefault(category, []).append(r)

    # Save by source
    source_dir = os.path.join(output_dir, "by_source")
    os.makedirs(source_dir, exist_ok=True)
    for source, results in source_results.items():
        with open(os.path.join(source_dir, f"{source}_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        score_sum = sum(r.get("score", 0.0) for r in results)
        total = len(results)
        accuracy = (score_sum / total) * 100 if total > 0 else 0
        with open(os.path.join(source_dir, f"{source}_summary.json"), "w") as f:
            json.dump({"source": source, "accuracy": accuracy, "correct": int(score_sum),
                        "total": total, "score_sum": score_sum}, f, indent=2)
        print(f"Source: {source} - Accuracy: {accuracy:.2f}% ({int(score_sum)}/{total})")

    # Save by category
    category_dir = os.path.join(output_dir, "by_category")
    os.makedirs(category_dir, exist_ok=True)
    for category, results in category_results.items():
        with open(os.path.join(category_dir, f"{category}_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        score_sum = sum(r.get("score", 0.0) for r in results)
        total = len(results)
        accuracy = (score_sum / total) * 100 if total > 0 else 0
        with open(os.path.join(category_dir, f"{category}_summary.json"), "w") as f:
            json.dump({"category": category, "accuracy": accuracy, "correct": int(score_sum),
                        "total": total, "score_sum": score_sum}, f, indent=2)
        print(f"Category: {category} - Accuracy: {accuracy:.2f}% ({int(score_sum)}/{total})")

    # Overall summary
    total_score = sum(r.get("score", 0.0) for r in all_results)
    total_samples = len(all_results)
    overall_acc = (total_score / total_samples) * 100 if total_samples > 0 else 0
    with open(os.path.join(output_dir, "overall_summary.json"), "w") as f:
        json.dump({"accuracy": overall_acc, "correct": int(total_score),
                    "total": total_samples, "score_sum": total_score}, f, indent=2)
    print(f"\nOverall Accuracy: {overall_acc:.2f}% ({int(total_score)}/{total_samples})")


def _patch_user_agent_receive(UserAgent):
    """Monkey-patch UserAgent.receive to stop the conversation on Terminate actions.

    The vendor agent.py sets final_answer when a Terminate action is parsed but
    does NOT return — it continues with execution and feedback.  Autogen's default
    _is_termination_msg checks for the literal string "TERMINATE", which never
    matches the model's JSON output containing {"name": "Terminate", ...}.
    This causes the conversation to keep going and eventually crash with
    ``'str' object has no attribute 'get'`` inside autogen internals.

    The patch intercepts the Terminate action right after parsing, records it,
    and returns immediately so the conversation ends cleanly.
    """
    from utils.prompt import CoTAPrompt

    _original_receive = UserAgent.receive

    def _patched_receive(self, message, sender, request_reply=False, silent=False):
        # Let the parent handle message bookkeeping (appends to _oai_messages, prints)
        self._process_received_message(message, sender, silent)

        parsed_results = self.parser.parse(message)
        parsed_content = parsed_results['content']
        parsed_status = parsed_results['status']

        # --- Handle parsing failure ---
        if not parsed_status:
            msg_for_check = {"content": message} if isinstance(message, str) else message
            if self.sender_hits_max_reply(sender) or self._is_termination_msg(msg_for_check):
                self._consecutive_auto_reply_counter[sender.name] = 0
                return
            self._consecutive_auto_reply_counter[sender.name] += 1
            feedback_msg = self.feedback_generator.get_prompt("parsing", parsed_results)
            self.step_id += 1
            self.send(feedback_msg, sender, request_reply=True)
            return

        # --- Parsing succeeded ---
        from utils.prompt import DirectAnswerPrompt
        if isinstance(self.parser.prompt_generator, DirectAnswerPrompt):
            self.final_answer = parsed_content
            self._consecutive_auto_reply_counter[sender.name] = 0
            return

        if isinstance(self.parser.prompt_generator, CoTAPrompt):
            if len(parsed_content) > 0:
                action_name = parsed_content['name']

                # ---- FIX: return early on Terminate ----
                if action_name == "Terminate":
                    if "answer" in parsed_content.get('arguments', {}):
                        self.final_answer = parsed_content['arguments']['answer']
                    self.called_tools += [parsed_content]
                    self._consecutive_auto_reply_counter[sender.name] = 0
                    return
                # ---- END FIX ----

                self.called_tools += [parsed_content]

            print("Called tools:", self.step_id, self.current_image_id, parsed_content, self.task)

            executed_results = self.executor.execute(self.step_id, self.current_image_id, parsed_content, self.task)
            executed_status = executed_results['status']

            if executed_status and 'image_paths' in executed_results:
                self.new_image_paths += executed_results['image_paths']

            msg_for_check = {"content": message} if isinstance(message, str) else message
            if self.sender_hits_max_reply(sender) or self._is_termination_msg(msg_for_check):
                self._consecutive_auto_reply_counter[sender.name] = 0
                return

            self._consecutive_auto_reply_counter[sender.name] += 1

            feedback_msg = self.feedback_generator.get_prompt("execution", executed_results)
            if executed_status and getattr(executed_results['content'], 'image', None):
                self.current_image_id += 1
            self.step_id += 1
            self.send(feedback_msg, sender, request_reply=True)

    UserAgent.receive = _patched_receive


def run_agent_on_sample(item, user_agent_cls, assistant_agent_cls, prompt_gen, feedback_gen,
                         parser_cls, executor_cls, action_registry, qwen_client,
                         input_folder, result_folder, max_steps, dataset_base_dir):
    """Run the SpatialAgent conversation loop on a single sample."""
    from utils.prompt import CoTAPrompt, FeedbackPrompt
    from utils.parser import Parser
    from utils.executor import Executor
    from agent import UserAgent
    from autogen.agentchat import AssistantAgent

    # Apply the Terminate-early-return patch (idempotent after first call)
    if not getattr(UserAgent.receive, '_patched', False):
        _patch_user_agent_receive(UserAgent)
        UserAgent.receive._patched = True

    # Resolve image paths
    image_paths = []
    for p in item.get("img_paths", []):
        if os.path.isabs(p):
            image_paths.append(p)
        else:
            image_paths.append(os.path.join(dataset_base_dir, p))

    task = {"id": item.get("id", 0), "image_paths": image_paths}

    # Create fresh components for each sample
    prompt_gen = CoTAPrompt(actions=ALL_ACTIONS)
    feedback_gen = FeedbackPrompt()
    parser = Parser(prompt_generator=prompt_gen)
    executor = Executor(
        input_folder=input_folder,
        result_folder=result_folder,
        action_registry=action_registry,
    )

    user_agent = UserAgent(
        name="user",
        prompt_generator=prompt_gen,
        feedback_generator=feedback_gen,
        parser=parser,
        executor=executor,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=max_steps,
        code_execution_config=False,
    )

    # Set image paths on the client so it can embed them in Qwen messages
    qwen_client.image_paths = image_paths

    # Create assistant with the Qwen client
    llm_config = {
        "config_list": [{"model": qwen_client.model_name, "model_client_cls": "QwenLocalClient"}],
        "cache_seed": None,
    }
    assistant = AssistantAgent(
        name="assistant",
        system_message=prompt_gen.get_task_prompt_only(),
        llm_config=llm_config,
        human_input_mode="NEVER",
    )
    assistant.register_model_client(
        QwenLocalClient,
        model=qwen_client.model,
        processor=qwen_client.processor,
    )
    # Propagate image_paths to the newly registered client
    for client in assistant.client._clients:
        if isinstance(client, QwenLocalClient):
            client.image_paths = image_paths

    # Build the question message, prepending image declarations so the model
    # knows which images are available for tool calls (e.g. image-0, image-1).
    question = item.get("question", "")
    if image_paths:
        image_labels = ", ".join(f"image-{i}" for i in range(len(image_paths)))
        question = f"The following input images are provided: {image_labels}.\n\n{question}"

    # Run the agent loop
    try:
        user_agent.initiate_chat(assistant, message=question, task=task)
    except Exception as e:
        print(f"Error on sample {task['id']}: {e}")
        traceback.print_exc()

    return user_agent.final_answer, user_agent.called_tools


def main():
    parser = argparse.ArgumentParser(description="SpatialAgent Inference on SpatialScore")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Qwen model")
    parser.add_argument("--model_name", type=str, default="qwen2_5vl-3b")
    parser.add_argument("--dataset_json_path", type=str, required=True, help="Path to dataset JSON")
    parser.add_argument("--dataset_base_dir", type=str, default=None,
                        help="Base directory for resolving relative image paths (default: parent of dataset JSON)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--checkpoints_dir", type=str, required=True, help="Directory with tool model checkpoints")
    parser.add_argument("--max_steps", type=int, default=5, help="Max agent conversation steps")
    parser.add_argument("--save_interval", type=int, default=10)
    args = parser.parse_args()

    if args.dataset_base_dir is None:
        args.dataset_base_dir = os.path.dirname(os.path.dirname(args.dataset_json_path))

    # 1. Load Qwen model
    import torch
    print("Loading Qwen model...")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, device_map="auto", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path, use_fast=False, trust_remote_code=True,
        min_pixels=256 * 28 * 28, max_pixels=2560 * 28 * 28,
    )

    # 2. Create ModelRegistry for tool models
    print("Initializing tool model registry...")
    model_registry = ModelRegistry(checkpoints_dir=args.checkpoints_dir)

    # 3. Create VLM function for SelfReasoning
    def vlm_fn(image_path, query):
        """Use Qwen for SelfReasoning sub-queries."""
        from qwen_vl_utils import process_vision_info

        device = next(model.parameters()).device
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": query},
            ]}
        ]
        input_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[input_text], images=image_inputs, videos=video_inputs,
                          padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            generated = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
            response = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        torch.cuda.empty_cache()
        return response

    # 4. Build action registry
    action_registry = make_action_registry(model_registry, vlm_fn)

    # 5. Create Qwen client
    qwen_client = QwenLocalClient(
        config={"model": args.model_name},
        model=model,
        processor=processor,
    )

    # 6. Load dataset
    with open(args.dataset_json_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")

    # 7. Create output directory
    output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Check for existing results to resume
    all_results = []
    start_idx = 0
    results_file = os.path.join(output_dir, "all_results.json")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            all_results = json.load(f)
        processed_ids = {r.get("id") for r in all_results}
        if processed_ids:
            start_idx = max(processed_ids) + 1
        print(f"Resuming from index {start_idx}, found {len(all_results)} existing results")

    # 8. Run inference
    input_folder = args.dataset_base_dir
    result_folder = os.path.join(output_dir, "agent_outputs")
    os.makedirs(result_folder, exist_ok=True)

    for i, item in enumerate(tqdm(data[start_idx:], desc="SpatialAgent inference", initial=start_idx, total=len(data))):
        actual_idx = i + start_idx
        if any(r.get("id") == actual_idx for r in all_results):
            continue

        final_answer, called_tools = run_agent_on_sample(
            item=item,
            user_agent_cls=None, assistant_agent_cls=None,
            prompt_gen=None, feedback_gen=None,
            parser_cls=None, executor_cls=None,
            action_registry=action_registry,
            qwen_client=qwen_client,
            input_folder=input_folder,
            result_folder=result_folder,
            max_steps=args.max_steps,
            dataset_base_dir=args.dataset_base_dir,
        )

        # Extract answer
        pred_answer = final_answer if final_answer else ""
        is_correct, score = evaluate_answer(pred_answer, item)

        result_entry = build_result_entry(item, pred_answer, is_correct, score)
        # Also store agent-specific metadata
        result_entry["called_tools"] = [t.get("name", "") if isinstance(t, dict) else str(t) for t in (called_tools or [])]
        result_entry["num_steps"] = len(called_tools or [])

        all_results.append(result_entry)

        # Save periodically
        if (i + 1) % args.save_interval == 0:
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)

        torch.cuda.empty_cache()

    # 9. Save final results
    save_results(all_results, output_dir)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
