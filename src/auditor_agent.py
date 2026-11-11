"""Claude-powered auditor agent: Claude decides which registry tools to call and answers from their results."""
import json
from collections import Counter
from typing import Any, Dict, Generator, List, Optional, Tuple

MODEL = "claude-opus-5-5"
MAX_TOOL_ROUNDS = 6
MAX_JSON_RETRIES = 2

SYSTEM_PROMPT = """You are the Disaster Auditor, an expert in natural-disaster classification, ML model evaluation, and explainable AI (SHAP).

You audit an 8-class disaster-type classifier (Drought, Earthquake, Epidemic, Extreme Temperature, Flood, Landslide, Storm, Wildfire) trained on EM-DAT records. Three leak-free SMOTETomek pipelines are compared: Random Forest, LightGBM and XGBoost, evaluated on a stratified 20% held-out test bank. Classes are imbalanced (Flood and Storm dominate), so Macro-F1 is the headline metric, not accuracy.

How to work:
- Use the tools for every fact about the models, metrics, cases or SHAP values. Never state a metric, count, confidence or SHAP value you did not get from a tool in this conversation.
- When the user wants to see, test or find a case, call find_case (or get_case for a specific id). The app shows each case you load as a card, so summarize what matters instead of repeating every field.
- If the question is about particular models, pass them in `models`; otherwise leave it out to use the models selected in the UI.
- To explain a prediction, call explain_prediction. To explain why a model did NOT pick a class, call it with target_class set to that class.
- For "which model is better" questions, combine get_model_metrics with get_significance; differences inside the McNemar noise band are not real wins.
- Ground explanations in disaster physics and seasonality (monsoon floods vs. storms, magnitude scales that only exist for some hazards, event duration patterns).
- Each user message may start with a [UI context] line describing the case open on screen and the selected models. Treat it as background, not as the question.
- Reply in the user's language: Thai (polite, ครับ/ค่ะ) if they write Thai, otherwise English. Be concise and use short markdown.
- Stay on disaster classification, these models and their evaluation. Politely decline unrelated requests."""


def _enum_or_none(value, allowed):
    return value is None or value in allowed


def _close_dangling_tool_calls(history: List[Dict[str, Any]], reason: str) -> None:
    """If the transcript ends on an assistant turn with unanswered tool calls, append error results for them.

    Appends rather than edits, so the history stays append-only (required to keep thinking blocks valid).
    """
    if not history or history[-1]["role"] != "assistant":
        return
    content = history[-1]["content"]
    ids = [b.id for b in content if getattr(b, "type", None) == "tool_use"] if isinstance(content, list) else []
    if ids:
        history.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": i, "is_error": True, "content": reason} for i in ids
        ]})


class AuditorTools:
    """Tool definitions and executors backed by the ModelRegistry."""

    def __init__(self, registry):
        self.registry = registry
        self.classes = list(registry.class_names)
        self.models = list(registry.available_models)
        self.regions = sorted({c["summary"]["region"] for c in registry.cases if c["summary"]["region"] != "Unknown"})
        self._error_breakdown = self._build_error_breakdown()

    # ---------------------------------------------------------------- schemas
    def definitions(self) -> List[Dict[str, Any]]:
        models_param = {
            "type": "array",
            "items": {"type": "string", "enum": self.models},
            "description": "Models to evaluate. Omit to use the models currently selected in the UI.",
        }
        tools = [
            {
                "name": "find_case",
                "description": (
                    "Pick a random held-out test case matching the filters, run the chosen models on it, and show it "
                    "to the user as a case card. Returns ground truth, location, timing, each model's prediction, "
                    "confidence, correctness and top SHAP signals. Use whenever the user wants to see or test a case."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "disaster_type": {"type": "string", "enum": self.classes, "description": "Ground-truth class to sample from."},
                        "region": {"type": "string", "enum": self.regions},
                        "only_hard": {"type": "boolean", "description": "Only cases at least one model misclassified."},
                        "only_disagreement": {"type": "boolean", "description": "Only cases where the three models disagree."},
                        "misclassified_by": {"type": "string", "enum": self.models, "description": "Only cases this model got wrong."},
                        "models": models_param,
                    },
                },
            },
            {
                "name": "get_case",
                "description": "Load a specific test case by id (e.g. CASE-0172), run the chosen models on it, and show it as a case card.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "case_id": {"type": "string", "description": "Case id such as CASE-0172."},
                        "models": models_param,
                    },
                    "required": ["case_id"],
                },
            },
            {
                "name": "explain_prediction",
                "description": (
                    "SHAP attributions for one model on one case: the features pushing toward and away from a class. "
                    "By default explains the model's predicted class; set target_class to ask why it did or didn't pick another class."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "case_id": {"type": "string"},
                        "model": {"type": "string", "enum": self.models},
                        "target_class": {"type": "string", "enum": self.classes},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["case_id", "model"],
                },
            },
            {
                "name": "get_model_metrics",
                "description": "Test-set metrics: Macro-F1 with bootstrap 95% CI, accuracy, weighted-F1, and per-class F1 and recall. Omit model for all three.",
                "input_schema": {
                    "type": "object",
                    "properties": {"model": {"type": "string", "enum": self.models}},
                },
            },
            {
                "name": "get_significance",
                "description": "Paired McNemar tests between each pair of models on the test set (p-values and whether the difference is significant).",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "get_error_breakdown",
                "description": (
                    "Per-class error analysis over the whole test bank: case counts, each model's errors per class, its most common "
                    "confusions, and how many cases the models disagree on. Use for 'where does X fail' or 'which classes get confused' questions."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "disaster_type": {"type": "string", "enum": self.classes, "description": "Limit to one ground-truth class."},
                    },
                },
            },
        ]
        # Streamed requests with client tools: let inputs stream as generated (we validate before running).
        for t in tools:
            t["eager_input_streaming"] = True
        return tools

    # ------------------------------------------------------------- validation
    def validate(self, name: str, args: Any) -> Optional[str]:
        """Return an error message if the (possibly partially parsed) input is invalid, else None."""
        if not isinstance(args, dict):
            return "input must be an object"

        def bool_ok(k):
            return k not in args or isinstance(args[k], bool)

        def models_ok():
            m = args.get("models")
            return m is None or (isinstance(m, list) and m and all(x in self.models for x in m))

        if name == "find_case":
            ok = (_enum_or_none(args.get("disaster_type"), self.classes)
                  and _enum_or_none(args.get("region"), self.regions)
                  and _enum_or_none(args.get("misclassified_by"), self.models)
                  and bool_ok("only_hard") and bool_ok("only_disagreement") and models_ok())
        elif name == "get_case":
            ok = isinstance(args.get("case_id"), str) and models_ok()
        elif name == "explain_prediction":
            top_k = args.get("top_k", 5)
            ok = (isinstance(args.get("case_id"), str) and args.get("model") in self.models
                  and _enum_or_none(args.get("target_class"), self.classes)
                  and isinstance(top_k, int) and not isinstance(top_k, bool) and 1 <= top_k <= 10)
        elif name == "get_model_metrics":
            ok = _enum_or_none(args.get("model"), self.models)
        elif name == "get_significance":
            ok = True
        elif name == "get_error_breakdown":
            ok = _enum_or_none(args.get("disaster_type"), self.classes)
        else:
            return f"unknown tool '{name}'"
        return None if ok else "input does not match the tool schema"

    # -------------------------------------------------------------- execution
    def execute(self, name: str, args: Dict[str, Any], default_models: List[str]) -> Tuple[Dict[str, Any], str, Optional[Dict], List[str]]:
        """Run a tool. Returns (result for Claude, one-line UI summary, audit to show as a card or None, models audited)."""
        if name == "find_case":
            models = list(args.get("models") or default_models)
            # A case picked because a model got it wrong must show that model's prediction.
            if args.get("misclassified_by") and args["misclassified_by"] not in models:
                models.append(args["misclassified_by"])
            case = self.registry.get_random_case(
                disaster_type=args.get("disaster_type"),
                only_hard=args.get("only_hard", False),
                only_disagreement=args.get("only_disagreement", False),
                region=args.get("region"),
                misclassified_by=args.get("misclassified_by"),
            )
            filters = ", ".join(
                [args[k] for k in ("disaster_type", "region") if args.get(k)]
                + (["hard"] if args.get("only_hard") else [])
                + (["disagreement"] if args.get("only_disagreement") else [])
                + ([f"missed by {args['misclassified_by']}"] if args.get("misclassified_by") else [])
            ) or "any case"
            if not case:
                return {"error": "No test case matches these filters."}, f"No case found ({filters})", None, models
            audit = self.registry.audit_case(case, models)
            return self._case_payload(audit), f"Found {case['case_id']} ({filters})", audit, models

        if name == "get_case":
            models = args.get("models") or default_models
            case = self.registry.get_case(args["case_id"].strip().upper())
            if not case:
                return {"error": f"Case '{args['case_id']}' not found."}, f"Case {args['case_id']} not found", None, models
            audit = self.registry.audit_case(case, models)
            return self._case_payload(audit), f"Loaded {case['case_id']}", audit, models

        if name == "explain_prediction":
            case = self.registry.get_case(args["case_id"].strip().upper())
            if not case:
                return {"error": f"Case '{args['case_id']}' not found."}, f"Case {args['case_id']} not found", None, []
            model = self.registry.get_model(args["model"])
            target = args.get("target_class")
            target_idx = self.classes.index(target) if target else case["evaluations"][args["model"]]["predicted_id"]
            shap = model.explain(case["features"], target_class_idx=target_idx, top_k=args.get("top_k", 5))
            slim = lambda rows: [{"feature": r["feature"], "value": r["value"], "shap": r["shap_value"]} for r in rows]
            result = {
                "case_id": case["case_id"],
                "model": args["model"],
                "explained_class": shap["target_class"],
                "model_predicted": case["evaluations"][args["model"]]["predicted_class"],
                "ground_truth": case["ground_truth_class"],
                "pushes_toward": slim(shap["top_positive"]),
                "pushes_away": slim(shap["top_negative"]),
            }
            return result, f"SHAP for {args['model']} on {case['case_id']} → {shap['target_class']}", None, []

        if name == "get_model_metrics":
            lb = self.registry.leaderboard
            data = {args["model"]: lb[args["model"]]} if args.get("model") else lb
            return data, f"Read metrics ({args.get('model') or 'all models'})", None, []

        if name == "get_significance":
            return self.registry.mcnemar, "Read McNemar significance tests", None, []

        if name == "get_error_breakdown":
            cls = args.get("disaster_type")
            data = {cls: self._error_breakdown["per_class"][cls]} if cls else self._error_breakdown["per_class"]
            result = {"per_class": data, "disagreement_cases": self._error_breakdown["disagreement_cases"],
                      "total_cases": self._error_breakdown["total_cases"]}
            return result, f"Read error breakdown ({cls or 'all classes'})", None, []

        return {"error": f"unknown tool '{name}'"}, f"Unknown tool {name}", None, []

    # ---------------------------------------------------------------- helpers
    def _case_payload(self, audit: Dict[str, Any]) -> Dict[str, Any]:
        case = audit["case"]
        s = case["summary"]
        models = {}
        for m_name, m in audit["models"].items():
            p = m["prediction"]
            models[m_name] = {
                "predicted": p["predicted_class"],
                "confidence": p["confidence"],
                "correct": m["is_correct"],
                "top_signals": [
                    {"feature": r["feature"], "value": r["value"], "shap": r["shap_value"]}
                    for r in m["shap"]["top_all"][:3]
                ],
            }
        return {
            "case_id": case["case_id"],
            "ground_truth": case["ground_truth_class"],
            "country": s["country"],
            "region": s["region"],
            "start_month": s["start_month"],
            "magnitude_scale": s["magnitude_scale"],
            "magnitude_missing": s["magnitude_missing"],
            "event_duration_log_norm": s["duration_norm"],
            "models": models,
            "models_disagree_in_full_bank": case.get("is_disagreement", False),
        }

    def _build_error_breakdown(self) -> Dict[str, Any]:
        per_class = {}
        for cls in self.classes:
            cases = [c for c in self.registry.cases if c["ground_truth_class"] == cls]
            entry = {"test_cases": len(cases), "models": {}}
            for m in self.models:
                wrong = [c["evaluations"][m]["predicted_class"] for c in cases if not c["evaluations"][m]["is_correct"]]
                entry["models"][m] = {
                    "errors": len(wrong),
                    "recall": round(1 - len(wrong) / len(cases), 4) if cases else None,
                    "top_confusions": [{"predicted_as": k, "count": v} for k, v in Counter(wrong).most_common(3)],
                }
            entry["disagreement_cases"] = sum(1 for c in cases if c.get("is_disagreement"))
            per_class[cls] = entry
        return {
            "per_class": per_class,
            "disagreement_cases": sum(1 for c in self.registry.cases if c.get("is_disagreement")),
            "total_cases": len(self.registry.cases),
        }


class ClaudeAuditorAgent:
    """Streaming tool-use loop. `history` is the Messages API transcript and is only ever appended to."""

    def __init__(self, tools: AuditorTools, api_key: str, workspace_id: Optional[str] = None):
        self.tools = tools
        self.api_key = api_key
        self.workspace_id = (workspace_id or "").strip() or None

    def run_turn(self, history: List[Dict[str, Any]], user_text: str, default_models: List[str]) -> Generator[Tuple[str, Any], None, None]:
        """Yields ("text", str) for answer tokens, ("retract", n) to drop the last n streamed characters
        when a failed attempt is retried, ("tool", dict) per tool call, and ("error", str) on failure."""
        import anthropic

        # Organization-level keys (not scoped to a workspace) must name the workspace on every request.
        headers = {"anthropic-workspace-id": self.workspace_id} if self.workspace_id else None
        client = anthropic.Anthropic(api_key=self.api_key, default_headers=headers)
        tool_defs = self.tools.definitions()
        _close_dangling_tool_calls(history, "Not run: the previous turn was interrupted.")
        history.append({"role": "user", "content": user_text})

        json_retries = 0
        rounds = 0
        while True:
            streamed = 0
            try:
                with client.beta.messages.stream(
                    model=MODEL,
                    max_tokens=16000,
                    system=SYSTEM_PROMPT,
                    tools=tool_defs,
                    messages=history,
                    # Agentic tool use: medium (this model's default) balances tool judgment and latency.
                    output_config={"effort": "medium"},
                    # System prompt and tools are identical every call; cache them across the loop.
                    cache_control={"type": "ephemeral"},
                    # On a safety-classifier decline, retry server-side on a fallback model.
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                ) as stream:
                    for event in stream:
                        if event.type == "text":
                            streamed += len(event.text)
                            yield "text", event.text
                    response = stream.get_final_message()
                json_retries = 0
            except ValueError:
                # Tool input JSON the SDK couldn't parse at all; no tool_use id to answer, so re-issue the turn.
                # Take back the text this failed attempt already streamed so the retry doesn't duplicate it.
                if streamed:
                    yield "retract", streamed
                json_retries += 1
                if json_retries > MAX_JSON_RETRIES:
                    yield "error", "Claude produced an unreadable tool call several times. Please rephrase and try again."
                    return
                continue
            except anthropic.AuthenticationError:
                yield "error", "Invalid Anthropic API key. Please check the key in the sidebar settings."
                return
            except anthropic.RateLimitError:
                yield "error", "Rate limited by the Anthropic API. Please wait a moment and try again."
                return
            except anthropic.APIStatusError as e:
                if "anthropic-workspace-id" in str(e.message):
                    yield "error", (
                        "This API key belongs to your organization but not to a specific workspace. "
                        "Enter the **Workspace ID** under Settings in the sidebar, or create a key inside a workspace in the Claude Console."
                    )
                else:
                    yield "error", f"Claude API error ({e.status_code}): {e.message}"
                return
            except anthropic.APIConnectionError:
                yield "error", "Could not reach the Anthropic API. Check your connection."
                return

            # Append the assistant turn unchanged (thinking blocks included) to keep history append-only.
            history.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "pause_turn":
                continue
            if response.stop_reason == "refusal":
                # A refusal can cut a tool call off mid-input: never run it, but answer it so history stays valid.
                _close_dangling_tool_calls(history, "Not run: the response was declined.")
                yield "error", "Claude declined to answer this request."
                return

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                if response.stop_reason == "max_tokens":
                    yield "text", "\n\n_(Response truncated.)_"
                return

            if response.stop_reason == "max_tokens":
                # A truncated tool input parses as a valid partial object: don't run it.
                history.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                     "content": "Tool input was truncated; nothing was run."} for b in tool_uses
                ]})
                yield "error", "Claude's response was cut off mid tool call. Please try a narrower question."
                return

            rounds += 1
            if rounds > MAX_TOOL_ROUNDS:
                # Claude ignored the "answer now" note; close out the pending calls and stop.
                history.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                     "content": "Tool budget exhausted; nothing was run."} for b in tool_uses
                ]})
                yield "error", "Stopped after too many tool calls for one question. Try asking something narrower."
                return

            # Run every tool before yielding anything: Streamlit can stop this generator at any yield
            # (a rerun), and an assistant tool_use left without its tool_result would 400 every later turn.
            results, events = [], []
            for block in tool_uses:
                problem = self.tools.validate(block.name, block.input)
                if problem:
                    results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                    "content": json.dumps({"INVALID_JSON": json.dumps(block.input, default=str), "reason": problem})})
                    events.append({"name": block.name, "input": block.input, "summary": f"Rejected invalid input ({problem})",
                                   "is_error": True, "audit": None, "models": []})
                    continue
                try:
                    result, summary, audit, models = self.tools.execute(block.name, block.input, default_models)
                    is_error = "error" in result
                except Exception as e:  # keep the loop alive; let Claude see what failed
                    result, summary, audit, models, is_error = {"error": str(e)}, f"{block.name} failed: {e}", None, [], True
                results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": is_error,
                                "content": json.dumps(result, ensure_ascii=False, default=str)})
                events.append({"name": block.name, "input": block.input, "summary": summary,
                               "is_error": is_error, "audit": audit, "models": models})
            if rounds == MAX_TOOL_ROUNDS:
                results.append({"type": "text", "text": "Tool budget for this question is used up. Answer now with what you have."})
            # All results for this turn go back in one user message.
            history.append({"role": "user", "content": results})
            for ev in events:
                yield "tool", ev
