import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics import f1_score

# Keys in a local .env file become environment variables (already-set variables win).
load_dotenv()

from src.agent import RESTART, DisasterAuditorLLM
from src.auditor_agent import AuditorTools, ClaudeAuditorAgent
from src.models.registry import ModelRegistry

# One minimal mark for the whole app: browser tab, sidebar title, assistant avatar and greeting.
APP_ICON = ":material/cyclone:"

st.set_page_config(
    page_title="Disaster Auditor",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------
# Chat-style polish that theme config can't express
# (colors and fonts live in .streamlit/config.toml)
# -------------------------------------------------------------
st.html("""
<style>
    /* Assistant turns: no bubble, just text on the page */
    [data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.25rem 0;
    }

    /* User turns: right-aligned soft bubble, no avatar */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
        background: #F0EEE6;
        border-radius: 1.1rem;
        padding: 0.55rem 1rem;
        margin-left: auto;
        width: fit-content;
        max-width: 85%;
    }
    [data-testid="stChatMessageAvatarUser"] { display: none; }

    /* Composer: floating rounded box */
    [data-testid="stChatInput"] {
        border-radius: 1.25rem;
        box-shadow: 0 0.25rem 1.25rem rgba(31, 30, 29, 0.07);
    }

    /* Thin scrollbar when a long draft overflows the box */
    [data-testid="stChatInputTextArea"] { scrollbar-width: thin; }

    /* Toolbar sits inside the composer's bottom edge, left of the send button */
    .st-key-composer { gap: 0; container-type: inline-size; }
    .st-key-composer_tools {
        margin-top: -2.9rem;
        padding: 0 3.4rem 0.45rem 0.6rem;
        position: relative;
        z-index: 2;
        pointer-events: none;
        flex-wrap: nowrap;  /* never let the model picker drop out of the box */
    }
    .st-key-composer_tools > * { pointer-events: auto; }
    .st-key-composer_tools [data-testid="stElementContainer"] { margin: 0; }
    .st-key-composer_tools > :nth-child(2) { min-width: 0; }  /* the spacer gives way first */

    /* "+" reads as a plain icon button: drop the menu chevron */
    .st-key-quick_menu [data-testid="stMenuButtonButton"] [data-testid="stIconMaterial"] { display: none; }

    /* Narrow composer: icon-only toggle and a slimmer model picker */
    @container (max-width: 420px) {
        .st-key-composer_tools [data-testid="stCheckbox"]:has(input[aria-label="Compare"]) [data-testid="stMarkdownContainer"] { display: none; }
        .st-key-model_picker { width: 7.5rem !important; }
    }

    /* Model picker reads as plain text, like a chat app's model name */
    .st-key-model_picker [data-baseweb="select"] > div {
        background: transparent;
        border-color: transparent;
        font-size: 0.875rem;
    }
    .st-key-model_picker [data-baseweb="select"] > div:hover { background: rgba(31, 30, 29, 0.05); }

    /* Slash-command menu (driven by the script in the sidebar) */
    #slash-menu {
        display: none;
        position: fixed;
        z-index: 1000;
        background: #FFFFFF;
        border: 1px solid #E8E6DC;
        border-radius: 0.9rem;
        box-shadow: 0 0.5rem 1.75rem rgba(31, 30, 29, 0.12);
        padding: 0.35rem;
        max-height: 18rem;
        overflow-y: auto;
        font-family: Inter, sans-serif;
    }
    #slash-menu .slash-head {
        font-size: 0.72rem;
        color: #8A8780;
        padding: 0.3rem 0.6rem 0.25rem;
    }
    #slash-menu .slash-item {
        display: flex;
        gap: 0.75rem;
        align-items: baseline;
        padding: 0.45rem 0.6rem;
        border-radius: 0.55rem;
        cursor: pointer;
    }
    #slash-menu .slash-item.active { background: #F0EEE6; }
    #slash-menu .slash-cmd {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #C15F3C;
        white-space: nowrap;
    }
    #slash-menu .slash-desc { font-size: 0.85rem; color: #5E5D59; }

    /* Responsive frame: in Chat view the header, messages and composer share one wide column that
       fills the page; the cap only kicks in on very wide monitors so lines stay readable */
    .st-key-chat_column, .st-key-topbar_chat {
        max-width: 80rem;
        margin-left: auto;
        margin-right: auto;
        width: 100%;
    }
    /* ...but the composer stays a comfortable centered width under the wide messages */
    .st-key-chat_column .st-key-composer_dock,
    .st-key-chat_column .st-key-quick_pills {
        max-width: 48rem;
        width: 100%;
        margin-left: auto;
        margin-right: auto;
    }
    /* Start-screen suggestion chips sit centered under the composer */
    .st-key-quick_pills [role="radiogroup"] { justify-content: center; }
    /* Message list fills the viewport instead of a fixed 600px, leaving room for header and composer */
    /* (Streamlit also pins the layout wrapper with flex: 0 0 600px, so size both) */
    [data-testid="stLayoutWrapper"]:has(> .st-key-chat_scroll) { flex: 0 0 auto !important; }
    [data-testid="stLayoutWrapper"]:has(> .st-key-chat_scroll),
    .st-key-chat_scroll {
        height: calc(100dvh - 19rem) !important;
        min-height: 16rem;
    }
    /* In a conversation the composer dock is pinned to the bottom of the viewport: messages scroll
       behind it and it never gets pushed down. (Sticky goes on the dock's layout wrapper, the
       element that is a direct child of the chat column, or it would have no room to stick in.) */
    [data-testid="stVerticalBlock"]:has(.st-key-chat_scroll) > [data-testid="stLayoutWrapper"]:has(> .st-key-composer_dock) {
        position: sticky;
        bottom: 0;
        z-index: 5;
        background: #FAF9F5;
        padding: 0.5rem 0 1rem;
        box-shadow: 0 -1rem 1rem -0.5rem #FAF9F5;  /* soft fade where messages pass underneath */
    }
    /* First question from the start screen: the message list takes the start screen's slot, and
       Streamlit keeps that screen's old children inside it (greyed out) until the reply finishes.
       Hide those leftovers so the greeting, chips and old composer vanish immediately. */
    .st-key-chat_scroll > [data-stale="true"]:not(:has([data-testid="stChatMessage"])),
    .st-key-chat_scroll [data-testid="stLayoutWrapper"]:has(> .st-key-composer_dock),
    .st-key-chat_scroll .st-key-quick_pills {
        display: none !important;
    }

    /* Top bar (model + Macro-F1 + view switcher) stays pinned just under Streamlit's 3.75rem header
       in every view, with the page color behind it so content scrolls cleanly underneath. */
    [data-testid="stLayoutWrapper"]:has(> .st-key-topbar),
    [data-testid="stLayoutWrapper"]:has(> .st-key-topbar_chat) {
        position: sticky;
        top: 3.75rem;
        z-index: 6;
        background: #FAF9F5;
        padding: 0.25rem 0 0.5rem;
        box-shadow: 0 1rem 1rem -0.5rem #FAF9F5;  /* soft fade where content passes underneath */
    }

    /* Chat view with a conversation: the page itself never scrolls. Every level from the page
       container down to the message list becomes a flex column, so the list takes exactly the
       space left between the top bar and the composer and is the only thing that scrolls. */
    [data-testid="stMain"]:has(.st-key-chat_column .st-key-chat_scroll) { overflow: hidden; }
    [data-testid="stMainBlockContainer"]:has(.st-key-chat_column .st-key-chat_scroll) {
        height: 100%;
        display: flex;
        flex-direction: column;
        box-sizing: border-box;
    }
    [data-testid="stMainBlockContainer"]:has(.st-key-chat_column .st-key-chat_scroll) > [data-testid="stVerticalBlock"],
    [data-testid="stLayoutWrapper"]:has(> .st-key-chat_column),
    .st-key-chat_column {
        flex: 1 1 0 !important;
        min-height: 0;
    }
    .st-key-chat_column [data-testid="stLayoutWrapper"]:has(> .st-key-chat_scroll),
    .st-key-chat_column .st-key-chat_scroll {
        flex: 1 1 0 !important;
        height: auto !important;
        min-height: 0;
    }

    /* Drop Streamlit's default 10rem bottom padding while chatting, so the page itself doesn't scroll */
    [data-testid="stMainBlockContainer"]:has(.st-key-chat_scroll) { padding-bottom: 0; }

    @media (max-width: 640px) {
        /* Phones: tighter page gutters, header stacks with the view switcher full width */
        [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; padding-bottom: 2rem; }
        .st-key-topbar, .st-key-topbar_chat { flex-wrap: wrap; row-gap: 0.5rem; }
        [data-testid="stLayoutWrapper"]:has(> .st-key-chat_scroll),
        .st-key-chat_scroll { height: calc(100dvh - 21rem) !important; }  /* header wraps to two rows here */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { max-width: 92%; }
    }

    /* Start-screen greeting */
    .greeting {
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: clamp(1.55rem, 4.5vw, 2.3rem);
        line-height: 1.25;
        font-weight: 400;
        letter-spacing: -0.01em;
        color: #1F1E1D;
        text-align: center;
        margin: 0;
    }
    /* The greeting mark is the app icon, drawn with Streamlit's Material Symbols font */
    .greeting-mark {
        font-family: 'Material Symbols Rounded';
        font-weight: 400;
        font-size: 0.95em;
        vertical-align: -0.12em;
        color: #C15F3C;
        margin-right: 0.4rem;
    }
</style>
""")


@st.cache_resource
def load_registry():
    return ModelRegistry()


registry = load_registry()


@st.cache_resource
def load_agent_tools(_registry):
    # Precomputes the per-class error breakdown once per server process.
    return AuditorTools(_registry)


agent_tools = load_agent_tools(registry)

VIEWS = ["Chat", "Chat + details", "Benchmarks"]

SLASH_COMMANDS = (
    [
        {"cmd": "/random", "desc": "Audit a random test case"},
        {"cmd": "/hard", "desc": "A case at least one model got wrong"},
        {"cmd": "/compare", "desc": "A disagreement case, all three models"},
    ]
    + [
        {"cmd": f"/category {c}", "desc": f"Audit {'an' if c[0] in 'AEIOU' else 'a'} {c} case"}
        for c in registry.class_names
    ]
    + [{"cmd": "/export", "desc": "Open benchmarks and the session report"}]
)

# Shows a command menu above the composer while the text starts with "/".
# st.chat_input has no keystroke events, so this listens on the page (st.html isn't iframed).
SLASH_MENU_JS = """
<script>
(() => {
  if (window.__slashMenu) return;
  window.__slashMenu = true;
  const COMMANDS = __COMMANDS__;
  const SEL = '[data-testid="stChatInputTextArea"]';
  const menu = document.createElement('div');
  menu.id = 'slash-menu';
  menu.setAttribute('role', 'listbox');
  document.body.appendChild(menu);
  let items = [], active = 0, ta = null;

  const open = () => menu.style.display === 'block';
  const hide = () => { menu.style.display = 'none'; items = []; };

  // React tracks the textarea value itself; use the native setter + input event so it notices.
  const setValue = (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set.call(el, v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  };

  // Build rows with DOM calls: the HTML sanitizer drops a script tag whose text looks like markup.
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text) n.textContent = text;
    return n;
  };
  const render = () => {
    menu.replaceChildren(el('div', 'slash-head', 'Commands'));
    items.forEach((c, i) => {
      const row = el('div', 'slash-item' + (i === active ? ' active' : ''));
      row.setAttribute('role', 'option');
      row.dataset.idx = String(i);
      row.append(el('span', 'slash-cmd', c.cmd), el('span', 'slash-desc', c.desc));
      menu.append(row);
    });
    const box = ta.closest('[data-testid="stChatInput"]') || ta;
    const r = box.getBoundingClientRect();
    menu.style.left = r.left + 'px';
    menu.style.width = r.width + 'px';
    menu.style.bottom = (window.innerHeight - r.top + 8) + 'px';
    menu.style.display = 'block';
    const cur = menu.querySelector('.slash-item.active');
    if (cur) cur.scrollIntoView({ block: 'nearest' });
  };

  const update = (el) => {
    const q = el.value.toLowerCase();
    if (!q.startsWith('/') || q.includes('\\n')) { hide(); return; }
    items = COMMANDS.filter(c => c.cmd.toLowerCase().startsWith(q.trimEnd()) || c.cmd.toLowerCase().startsWith(q));
    if (!items.length || (items.length === 1 && items[0].cmd.toLowerCase() === q.trim())) { hide(); return; }
    ta = el;
    active = Math.min(active, items.length - 1);
    render();
  };

  const pick = (i) => {
    const c = items[i];
    if (!c || !ta) return;
    setValue(ta, c.cmd);
    hide();
    ta.focus();
  };

  document.addEventListener('input', (e) => {
    if (e.target.matches && e.target.matches(SEL)) { active = 0; update(e.target); }
  }, true);

  // Capture phase on document runs before React's handlers, so Enter picks instead of sending.
  document.addEventListener('keydown', (e) => {
    if (!open() || e.target !== ta) return;
    if (e.key === 'ArrowDown') { active = (active + 1) % items.length; render(); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { active = (active - 1 + items.length) % items.length; render(); e.preventDefault(); }
    else if (e.key === 'Enter' || e.key === 'Tab') { e.preventDefault(); e.stopImmediatePropagation(); pick(active); }
    else if (e.key === 'Escape') { e.preventDefault(); hide(); }
  }, true);

  menu.addEventListener('mousedown', (e) => {
    const row = e.target.closest('[data-idx]');
    if (row) { e.preventDefault(); pick(Number(row.dataset.idx)); }
  });
  document.addEventListener('focusout', (e) => { if (e.target === ta) setTimeout(hide, 120); }, true);
  window.addEventListener('resize', hide);
})();
</script>
""".replace("__COMMANDS__", json.dumps(SLASH_COMMANDS))


# -------------------------------------------------------------
# Session state
# -------------------------------------------------------------
def new_stats():
    # y_true / y_pred hold the primary model's labels per audit so the session Macro-F1 is real.
    return {"tested": 0, "correct": 0, "hard_cases": 0, "y_true": [], "y_pred": []}


def config_value(name: str) -> str:
    """A setting from .streamlit/secrets.toml, else the environment (which .env feeds), else ""."""
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets file at all
        value = None
    return str(value or os.getenv(name, "")).strip()


def default_provider() -> str:
    """LLM_PROVIDER if set ("gemini" / "claude"); otherwise whichever provider has a key, Gemini first."""
    choice = config_value("LLM_PROVIDER").lower()
    if choice in ("claude", "anthropic"):
        return "Anthropic Claude"
    if choice == "gemini":
        return "Google Gemini"
    if not config_value("GEMINI_API_KEY") and config_value("ANTHROPIC_API_KEY"):
        return "Anthropic Claude"
    return "Google Gemini"


DEFAULTS = {
    "messages": [],
    "selected_model": "Random Forest",
    "comparison_mode": False,
    "comparison_models": ["Random Forest", "LightGBM", "XGBoost"],
    "layout_mode": "Chat",
    "last_audit": None,
    "hard_threshold": 0.55,
    "api_provider": default_provider(),
    "gemini_api_key": config_value("GEMINI_API_KEY"),
    "claude_api_key": config_value("ANTHROPIC_API_KEY"),
    # Only needed for organization-level keys that aren't scoped to a workspace.
    "claude_workspace_id": config_value("ANTHROPIC_WORKSPACE_ID"),
    "pending_quick_action": None,
    "pending_question": None,
    # Claude's Messages API transcript (tool calls and results included); only ever appended to.
    "agent_history": [],
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v
if "stats" not in st.session_state or "y_true" not in st.session_state.stats:
    st.session_state.stats = new_stats()
if st.session_state.layout_mode not in VIEWS:
    st.session_state.layout_mode = "Chat"


# -------------------------------------------------------------
# Audit controller
# -------------------------------------------------------------
def active_models():
    return st.session_state.comparison_models if st.session_state.comparison_mode else [st.session_state.selected_model]


def record_audit(case, models_to_test):
    """Run the audit, store it as the active case, and update the session scoreboard."""
    audit_result = registry.audit_case(case, models_to_test)
    register_audit(audit_result, models_to_test)
    return audit_result


def register_audit(audit_result, models_to_test):
    """Make an audit the active case and score it (skipped if it's the case already on screen)."""
    case = audit_result["case"]
    previous = st.session_state.last_audit
    st.session_state.last_audit = audit_result
    if previous is not None and previous["case"]["case_id"] == case["case_id"]:
        return

    stats = st.session_state.stats
    primary = audit_result["models"][models_to_test[0]]
    stats["tested"] += 1
    stats["y_true"].append(case["ground_truth_class"])
    stats["y_pred"].append(primary["prediction"]["predicted_class"])
    if primary["is_correct"]:
        stats["correct"] += 1
    # Hard case: misclassified, or correct but below the confidence threshold.
    if not primary["is_correct"] or primary["prediction"]["confidence"] < st.session_state.hard_threshold:
        stats["hard_cases"] += 1


def session_macro_f1():
    stats = st.session_state.stats
    if not stats["y_true"]:
        return None
    return f1_score(stats["y_true"], stats["y_pred"], average="macro", zero_division=0)


def run_audit(case, prompt_label=None, target_models=None):
    if not case:
        st.toast("No matching case found in the benchmark bank.", icon=":material/warning:")
        return

    models_to_test = target_models or active_models()
    audit_result = record_audit(case, models_to_test)

    user_text = prompt_label or f"Audit case `{case['case_id']}` ({case['ground_truth_class']})"
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.messages.append({"role": "assistant", "audit": audit_result, "models_tested": models_to_test})
    st.rerun()


QUICK_ACTIONS = {
    ":material/shuffle: Random case": ({}, "Audit a random historical case from the test bank"),
    ":material/warning: Hard case": ({"only_hard": True}, "Examine a case at least one model misclassified"),
    ":material/compare_arrows: Model disagreement": ({"only_disagreement": True}, "Find a case where the models disagree"),
    ":material/water_drop: Flood": ({"disaster_type": "Flood"}, "Audit a Flood case"),
    ":material/landslide: Earthquake": ({"disaster_type": "Earthquake"}, "Audit an Earthquake case"),
    ":material/storm: Storm": ({"disaster_type": "Storm"}, "Audit a Storm case"),
}


def queue_quick_action():
    # Widget callback: remember the pick and clear the pill so it doesn't re-fire on every rerun.
    st.session_state.pending_quick_action = st.session_state.quick_pills
    st.session_state.quick_pills = None


def run_quick_action(label):
    filters, prompt_label = QUICK_ACTIONS[label]
    if filters.get("only_disagreement"):
        st.session_state.comparison_mode = True
    run_audit(registry.get_random_case(**filters), prompt_label)


def handle_slash_command(command_text: str):
    cmd = command_text.strip().lower()
    if cmd.startswith("/random"):
        run_audit(registry.get_random_case(), "`/random`")
    elif cmd.startswith("/hard"):
        run_audit(registry.get_random_case(only_hard=True), "`/hard`")
    elif cmd.startswith("/compare"):
        st.session_state.comparison_mode = True
        run_audit(registry.get_random_case(only_disagreement=True), "`/compare`")
    elif cmd.startswith("/category"):
        parts = cmd.split(maxsplit=1)
        cat_query = parts[1].strip() if len(parts) > 1 else "Flood"
        matched = next((c for c in registry.class_names if c.lower() == cat_query.lower()), None)
        if matched:
            run_audit(registry.get_random_case(disaster_type=matched), f"`/category {matched}`")
        else:
            st.toast(f"Unknown category '{cat_query}'. Available: {', '.join(registry.class_names)}", icon=":material/info:")
    elif cmd.startswith("/export"):
        st.session_state.layout_mode = "Benchmarks"
        st.rerun()
    else:
        st.toast(f"Unrecognized command '{command_text}'. Try /random, /hard, /compare, /category <name>, /export", icon=":material/help:")


def submit_question(text: str):
    """Show the user's turn right away; the reply streams on the next run."""
    st.session_state.messages.append({"role": "user", "content": text})
    st.session_state.pending_question = text
    st.rerun()


def claude_agent_enabled():
    return st.session_state.api_provider == "Anthropic Claude" and bool(st.session_state.claude_api_key.strip())


def answer_pending_question():
    question = st.session_state.pending_question
    st.session_state.pending_question = None
    if claude_agent_enabled():
        answer_with_agent(question)
    else:
        answer_with_plain_llm(question)


def ui_context():
    parts = []
    audit = st.session_state.last_audit
    if audit:
        c = audit["case"]
        s_ = c["summary"]
        parts.append(
            f"case open on screen: {c['case_id']} (ground truth {c['ground_truth_class']}; "
            f"{s_['country']}, {s_['region']}, month {s_['start_month']})"
        )
    parts.append(f"selected models: {', '.join(active_models())}")
    return "[UI context] " + "; ".join(parts)


def steps_label(steps):
    n = len(steps)
    return f"Used {n} tool{'s' if n != 1 else ''}" if n else "Answered directly"


def render_step(step):
    with st.status(step["label"], type="step", state="error" if step["is_error"] else "complete"):
        st.caption(step["detail"])


def answer_with_agent(question):
    """Claude decides which registry tools to call; steps, case cards and the answer render as they arrive."""
    agent = ClaudeAuditorAgent(
        agent_tools,
        st.session_state.claude_api_key.strip(),
        workspace_id=st.session_state.claude_workspace_id,
    )
    text, steps, audits, error = "", [], [], None

    with st.chat_message("assistant", avatar=APP_ICON):
        steps_box = st.status(":shimmer[Working on it]", type="compact")
        cards_box = st.container()
        answer_ph = st.empty()

        for kind, payload in agent.run_turn(
            st.session_state.agent_history, f"{ui_context()}\n\n{question}", active_models()
        ):
            if kind == "text":
                text += payload
                answer_ph.markdown(text)
            elif kind == "retract":
                # A request attempt failed mid-stream and is being retried; drop its partial text.
                text = text[:-payload]
                answer_ph.markdown(text)
            elif kind == "tool":
                step = {
                    "label": payload["summary"],
                    "is_error": payload["is_error"],
                    "detail": f"`{payload['name']}` {json.dumps(payload['input'], ensure_ascii=False)}",
                }
                steps.append(step)
                with steps_box:
                    render_step(step)
                if payload["audit"]:
                    register_audit(payload["audit"], payload["models"])
                    audits.append({"audit": payload["audit"], "models": payload["models"]})
                    with cards_box:
                        render_audit_card(payload["audit"], payload["models"], f"live_{len(audits)}")
            elif kind == "error":
                error = payload

        if error:
            text += ("\n\n" if text else "") + f":orange[:material/warning:] {error}"
            answer_ph.markdown(text)
        steps_box.update(label=steps_label(steps), state="error" if error and not steps else "complete")

    st.session_state.messages.append({"role": "assistant", "content": text, "steps": steps, "audits": audits})
    st.rerun()


def answer_with_plain_llm(question):
    """Gemini or the local fallback: audit a case if the question asks for one, then stream a reply."""
    lower_q = question.lower()
    test_keywords = ["สุ่ม", "audit", "test", "case", "ลอง", "random", "hard", "ผิด", "misclassified", "เคส"]
    if any(w in lower_q for w in test_keywords):
        cat_match = next((c for c in registry.class_names if c.lower() in lower_q), None)
        only_h = any(w in lower_q for w in ["hard", "ผิด", "wrong", "misclassified"])
        case = registry.get_random_case(disaster_type=cat_match, only_hard=only_h)
        if case:
            models_to_test = active_models()
            audit_result = record_audit(case, models_to_test)
            msg = {"role": "assistant", "audit": audit_result, "models_tested": models_to_test}
            st.session_state.messages.append(msg)
            render_message(msg, len(st.session_state.messages) - 1)

    prov_code = "gemini" if st.session_state.api_provider == "Google Gemini" else "anthropic"
    api_key = st.session_state.gemini_api_key if prov_code == "gemini" else st.session_state.claude_api_key
    llm = DisasterAuditorLLM(provider=prov_code, api_key=api_key)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if m.get("content")
    ]
    with st.chat_message("assistant", avatar=APP_ICON):
        reply, answer_ph = "", st.empty()
        for piece in llm.chat_stream(
            user_message=question,
            history=history,
            current_case=st.session_state.last_audit,
        ):
            # RESTART: the model failed mid-answer and a fallback model is starting over.
            reply = "" if piece is RESTART else reply + piece
            answer_ph.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()


# -------------------------------------------------------------
# Rendering helpers
# -------------------------------------------------------------
def verdict_badge(is_correct):
    if is_correct:
        st.badge("Correct", icon=":material/check:", color="green")
    else:
        st.badge("Misclassified", icon=":material/close:", color="red")


def render_audit_card(audit, tested_models, key_suffix):
    case = audit["case"]
    summary = case["summary"]
    with st.container(border=True):
        st.markdown(f":material/description: **Case {case['case_id']}** · ground truth **{case['ground_truth_class']}**")
        st.caption(
            f"{summary['country']} ({summary['region']}) · month {summary['start_month']} · "
            f"magnitude scale `{summary['magnitude_scale']}`"
        )

        if len(tested_models) == 1:
            m_name = tested_models[0]
            m_data = audit["models"][m_name]
            pred = m_data["prediction"]
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.markdown(f"{m_name} predicted **{pred['predicted_class']}** · {pred['confidence'] * 100:.1f}% confidence")
                verdict_badge(m_data["is_correct"])
            st.progress(pred["confidence"])
            st.markdown("**Top decision signals (SHAP)**")
            for exp in m_data["shap"]["human_explanations"][:3]:
                st.markdown(f"- {exp}")
        else:
            cols = st.columns(len(tested_models))
            for col, m_name in zip(cols, tested_models):
                m_data = audit["models"][m_name]
                with col:
                    st.markdown(f"**{m_name}**")
                    st.markdown(f"{m_data['prediction']['predicted_class']} · {m_data['prediction']['confidence'] * 100:.1f}%")
                    verdict_badge(m_data["is_correct"])

        is_latest = audit is st.session_state.last_audit
        if is_latest and st.session_state.layout_mode == "Chat":
            if st.button("Open details", icon=":material/right_panel_open:", type="tertiary", key=f"details_{key_suffix}"):
                st.session_state.layout_mode = "Chat + details"
                st.rerun()


def render_message(msg, idx):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
        return
    with st.chat_message("assistant", avatar=APP_ICON):
        if msg.get("steps"):
            with st.expander(steps_label(msg["steps"]), type="compact"):
                for step in msg["steps"]:
                    render_step(step)
        if msg.get("audit"):
            render_audit_card(msg["audit"], msg.get("models_tested", [st.session_state.selected_model]), idx)
        for j, item in enumerate(msg.get("audits", [])):
            render_audit_card(item["audit"], item["models"], f"{idx}_{j}")
        if msg.get("content"):
            st.markdown(msg["content"])


def sync_model_picker():
    # Separate widget key + callback: selected_model survives views where the dropdown isn't rendered.
    st.session_state.selected_model = st.session_state.model_picker


def render_composer(placeholder):
    """Claude-style composer: text box with + on the bottom left and the model picker bottom right.

    Everything lives in one dock so that, in a conversation, CSS can pin it to the bottom of the viewport.
    """
    with st.container(key="composer_dock", gap="small"):
        with st.container(key="composer"):
            user_query = st.chat_input(placeholder, submit_mode="disable", height=110)
            with st.container(key="composer_tools", horizontal=True, vertical_alignment="center", gap="small"):
                picked = st.menu_button(
                    ":material/add:",
                    options=list(QUICK_ACTIONS),
                    type="tertiary",
                    key="quick_menu",
                    help="Quick audit",
                )
                st.space("stretch")
                compare = st.toggle("Compare", value=st.session_state.comparison_mode)
                st.selectbox(
                    "Active model",
                    options=registry.available_models,
                    index=registry.available_models.index(st.session_state.selected_model),
                    key="model_picker",
                    on_change=sync_model_picker,
                    label_visibility="collapsed",
                    width=150,
                )

        if st.session_state.comparison_mode:
            selected_comps = st.pills(
                "Models to compare",
                options=registry.available_models,
                default=st.session_state.comparison_models,
                selection_mode="multi",
                label_visibility="collapsed",
            )
            if selected_comps and list(selected_comps) != st.session_state.comparison_models:
                st.session_state.comparison_models = list(selected_comps)

    if compare != st.session_state.comparison_mode:
        st.session_state.comparison_mode = compare
        st.rerun()
    if picked:
        run_quick_action(picked)
    return user_query


def handle_input(user_query):
    if not user_query:
        return
    if user_query.startswith("/"):
        handle_slash_command(user_query)
    else:
        submit_question(user_query)


def greeting_text():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    if hour < 18:
        return "Good afternoon"
    return "Good evening"


def render_chat(composer_placeholder):
    # Streamlit matches elements between runs by position and keeps unmatched old ones on screen
    # (greyed out) until the run ends. So the start screen is ONE container in slot 0, and the
    # conversation's message list takes that same slot: the greeting disappears the moment the
    # first question is asked, instead of lingering while the answer streams.
    if not st.session_state.messages:
        # Start screen: greeting, centered composer, suggestion chips
        with st.container(key="start_screen"):
            st.space("large")
            st.space("large")
            st.html(
                f'<p class="greeting"><span class="greeting-mark">{APP_ICON.split("/")[1].strip(":")}</span>{greeting_text()}. '
                "Which disaster model shall we audit?</p>"
            )
            st.space("medium")
            user_query = render_composer(composer_placeholder)
            st.pills(
                "Suggestions",
                options=list(QUICK_ACTIONS),
                label_visibility="collapsed",
                key="quick_pills",
                on_change=queue_quick_action,
            )
        handle_input(user_query)
        return

    # Conversation: scrolling message area with the composer pinned below it.
    # Height is overridden in CSS (.st-key-chat_scroll) to follow the viewport.
    message_area = st.container(height=600, border=False, key="chat_scroll")
    # Draw the composer before filling the messages so it is already in place while a reply streams.
    user_query = render_composer(composer_placeholder)
    with message_area:
        for i, msg in enumerate(st.session_state.messages):
            render_message(msg, i)
        if st.session_state.pending_question:
            answer_pending_question()
    handle_input(user_query)


def render_case_details():
    if st.session_state.last_audit is None:
        st.info("Run an audit in the chat to inspect its SHAP attributions and probability distribution.", icon=":material/info:")
        return

    audit = st.session_state.last_audit
    case = audit["case"]
    primary_model = st.session_state.selected_model
    m_res = audit["models"].get(primary_model) or list(audit["models"].values())[0]
    pred = m_res["prediction"]

    with st.container(horizontal=True):
        st.metric("Ground truth", case["ground_truth_class"], border=True)
        st.metric(f"{primary_model} prediction", pred["predicted_class"], border=True)
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.caption(f"Confidence `{pred['confidence'] * 100:.2f}%` · case `{case['case_id']}` · {case['summary']['country']}")
        verdict_badge(m_res["is_correct"])

    st.markdown("**8-class probability distribution**")
    probs_df = pd.DataFrame(
        list(pred["probabilities"].items()),
        columns=["Disaster class", "Probability"],
    ).sort_values("Probability", ascending=True)
    st.bar_chart(
        probs_df.set_index("Disaster class"),
        horizontal=True,
        color="#C15F3C",
        height=240,
    )

    st.markdown("**Top SHAP decision drivers**")
    shap_df = pd.DataFrame([
        {
            "Feature": item["feature"],
            "SHAP impact": item["shap_value"],
            "Direction": "Supports (+)" if item["shap_value"] > 0 else "Opposes (-)",
        }
        for item in m_res["shap"]["top_all"]
    ])
    st.dataframe(
        shap_df,
        column_config={"SHAP impact": st.column_config.NumberColumn(format="%.4f")},
        hide_index=True,
    )

    st.markdown("**In plain language**")
    for exp in m_res["shap"]["human_explanations"][:3]:
        st.caption(f"• {exp}")


def render_leaderboard():
    lb_rows = []
    for m_name, metrics in registry.leaderboard.items():
        ci = metrics["macro_f1_ci"]
        lb_rows.append({
            "Model": m_name,
            "Macro-F1": metrics["macro_f1"],
            "95% CI": f"[{ci[0]:.4f}, {ci[1]:.4f}]",
            "Accuracy": metrics["accuracy"] * 100,
            "Weighted-F1": metrics["weighted_f1"],
        })
    st.dataframe(
        pd.DataFrame(lb_rows),
        column_config={
            "Macro-F1": st.column_config.NumberColumn(format="%.4f"),
            "Accuracy": st.column_config.NumberColumn(format="%.2f%%"),
            "Weighted-F1": st.column_config.NumberColumn(format="%.4f"),
        },
        hide_index=True,
    )

    st.markdown("**Per-class F1**")
    class_f1 = {"Disaster class": registry.class_names}
    for m_name, metrics in registry.leaderboard.items():
        class_f1[m_name] = [metrics["per_class_f1"].get(cls, 0.0) for cls in registry.class_names]
    st.bar_chart(pd.DataFrame(class_f1).set_index("Disaster class"), height=260, stack=False)

    st.markdown("**Paired significance (McNemar)**")
    for pair, result in registry.mcnemar.items():
        note = "significant (p < 0.05)" if result["significant"] else "within noise (p ≥ 0.05)"
        st.caption(f"• **{pair}**: p = `{result['p_value']:.4f}`, {note}")


def render_disagreement():
    meta = registry.metadata
    dis_pct = meta["disagreement_cases"] / meta["total_test_cases"] * 100
    st.markdown(
        f"The test bank has **{meta['disagreement_cases']}** cases (**{dis_pct:.1f}%**) where "
        "Random Forest, LightGBM, and XGBoost predict different classes."
    )
    if st.button("Audit a disagreement case", icon=":material/shuffle:"):
        st.session_state.comparison_mode = True
        if st.session_state.layout_mode == "Benchmarks":
            st.session_state.layout_mode = "Chat + details"
        c = registry.get_random_case(only_disagreement=True)
        run_audit(c, f"Evaluate disagreement case `{c['case_id']}`" if c else None)
    st.markdown("**Why disagreements happen**")
    st.caption("• Boundary confusion: Flood and Storm overlap during monsoon months.")
    st.caption("• Minority-class recall: Random Forest's bagging vs. the boosters' stronger regularization on Drought and Wildfire.")


def render_export():
    stats = st.session_state.stats
    f1_val = session_macro_f1()
    report_df = pd.DataFrame([
        {"Metric": "Total tested", "Value": str(stats["tested"])},
        {"Metric": "Total correct", "Value": str(stats["correct"])},
        {"Metric": "Session accuracy", "Value": f"{stats['correct'] / stats['tested'] * 100:.1f}%" if stats["tested"] else "—"},
        {"Metric": "Session Macro-F1", "Value": f"{f1_val:.4f}" if f1_val is not None else "—"},
        {"Metric": "Hard cases", "Value": str(stats["hard_cases"])},
        {"Metric": "Disagreement bank total", "Value": str(registry.metadata["disagreement_cases"])},
    ])
    st.dataframe(report_df, hide_index=True)
    st.download_button(
        "Download session report (CSV)",
        data=report_df.to_csv(index=False).encode("utf-8"),
        file_name="disaster_auditor_session_report.csv",
        mime="text/csv",
        icon=":material/download:",
    )


# -------------------------------------------------------------
# Sidebar: new audit, session, settings
# -------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### {APP_ICON} Disaster Auditor")

    if st.button("New audit", icon=":material/edit_square:", type="primary", width="stretch"):
        st.session_state.messages = []
        st.session_state.stats = new_stats()
        st.session_state.last_audit = None
        st.session_state.pending_question = None
        st.session_state.agent_history = []
        st.rerun()

    st.markdown("**This session**")
    tested = st.session_state.stats["tested"]
    correct = st.session_state.stats["correct"]
    f1_val = session_macro_f1()
    with st.container(horizontal=True):
        st.metric("Macro-F1", f"{f1_val:.3f}" if f1_val is not None else "—",
                  help="Macro-F1 of the active model over the cases audited this session")
        st.metric("Accuracy", f"{correct / tested * 100:.0f}%" if tested else "—")
    with st.container(horizontal=True):
        st.metric("Tested", tested)
        st.metric("Hard cases", st.session_state.stats["hard_cases"],
                  help="Cases the active model misclassified, or got right with confidence below the threshold in settings")

    st.markdown("**Audit a category**")
    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        cat_select = st.selectbox(
            "Disaster type",
            ["All categories"] + registry.class_names,
            label_visibility="collapsed",
        )
        if st.button("Run", icon=":material/play_arrow:"):
            target_cat = None if cat_select == "All categories" else cat_select
            run_audit(registry.get_random_case(disaster_type=target_cat), f"Audit a **{cat_select}** case")

    is_gemini = st.session_state.api_provider == "Google Gemini"
    active_key = st.session_state.gemini_api_key if is_gemini else st.session_state.claude_api_key
    llm_connected = bool(active_key and active_key.strip())
    llm_label = "Gemini (chat)" if is_gemini else "Claude Opus 5.5 (agent with tools)"

    with st.expander("Settings", icon=":material/settings:", expanded=not llm_connected):
        prov_choice = st.selectbox(
            "LLM provider",
            ["Google Gemini", "Anthropic Claude"],
            index=0 if is_gemini else 1,
        )
        if prov_choice != st.session_state.api_provider:
            st.session_state.api_provider = prov_choice
            st.rerun()

        if is_gemini:
            gem_key = st.text_input(
                "Gemini API key",
                type="password",
                value=st.session_state.gemini_api_key,
                placeholder="AIzaSy...",
                help="Get a key at https://aistudio.google.com",
            )
            if gem_key.startswith("sk-ant-"):
                st.session_state.claude_api_key = gem_key
                st.session_state.api_provider = "Anthropic Claude"
                st.rerun()
            elif gem_key != st.session_state.gemini_api_key:
                st.session_state.gemini_api_key = gem_key
                st.rerun()
        else:
            ant_key = st.text_input(
                "Anthropic API key",
                type="password",
                value=st.session_state.claude_api_key,
                placeholder="sk-ant-api03-...",
            )
            if ant_key.startswith("AIzaSy"):
                st.session_state.gemini_api_key = ant_key
                st.session_state.api_provider = "Google Gemini"
                st.rerun()
            elif ant_key != st.session_state.claude_api_key:
                st.session_state.claude_api_key = ant_key
                st.rerun()
            ws_id = st.text_input(
                "Workspace ID (optional)",
                value=st.session_state.claude_workspace_id,
                placeholder="wrkspc_...",
                help="Only needed if your key isn't scoped to a workspace. Copy it from the Claude Console's Workspaces page.",
            )
            if ws_id.strip() != st.session_state.claude_workspace_id:
                st.session_state.claude_workspace_id = ws_id.strip()
                st.rerun()

        st.session_state.hard_threshold = st.slider(
            "Hard-case confidence threshold",
            min_value=0.30,
            max_value=0.90,
            value=st.session_state.hard_threshold,
            step=0.05,
            help="Correct predictions below this confidence also count as hard cases",
        )

    status = ":green[●] connected" if llm_connected else ":orange[●] offline, add an API key"
    st.caption(f"{llm_label}: {status}")
    st.caption(":green[●] Random Forest · LightGBM · XGBoost ready")
    st.caption(
        f"Test bank: {registry.metadata['total_test_cases']:,} records · "
        f"{len(registry.class_names)} classes · {registry.metadata['disagreement_cases']} disagreements"
    )


# -------------------------------------------------------------
# Main area
# -------------------------------------------------------------
# In Chat view the header narrows to the same centered column as the conversation.
topbar_key = "topbar_chat" if st.session_state.layout_mode == "Chat" else "topbar"
with st.container(horizontal=True, vertical_alignment="center", key=topbar_key):
    st.markdown(
        f"**{st.session_state.selected_model}** · test Macro-F1 "
        f"{registry.leaderboard[st.session_state.selected_model]['macro_f1']:.4f}"
    )
    st.space("stretch")
    view_choice = st.segmented_control(
        "View",
        options=VIEWS,
        default=st.session_state.layout_mode,
        label_visibility="collapsed",
    )
    if view_choice and view_choice != st.session_state.layout_mode:
        st.session_state.layout_mode = view_choice
        st.rerun()

# Quick-action chip picked on the previous run (queued by its callback)
if st.session_state.pending_quick_action:
    pick = st.session_state.pending_quick_action
    st.session_state.pending_quick_action = None
    run_quick_action(pick)

COMPOSER = "Ask about a case, or type / for commands"
view = st.session_state.layout_mode

if view == "Chat":
    with st.container(key="chat_column"):
        render_chat(COMPOSER)

elif view == "Chat + details":
    col_chat, col_details = st.columns([0.55, 0.45], gap="large")
    with col_chat:
        render_chat(COMPOSER)
    with col_details:
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown("**Case details**")
                st.space("stretch")
                if st.button("Close", icon=":material/close:", type="tertiary"):
                    st.session_state.layout_mode = "Chat"
                    st.rerun()
            render_case_details()

else:  # Benchmarks
    tab_leader, tab_disagree, tab_case, tab_export = st.tabs([
        ":material/leaderboard: Leaderboard",
        ":material/compare_arrows: Disagreements",
        ":material/search: Last case",
        ":material/download: Export",
    ])
    with tab_leader:
        render_leaderboard()
    with tab_disagree:
        render_disagreement()
    with tab_case:
        render_case_details()
    with tab_export:
        render_export()

# Last on the page: the slash-command menu script (renders nothing visible).
# Kept out of the sidebar because a collapsed sidebar doesn't mount its contents.
st.html(SLASH_MENU_JS, unsafe_allow_javascript=True)
