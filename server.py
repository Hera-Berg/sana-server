"""
sana_server/server.py
----------------------
Unified Sana server. Lives at /Desktop/sana/sana_server/server.py.
Can load/unload any model from the sibling model folders.

Model discovery: scans ../ for folders matching sana_* that contain
  checkpoints/sft_best.pt or checkpoints/pretrain_best.pt

API:
  GET  /api/models          -> list of available models + current active
  POST /api/load            -> {"model_id": "sana_23M_v2"} — load a model
  POST /api/chat            -> {"message": "...", "history": [...]} — generate
  POST /api/reset           -> clear conversation state
  GET  /api/status          -> current model info + loading state
  GET  /                    -> serve index.html
  GET  /static/*            -> serve static files

Run:
    cd /Desktop/sana/sana_server
    python server.py
    # or: python server.py --port 8100
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch

# ── Path setup ────────────────────────────────────────────────────────────────
# Server lives at /Desktop/sana/sana_server/server.py
# Models live at /Desktop/sana/sana_*/
SERVER_DIR = Path(__file__).parent.resolve()
SANA_ROOT = SERVER_DIR.parent  # /Desktop/sana/
STATIC_DIR = SERVER_DIR / "static"

# ── Global model state ────────────────────────────────────────────────────────
state = {
    "model": None,  # loaded Sana instance
    "tokenizer": None,  # loaded Tokenizer instance
    "cfg": None,  # ModelConfig from checkpoint
    "model_id": None,  # e.g. "sana_23M_v2"
    "model_path": None,  # absolute path to model dir
    "loading": False,  # True while weights are being loaded
    "error": None,  # last error message or None
    "device": None,
}
state_lock = threading.Lock()


# ── Model discovery ───────────────────────────────────────────────────────────

# Portfolio mode: only these model IDs are exposed by the server.
# Set to None to show every discoverable sana_* folder again.
PORTFOLIO_MODELS = {"sana_100M", "sana_24M"}

MODEL_META = {
    "sana_24M": {
        "label": "Sana 24M",
        "params": "24M",
        "desc": "Hand-implemented ~24M transformer (RMSNorm + RoPE + SwiGLU + KV "
                "cache, pure PyTorch). Pretrained on FineWeb-Edu, SFT for persona, "
                "emotion tokens, and dry short answers.",
        "tags": [
            "jellyfish biology",
            "general science",
            "emotion tokens",
            "from scratch",
        ],
    },
    "sana_100M": {
        "label": "Sana 100M",
        "params": "100M",
        "desc": "Larger sibling of the 24M model, same hand-built architecture. "
                "Broader pretrain knowledge and more coherent multi-turn answers.",
        "tags": [
            "jellyfish biology",
            "general science",
            "world knowledge",
            "emotion tokens",
            "from scratch",
        ],
    },
}


def discover_models():
    """Scan SANA_ROOT for sana_* folders with a valid checkpoint."""
    models = []
    for folder in sorted(SANA_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        if not folder.name.startswith("sana_"):
            continue
        if PORTFOLIO_MODELS is not None and folder.name not in PORTFOLIO_MODELS:
            continue  # portfolio mode: hide everything not in the allowlist
        ckpt = _find_checkpoint(folder)
        if ckpt is None:
            continue
        meta = MODEL_META.get(folder.name, {})
        models.append(
            {
                "id": folder.name,
                "label": meta.get("label", folder.name),
                "params": meta.get("params", "?M"),
                "desc": meta.get("desc", ""),
                "tags": meta.get("tags", []),
                "path": str(folder),
                "ckpt": str(ckpt),
            }
        )
    return models


def _find_checkpoint(model_dir: Path):
    # v1/v2 checkpoint names
    for name in (
        "checkpoints/sft_best.pt",
        "checkpoints/pretrain_best.pt",
        "checkpoints/sft_final.pt",
        "checkpoints/pretrain_final.pt",
    ):
        p = model_dir / name
        if p.exists():
            return p

    # finetune checkpoint structure: checkpoints/finetune/*.pt
    # Real folders contain ckpt_epoch_NN.pt (plus config.json + tokenizer.json).
    # Prefer an explicit *_best.pt; otherwise pick the highest epoch number.
    import re as _re
    ft_dir = model_dir / "checkpoints" / "finetune"
    if ft_dir.exists():
        for best_name in ("sana_best.pt", "ckpt_best.pt", "best.pt"):
            best = ft_dir / best_name
            if best.exists():
                return best

        epochs = list(ft_dir.glob("ckpt_epoch*.pt")) + list(
            ft_dir.glob("sana_epoch*.pt")
        )
        if epochs:
            def _epoch_num(p):
                m = _re.search(r"epoch[_]?(\d+)", p.name)
                return int(m.group(1)) if m else 0
            return sorted(epochs, key=_epoch_num)[-1]


    return None


# ── Model loading ─────────────────────────────────────────────────────────────


def load_model_async(model_id: str, model_path: str):
    """Load a model in a background thread. Updates state as it goes."""

    def _load():
        with state_lock:
            state["loading"] = True
            state["error"] = None

        try:
            mdir = Path(model_path)
            ckpt_path = _find_checkpoint(mdir)
            if ckpt_path is None:
                raise FileNotFoundError(f"No checkpoint in {model_path}")

            # Add model dir to sys.path so its configs/model/tokenizer are importable
            mdir_str = str(mdir)
            if mdir_str not in sys.path:
                sys.path.insert(0, mdir_str)

            # Import from the specific model folder — each has its own versions
            import importlib
            import importlib.util

            def load_from(rel_path, module_name):
                spec = importlib.util.spec_from_file_location(
                    module_name, mdir / rel_path
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Unload old model first to free VRAM
            with state_lock:
                if state["model"] is not None:
                    del state["model"]
                    state["model"] = None
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            print(f"[server] Loading {ckpt_path} onto {device} ...")
            t0 = time.time()

            if _is_finetune_layout(mdir):
                # ── new layout: custom Tokenizer + model/model.py ──────────
                # Folder structure (see README + screenshots):
                #   model/model.py                      -> SanaConfig, SanaModel
                #   tokenizer/tokenizer.py              -> Tokenizer (pure python)
                #   checkpoints/finetune/config.json    -> SanaConfig.load()
                #   checkpoints/finetune/tokenizer.json -> Tokenizer(path)
                #   checkpoints/finetune/ckpt_epoch_NN.pt
                sana_mod = load_from("model/model.py", f"{model_id}.model")
                tok_mod  = load_from("tokenizer/tokenizer.py", f"{model_id}.tokenizer")
                SanaConfig = sana_mod.SanaConfig
                SanaModel  = sana_mod.SanaModel
                Tokenizer  = tok_mod.Tokenizer

                ft_dir = mdir / "checkpoints" / "finetune"
                cfg = SanaConfig.load(str(ft_dir / "config.json"))

                tok_path = ft_dir / "tokenizer.json"
                if not tok_path.exists():
                    tok_path = mdir / "tokenizer" / "sana_tokenizer" / "tokenizer.json"
                tokenizer = Tokenizer(str(tok_path))

                ckpt = torch.load(str(ckpt_path), map_location="cpu")
                state_dict = ckpt.get("model", ckpt)
                # strip torch.compile prefix if present
                state_dict = {
                    k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
                }
                model = SanaModel(cfg).to(device)
                model.load_state_dict(state_dict, strict=False)
                model.eval()

                # Mark this path so generate_response uses the SanaModel route.
                model._finetune = True
                tokenizer._finetune = True

            elif _is_v3_model(model_id):
                # ── v3 loader: HuggingFace tokenizer + model/model.py ──────
                from transformers import PreTrainedTokenizerFast

                sana_mod = load_from("model/model.py", f"{model_id}.model")
                SanaConfig = sana_mod.SanaConfig
                SanaModel  = sana_mod.SanaModel

                ft_dir  = mdir / "checkpoints" / "finetune"
                cfg     = SanaConfig.load(str(ft_dir / "config.json"))
                tok_dir = ft_dir / "tokenizer"
                tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_dir))

                ckpt  = torch.load(str(ckpt_path), map_location=device, weights_only=True)
                model = SanaModel(cfg).to(device)
                model.load_state_dict(ckpt.get("model", ckpt))
                model.eval()

                # Attach v3 marker so generate_response knows which path to use
                model._v3 = True
                tokenizer._v3 = True

                # Store special token ids on tokenizer for easy access
                def _tid(name):
                    return tokenizer.convert_tokens_to_ids(name)
                tokenizer.user_tok = _tid("<|user|>")
                tokenizer.sana_tok = _tid("<|sana|>")
                tokenizer.end_tok  = _tid("<|end|>")

            else:
                # ── v1/v2 loader: custom tokenizer + configs/model_config.py ──
                cfg_mod = load_from("configs/model_config.py", f"{model_id}.model_config")
                gpt_mod = load_from("model/gpt.py", f"{model_id}.gpt")
                tok_mod = load_from("tokenizer/tokenizer.py", f"{model_id}.tokenizer")

                ModelConfig = cfg_mod.ModelConfig

                ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
                cfg  = ckpt.get("cfg", ModelConfig())

                ModelClass = (
                    getattr(gpt_mod, "Sana", None)
                    or getattr(gpt_mod, "TinyGPT", None)
                    or getattr(gpt_mod, "GPT", None)
                )
                if ModelClass is None:
                    raise AttributeError(
                        f"gpt.py in {model_id} has no Sana, TinyGPT, or GPT class"
                    )
                model = ModelClass(cfg).to(device)
                model.load_state_dict(ckpt["model_state"])
                model.eval()

                tokenizer_class = tok_mod.Tokenizer
                tok_model_path  = str(mdir / "tokenizer" / "tok_v2.model")
                if not os.path.exists(tok_model_path):
                    tok_model_path = str(mdir / "tokenizer" / "tok.model")
                tokenizer = tokenizer_class(model_path=tok_model_path)

            elapsed = time.time() - t0
            try:
                n_params = model.num_parameters()
            except AttributeError:
                n_params = sum(p.numel() for p in model.parameters())
            print(
                f"[server] Loaded {model_id} in {elapsed:.1f}s  "
                f"({n_params:,} params)"
            )

            with state_lock:
                state["model"] = model
                state["tokenizer"] = tokenizer
                state["cfg"] = cfg
                state["model_id"] = model_id
                state["model_path"] = model_path
                state["loading"] = False
                state["device"] = device

        except Exception as e:
            import traceback

            err = traceback.format_exc()
            print(f"[server] Load failed: {err}")
            with state_lock:
                state["loading"] = False
                state["error"] = str(e)
                state["model"] = None
                state["model_id"] = None

    t = threading.Thread(target=_load, daemon=True)
    t.start()


# ── Generation ────────────────────────────────────────────────────────────────


def _is_v2_tokenizer(tokenizer) -> bool:
    """
    v2 tokenizers have build_inference_prompt + END_ID (uppercase).
    v1 tokenizers have apply_chat_template + eos_id (lowercase).
    """
    return hasattr(tokenizer, "build_inference_prompt") and hasattr(tokenizer, "END_ID")


def _is_v3_model(model_id: str) -> bool:
    """v3 models use HuggingFace PreTrainedTokenizerFast + model/model.py."""
    return model_id.endswith("_v3")


def _is_finetune_layout(model_dir: Path) -> bool:
    """
    New portfolio layout: model/model.py (SanaConfig/SanaModel) + a custom
    tokenizer/tokenizer.py, with config.json + tokenizer.json living inside
    checkpoints/finetune/ alongside ckpt_epoch_NN.pt files.
    Detected by structure, not by folder name, so sana_24M / sana_100M both match.
    """
    ft_dir = model_dir / "checkpoints" / "finetune"
    return (
        (model_dir / "model" / "model.py").exists()
        and (model_dir / "tokenizer" / "tokenizer.py").exists()
        and (ft_dir / "config.json").exists()
    )


def _get_inf_cfg(model_id: str):
    """Load InferenceConfig from the model's own model_config.py."""
    try:
        import importlib.util

        mdir = Path(state["model_path"])
        spec = importlib.util.spec_from_file_location(
            f"{model_id}.inf_cfg", mdir / "configs/model_config.py"
        )
        cfg_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_mod)
        return cfg_mod.InferenceConfig()
    except Exception:

        class _Defaults:
            max_new_tokens = 200
            temperature = 1.0
            top_k = 50
            top_p = 0.85
            repetition_penalty = 1.05

        return _Defaults()


def _build_v1_prompt(
    tokenizer, message: str, history: list, max_history: int
) -> list[int]:
    """
    v1 prompt format (from v1 chat.py — apply_chat_template style):
      [PAST MESSAGES]
      You: ...
      Assistant: ...
      [CURRENT MESSAGE]
      You: {message}
      [ASSISTANT REPLY]
      Assistant:
    v1 uses tokenizer.encode(text, add_bos=True) and tokenizer.eos_id (lowercase).
    """
    lines = []
    recent = history[-(max_history * 2) :]
    if recent:
        lines.append("[PAST MESSAGES]")
        for turn in recent:
            role = "You" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role}: {turn['content']}")
    lines.append("[CURRENT MESSAGE]")
    lines.append(f"You: {message}")
    lines.append("[ASSISTANT REPLY]")
    lines.append("Assistant:")
    prompt_text = "\n".join(lines)
    return tokenizer.encode(prompt_text, add_bos=True)


def _build_v2_prompt(
    tokenizer, message: str, history: list, max_history: int
) -> list[int]:
    """
    v2 prompt format (from v2 chat.py — build_inference_prompt):
      <bos><|user|>msg<|end|><|sana|>reply<|end|>...<|user|>msg<|end|><|sana|>
    v2 uses tokenizer.EOS_ID and tokenizer.END_ID (uppercase).
    """
    return tokenizer.build_inference_prompt(
        current_message=message,
        history=history,
        max_history_turns=max_history,
    )


def generate_response(message: str, history: list) -> str:
    with state_lock:
        model = state["model"]
        tokenizer = state["tokenizer"]
        device = state["device"]
        model_id = state["model_id"]

    if model is None:
        return "[no model loaded]"

    inf_cfg = _get_inf_cfg(model_id)
    MAX_HISTORY = 4

    if getattr(tokenizer, "_finetune", False):
        # ── new layout: custom Tokenizer + SanaModel.generate() ──────────
        # Mirrors inference/inference.py: build_inference_prompt takes the
        # message + history as (user, sana) tuples, and SanaModel.generate()
        # accepts only temperature/top_p/eos_token_id/repetition_penalty.
        import re as _re

        # /api/chat sends history as [{"role","content"}, ...]; the tokenizer
        # wants [(user_text, sana_text), ...]. Pair them up newest-last.
        turns = []
        pending_user = None
        for msg in history[-(MAX_HISTORY * 2):]:
            if msg.get("role") == "user":
                pending_user = msg.get("content", "")
            elif msg.get("role") in ("sana", "assistant") and pending_user is not None:
                turns.append((pending_user, msg.get("content", "")))
                pending_user = None

        prompt_ids = tokenizer.build_inference_prompt(
            message=message,
            history=turns,
            max_seq_len=model.config.max_seq_len,
        )

        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_tensor,
                max_new_tokens=inf_cfg.max_new_tokens,
                temperature=inf_cfg.temperature,
                top_p=inf_cfg.top_p,
                eos_token_id=tokenizer.END_ID,
                repetition_penalty=inf_cfg.repetition_penalty,
            )

        new_ids = output_ids[0, len(prompt_ids):].tolist()
        if tokenizer.END_ID in new_ids:
            new_ids = new_ids[:new_ids.index(tokenizer.END_ID)]
        response = tokenizer.decode(new_ids, skip_special=False)

        # Same emotion-token -> emoji cleanup the CLI does, so UI matches.
        EMOTION_MAP = {
            "<sana_salute>": "🪼", "<sana_happy>": "✨", "<sana_think>": "🤔",
            "<sana_sad>": "😔", "<sana_smug>": "😏",
        }
        for tag in ("</sana_think>", "</sana_happy>", "</sana_smug>",
                    "</sana_sad>", "</sana_salute>"):
            response = response.replace(tag, "")
        for pat in (r"<\|pad\|>", r"<\|user\|>", r"<\|sana\|>", r"<\|end\|>"):
            response = _re.sub(pat, "", response)
        for token, emoji in EMOTION_MAP.items():
            response = response.replace(token, emoji)
        response = _re.sub(r"\s+", " ", response).strip()
        return response

    if getattr(tokenizer, "_v3", False):
        # ── v3: HuggingFace tokenizer + SanaModel.generate() ─────────────
        import re as _re

        # Build prompt: pack history then new user turn
        max_seq = model.cfg.max_seq_len
        user_tok = tokenizer.user_tok
        sana_tok = tokenizer.sana_tok
        end_tok  = tokenizer.end_tok

        new_turn = (
            [user_tok]
            + tokenizer.encode(message, add_special_tokens=False)
            + [end_tok, sana_tok]
        )
        reserved = len(new_turn) + 80
        budget   = max_seq - reserved

        # Fill history newest-first until budget exhausted
        segs = []
        used = 0
        for turn in reversed(history[-(MAX_HISTORY * 2):]):
            tok = user_tok if turn["role"] == "user" else sana_tok
            seg = ([tok]
                   + tokenizer.encode(turn["content"], add_special_tokens=False)
                   + [end_tok])
            if used + len(seg) > budget:
                break
            segs.insert(0, seg)
            used += len(seg)

        ids = []
        for seg in segs:
            ids.extend(seg)
        ids.extend(new_turn)

        input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=inf_cfg.max_new_tokens,
                temperature=inf_cfg.temperature,
                top_p=inf_cfg.top_p,
                eos_token_id=end_tok,
                repetition_penalty=inf_cfg.repetition_penalty,
            )

        new_ids = output_ids[0, len(ids):].tolist()
        if end_tok in new_ids:
            new_ids = new_ids[:new_ids.index(end_tok)]

        response = tokenizer.decode(new_ids, skip_special_tokens=False).strip()

        # Handle tool calls
        TOOL_RE = _re.compile(r"<tool>(.*?)</tool>", _re.DOTALL)
        if "<tool>" in response:
            try:
                sys.path.insert(0, state["model_path"])
                from tools.tool_router import route_tool
                response = _re.sub(
                    r"<tool_result>.*?</tool_result>", "", response, flags=_re.DOTALL
                )
                def _replace(m):
                    result = route_tool(m.group(1).strip())
                    return f"<tool>{m.group(1)}</tool><tool_result>{result}</tool_result>"
                response = TOOL_RE.sub(_replace, response)
            except Exception:
                pass  # tool routing failed — return raw response

        return response

    elif _is_v2_tokenizer(tokenizer):
        # ── v2: special token format ──────────────────────────────────────
        # EOS_ID and END_ID are uppercase attributes
        # model.generate() accepts stop_token_ids
        # tokenizer.decode() accepts skip_special=True
        prompt_ids = _build_v2_prompt(tokenizer, message, history, MAX_HISTORY)
        eos_id = tokenizer.EOS_ID
        stop_ids = [tokenizer.END_ID]

        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=inf_cfg.max_new_tokens,
                temperature=inf_cfg.temperature,
                top_k=inf_cfg.top_k,
                top_p=inf_cfg.top_p,
                repetition_penalty=inf_cfg.repetition_penalty,
                eos_token_id=eos_id,
                stop_token_ids=stop_ids,
            )

        new_ids = output_ids[0, len(prompt_ids) :].tolist()
        stop_set = {eos_id, tokenizer.END_ID}
        while new_ids and new_ids[-1] in stop_set:
            new_ids.pop()
        return tokenizer.decode(new_ids, skip_special=True).strip()

    else:
        # ── v1: string-marker format ──────────────────────────────────────
        # eos_id is lowercase attribute
        # model.generate() does NOT accept stop_token_ids
        # tokenizer.decode() does NOT accept skip_special kwarg
        prompt_ids = _build_v1_prompt(tokenizer, message, history, MAX_HISTORY)
        eos_id = tokenizer.eos_id

        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=inf_cfg.max_new_tokens,
                temperature=inf_cfg.temperature,
                top_k=inf_cfg.top_k,
                top_p=inf_cfg.top_p,
                repetition_penalty=inf_cfg.repetition_penalty,
                eos_token_id=eos_id,
            )

        new_ids = output_ids[0, len(prompt_ids) :].tolist()
        if new_ids and new_ids[-1] == eos_id:
            new_ids.pop()
        return tokenizer.decode(new_ids).strip()


# ── HTTP Handler ──────────────────────────────────────────────────────────────


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress default access log noise

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, content_type: str):
        try:
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()

    def read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")

        elif path == "/api/models":
            models = discover_models()
            with state_lock:
                current = state["model_id"]
                loading = state["loading"]
            self.send_json({"models": models, "active": current, "loading": loading})

        elif path == "/api/status":
            with state_lock:
                s = {
                    "model_id": state["model_id"],
                    "loading": state["loading"],
                    "error": state["error"],
                    "ready": state["model"] is not None,
                    "device": str(state["device"]) if state["device"] else None,
                }
                if state["cfg"]:
                    cfg = state["cfg"]
                    try:
                        n_params = state["model"].num_parameters()
                    except AttributeError:
                        n_params = sum(p.numel() for p in state["model"].parameters())
                    s["params"] = n_params
                    s["vocab"] = cfg.vocab_size
                    s["d_model"] = getattr(cfg, "d_model", None) or getattr(cfg, "hidden_dim", None)
                    s["n_layer"] = getattr(cfg, "n_layer", None) or getattr(cfg, "num_layers", None)
            self.send_json(s)

        else:
            # Serve static files
            rel = path.lstrip("/")
            p = STATIC_DIR / rel
            ext_map = {
                ".html": "text/html",
                ".css": "text/css",
                ".js": "application/javascript",
                ".png": "image/png",
                ".ico": "image/x-icon",
                ".svg": "image/svg+xml",
            }
            ct = ext_map.get(Path(rel).suffix, "application/octet-stream")
            self.send_file(p, ct)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/load":
            body = self.read_body()
            mid = body.get("model_id", "")
            models = discover_models()
            match = next((m for m in models if m["id"] == mid), None)
            if match is None:
                self.send_json({"error": f"Unknown model: {mid}"}, 400)
                return
            with state_lock:
                if state["loading"]:
                    self.send_json({"error": "Already loading"}, 409)
                    return
            load_model_async(mid, match["path"])
            self.send_json({"status": "loading", "model_id": mid})

        elif path == "/api/chat":
            with state_lock:
                ready = state["model"] is not None
                loading = state["loading"]
            if not ready:
                msg = "loading..." if loading else "no model loaded"
                self.send_json({"error": msg}, 503)
                return
            body = self.read_body()
            message = body.get("message", "").strip()
            history = body.get("history", [])
            if not message:
                self.send_json({"error": "empty message"}, 400)
                return
            try:
                reply = generate_response(message, history)
                self.send_json({"reply": reply})
            except Exception as e:
                self.send_json({"error": str(e)}, 500)

        elif path == "/api/reset":
            self.send_json({"status": "ok"})

        else:
            self.send_json({"error": "not found"}, 404)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Model ID to load on startup (e.g. sana_23M_v2)",
    )
    args = parser.parse_args()

    # Auto-load on startup if requested
    if args.load:
        models = discover_models()
        match = next((m for m in models if m["id"] == args.load), None)
        if match:
            print(f"[server] Auto-loading {args.load} on startup...")
            load_model_async(args.load, match["path"])
        else:
            print(
                f"[server] Warning: --load model '{args.load}' not found. "
                f"Available: {[m['id'] for m in models]}"
            )

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[server] Sana server running on http://0.0.0.0:{args.port}")
    print(f"[server] Scanning for models in: {SANA_ROOT}")
    models = discover_models()
    for m in models:
        print(f"  • {m['id']} ({m['params']})  ckpt: {Path(m['ckpt']).name}")
    if not models:
        print(
            "  (no models found — make sure sana_* folders are in the parent directory)"
        )
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] Shutting down.")


if __name__ == "__main__":
    main()
