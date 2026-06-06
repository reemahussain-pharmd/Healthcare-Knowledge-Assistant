"""
LLM client supporting OpenAI, Google Gemini, and built-in Demo Mode.
API keys are loaded from environment variables (never hardcoded).
Demo Mode works with NO API key — answers are built from retrieved chunks.
"""

import os
import re
import time
import logging
import warnings
from typing import Optional

# Suppress SSL warnings for corporate network environments
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

logger = logging.getLogger(__name__)

HEALTHCARE_SYSTEM_PROMPT = """You are a healthcare research assistant specializing in clinical research, \
medical guidelines, pharmacology, and clinical trial analysis.

Answer ONLY using the information provided in the context below.
If the answer is not present in the provided context, respond with:
"The provided documents do not contain sufficient information to answer this question."

Rules:
- Always cite the source documents by referencing [Source N] tags from the context.
- Do NOT fabricate, hallucinate, or infer information beyond what is stated.
- Use precise medical and scientific language.
- If multiple sources address the question, synthesize them clearly.
- For dosage, safety, or treatment information, always note that clinical decisions \
  require consultation with a qualified healthcare professional."""


class LLMClient:
    """
    Unified LLM client.
    Providers: 'openai' | 'gemini' | 'demo'
    'auto' → tries OpenAI → Gemini → falls back to Demo Mode.
    """

    def __init__(self, provider: str = "auto", model: str = None):
        self.provider = self._resolve_provider(provider)
        self.model = model or self._default_model()
        self._client = None
        logger.info(f"LLM client: provider={self.provider}, model={self.model}")

    def _resolve_provider(self, provider: str) -> str:
        if provider == "demo":
            return "demo"
        if provider != "auto":
            return provider.lower()
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            return "gemini"
        # No key found — use Demo Mode automatically
        logger.warning("No LLM API key found — running in Demo Mode.")
        return "demo"

    def _default_model(self) -> str:
        return {
            "openai": "gpt-3.5-turbo",
            "gemini": "gemini-2.0-flash-lite",   # updated to latest available
            "demo":   "Demo Mode (No API Key)",
        }.get(self.provider, "demo")

    # ── OpenAI ────────────────────────────────────────────────────────────────
    def _init_openai(self):
        if self._client is not None:
            return
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        except ImportError:
            raise ImportError("pip install openai")

    def _call_openai(self, full_prompt, system_prompt, temperature, max_tokens):
        self._init_openai()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": full_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip(), response.usage.total_tokens

    # ── Gemini ────────────────────────────────────────────────────────────────
    def _init_gemini(self):
        """No persistent client needed — we call REST API directly."""
        pass

    def _call_gemini(self, full_prompt, temperature, max_tokens):
        """
        Call Gemini via direct REST API (requests + SSL disabled).
        Works on corporate/restricted networks that block gRPC.
        Tries new google-genai SDK first; falls back to raw REST.
        """
        import requests, json

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("No GEMINI_API_KEY found in environment.")

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        resp = requests.post(
            url,
            json=payload,
            verify=False,          # bypass corporate SSL inspection
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return text, 0

    # ── Demo Mode (No API Key required) ──────────────────────────────────────
    def _call_demo(self, question: str, context: str) -> tuple:
        """
        Build a structured answer directly from retrieved chunks.
        No external API call — works 100% offline.
        """
        time.sleep(0.5)  # simulate a brief processing delay

        # Extract source blocks like "[Source 1] Title ..."
        source_blocks = re.split(r"\[Source \d+\]", context)
        sources_text  = [b.strip() for b in source_blocks if b.strip()]

        if not sources_text:
            return (
                "The provided documents do not contain sufficient information to answer this question.",
                0,
            )

        q_lower = question.lower()

        # Pull the most relevant sentences from each source
        answer_parts = []
        cited = []
        source_headers = re.findall(r"\[Source \d+\][^\n]*", context)

        for i, (header, text) in enumerate(
            zip(source_headers, sources_text), start=1
        ):
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 40]
            relevant = []
            for sent in sentences:
                sl = sent.lower()
                # Score sentence by keyword overlap with question
                q_words = set(re.findall(r"\b\w{4,}\b", q_lower))
                s_words = set(re.findall(r"\b\w{4,}\b", sl))
                if len(q_words & s_words) >= 1:
                    relevant.append(sent)
            if relevant:
                answer_parts.append(f"According to {header}:\n" + " ".join(relevant[:4]))
                cited.append(f"[Source {i}]")

        if not answer_parts:
            # Fall back: return first 3 sentences from top source
            first_sentences = [
                s.strip() for s in re.split(r"(?<=[.!?])\s+", sources_text[0])
                if len(s.strip()) > 40
            ][:3]
            answer_parts = [
                f"Based on the retrieved documents:\n" + " ".join(first_sentences)
            ]
            cited = ["[Source 1]"]

        answer = "\n\n".join(answer_parts)
        answer += (
            f"\n\n---\n📌 **Sources cited:** {', '.join(cited)}\n"
            f"⚠️ *Demo Mode — answers extracted directly from document text. "
            f"Add an OpenAI or Gemini API key in `.env` for full AI-generated answers.*"
        )
        return answer, 0

    # ── Public generate() ─────────────────────────────────────────────────────
    def generate(
        self,
        user_prompt: str,
        context: str,
        system_prompt: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1500,
    ) -> dict:
        sys_prompt   = system_prompt or HEALTHCARE_SYSTEM_PROMPT
        full_prompt  = (
            f"{sys_prompt}\n\n"
            f"=== CONTEXT FROM DOCUMENTS ===\n{context}\n\n"
            f"=== QUESTION ===\n{user_prompt}\n\n"
            f"=== ANSWER ==="
        )

        start = time.time()
        try:
            if self.provider == "openai":
                answer, tokens = self._call_openai(full_prompt, sys_prompt, temperature, max_tokens)
            elif self.provider == "gemini":
                answer, tokens = self._call_gemini(full_prompt, temperature, max_tokens)
            else:
                answer, tokens = self._call_demo(user_prompt, context)
        except Exception as e:
            # On any API error (quota, network, etc.) fall back to Demo Mode
            logger.warning(f"LLM API error ({self.provider}): {e} — falling back to Demo Mode")
            self.provider = "demo"
            self.model    = "Demo Mode (Fallback)"
            answer, tokens = self._call_demo(user_prompt, context)

        return {
            "answer":         answer,
            "provider":       self.provider,
            "model":          self.model,
            "tokens_used":    tokens,
            "response_time_s": round(time.time() - start, 2),
        }

    @property
    def provider_info(self) -> dict:
        return {"provider": self.provider, "model": self.model}
