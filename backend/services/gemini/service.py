"""GeminiService — the core AI agent class."""

import json
import hashlib
import logging
import asyncio
from datetime import datetime
from typing import Any, Optional

from google import genai
from google.genai import types as genai_types
from cachetools import TTLCache
from pydantic import BaseModel, Field, ValidationError

from config import get_settings
from models.simulation import Simulation
from models.scenario import Scenario
from models.decision import Decision
from schemas.reflection import (
    ReflectionResponse,
    ProcessQuality,
    PatternDetection,
    Counterfactual,
    ActionableInsight,
    DecisionExplanation,
    WhyThisDecisionResponse,
    ProDecision,
    ProComparisonResponse,
)

from services.gemini.helpers import (
    logger,
    settings,
    _cache,
    _context_cache_store,
    THINKING_LEVELS,
    _get_market_state,
    _decision_trace,
    _scenario_summary,
    _cache_key,
)
from services.gemini.schemas import (
    _ReflectionGeminiOutput,
    _CounterfactualGeminiOutput,
    _WhyGeminiOutput,
    _ProGeminiOutput,
    _CoachingGeminiOutput,
    _ProfileUpdateGeminiOutput,
    _BatchAnalysisGeminiOutput,
    _BiasHeatmapGeminiOutput,
    _BiasHeatmapEntry,
    _RationaleReviewOutput,
    _RationaleCritique,
    _IsolatedCounterfactualOutput,
    _PlaybookOutput,
    _PlaybookAdherenceOutput,
    _LiveNudgeOutput,
    _ChallengeOutput,
    _AdaptiveScenarioOutput,
    _GeneratedModulesGeminiOutput,
    _CredibilityCheckOutput,
    _URLScenarioOutput,
    _BiasClassifierGeminiOutput,
    _ConfidenceCalibrationGeminiOutput,
    _BehaviorHistoryGeminiOutput,
)

class GeminiService:
    """Behavioral-AI agent backed by Gemini with structured JSON output."""

    def __init__(self):
        self.use_mock = settings.use_mock_gemini
        self.max_retries = settings.gemini_max_retries
        self.timeout = settings.gemini_timeout_seconds
        self.model_name = settings.gemini_model

        if not self.use_mock and settings.gemini_api_key:
            self.client = genai.Client(api_key=settings.gemini_api_key)
        else:
            self.client = None

    # ── Low-level Gemini call with retries + timeout + schema validation ──

    async def _call_gemini(
        self,
        prompt: str,
        response_schema: type,  # Pydantic model for validation
        cache_key_str: str | None = None,
        call_type: str = "reflection",
        use_search_grounding: bool = False,
        use_url_context: bool = False,
        cached_content_name: str | None = None,
    ) -> dict:
        """
        Call Gemini with automatic retries, timeout, rate-limit back-off,
        Pydantic schema validation, thinking-level control, optional
        Google Search grounding, URL context, and optional context caching.
        Returns the validated dict.
        """
        # Check in-memory cache first
        if cache_key_str and cache_key_str in _cache:
            logger.info("Cache hit for %s", cache_key_str)
            return _cache[cache_key_str]

        last_error: Exception | None = None

        # Determine thinking level from call type
        thinking_level = THINKING_LEVELS.get(call_type, "medium")

        # Build tools list (e.g. Search grounding, URL context)
        tools = []
        if use_search_grounding and settings.gemini_enable_search_grounding:
            tools.append(genai_types.Tool(google_search=genai_types.GoogleSearch()))
        if use_url_context and settings.gemini_enable_url_context:
            tools.append(genai_types.Tool(url_context=genai_types.UrlContext()))

        for attempt in range(1, self.max_retries + 1):
            try:
                # Build config
                config_kwargs = {
                    "response_mime_type": "application/json",
                    "temperature": 1.0,  # Gemini 3: keep at 1.0 to avoid looping/degraded reasoning
                    "thinking_config": genai_types.ThinkingConfig(
                        include_thoughts=True,
                    ),
                }
                if tools:
                    config_kwargs["tools"] = tools

                # Use cached_content if available (context caching)
                generate_kwargs = {
                    "model": self.model_name,
                    "contents": prompt,
                    "config": genai_types.GenerateContentConfig(**config_kwargs),
                }
                if cached_content_name:
                    generate_kwargs["cached_content"] = cached_content_name

                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.models.generate_content,
                        **generate_kwargs,
                    ),
                    timeout=self.timeout,
                )

                raw_text = response.text.strip()
                # Parse JSON
                data = json.loads(raw_text)

                # Validate against Pydantic schema
                validated = response_schema.model_validate(data)
                result = validated.model_dump()

                # Extract grounding metadata if search grounding was used
                if use_search_grounding:
                    grounding = self._extract_grounding_metadata(response)
                    if grounding:
                        result["_grounding_metadata"] = grounding

                # Extract URL context metadata if URL context was used
                if use_url_context:
                    url_meta = self._extract_url_context_metadata(response)
                    if url_meta:
                        result["_url_context_metadata"] = url_meta

                # Cache the result
                if cache_key_str:
                    _cache[cache_key_str] = result

                # Log thinking level and grounding usage
                logger.info(
                    "Gemini call [%s] thinking=%s grounding=%s cached_ctx=%s",
                    call_type, thinking_level, bool(tools),
                    bool(cached_content_name),
                )

                return result

            except asyncio.TimeoutError:
                logger.warning("Gemini timeout attempt %d/%d", attempt, self.max_retries)
                last_error = TimeoutError("Gemini call timed out")

            except json.JSONDecodeError as e:
                logger.warning("Gemini returned invalid JSON attempt %d: %s", attempt, e)
                last_error = e

            except ValidationError as e:
                logger.warning("Gemini output failed schema validation attempt %d: %s", attempt, e)
                last_error = e

            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate" in error_str or "quota" in error_str:
                    wait = 2 ** attempt
                    logger.warning("Rate limited, backing off %ds", wait)
                    await asyncio.sleep(wait)
                else:
                    logger.error("Gemini error attempt %d: %s", attempt, e)
                last_error = e

            # Exponential back-off between retries
            if attempt < self.max_retries:
                await asyncio.sleep(1.5 ** attempt)

        raise RuntimeError(
            f"Gemini call failed after {self.max_retries} attempts: {last_error}"
        )

    # ── Grounding Metadata Extraction ──────────────────────────────────────

    @staticmethod
    def _extract_grounding_metadata(response) -> dict | None:
        """
        Extract structured grounding metadata from a search-grounded response.
        Returns dict with grounding_chunks (source URIs) and grounding_supports
        (text segments linked to sources), or None if no grounding data found.

        See: https://ai.google.dev/gemini-api/docs/grounding/search-suggestions
        """
        try:
            candidate = response.candidates[0] if response.candidates else None
            if not candidate or not hasattr(candidate, "grounding_metadata"):
                return None

            gm = candidate.grounding_metadata
            if not gm:
                return None

            result = {}

            # Extract search queries used
            if hasattr(gm, "web_search_queries") and gm.web_search_queries:
                result["search_queries"] = list(gm.web_search_queries)

            # Extract grounding chunks — source URIs and titles
            if hasattr(gm, "grounding_chunks") and gm.grounding_chunks:
                chunks = []
                for chunk in gm.grounding_chunks:
                    if hasattr(chunk, "web") and chunk.web:
                        chunks.append({
                            "uri": getattr(chunk.web, "uri", ""),
                            "title": getattr(chunk.web, "title", ""),
                        })
                if chunks:
                    result["grounding_chunks"] = chunks

            # Extract grounding supports — text segments linked to source indices
            if hasattr(gm, "grounding_supports") and gm.grounding_supports:
                supports = []
                for support in gm.grounding_supports:
                    s = {}
                    if hasattr(support, "segment") and support.segment:
                        s["text"] = getattr(support.segment, "text", "")
                    if hasattr(support, "grounding_chunk_indices"):
                        s["chunk_indices"] = list(
                            support.grounding_chunk_indices or []
                        )
                    if hasattr(support, "confidence_scores"):
                        s["confidence_scores"] = [
                            round(c, 3)
                            for c in (support.confidence_scores or [])
                        ]
                    if s:
                        supports.append(s)
                if supports:
                    result["grounding_supports"] = supports

            # Extract search entry point (rendered HTML for search suggestions)
            if hasattr(gm, "search_entry_point") and gm.search_entry_point:
                sep = gm.search_entry_point
                if hasattr(sep, "rendered_content") and sep.rendered_content:
                    result["search_suggestion_html"] = sep.rendered_content

            return result if result else None

        except Exception as e:
            logger.debug("Failed to extract grounding metadata: %s", e)
            return None

    # ── URL Context Metadata Extraction ────────────────────────────────────

    @staticmethod
    def _extract_url_context_metadata(response) -> dict | None:
        """
        Extract url_context_metadata from a response that used the URL context tool.
        Returns dict with url_metadata (retrieved URLs and their statuses),
        or None if no URL context data found.
        """
        try:
            candidate = response.candidates[0] if response.candidates else None
            if not candidate:
                return None

            ucm = getattr(candidate, "url_context_metadata", None)
            if not ucm:
                return None

            result = {}
            url_metadata = getattr(ucm, "url_metadata", None)
            if url_metadata:
                urls = []
                for um in url_metadata:
                    entry = {}
                    if hasattr(um, "retrieved_url") and um.retrieved_url:
                        entry["retrieved_url"] = um.retrieved_url
                    if hasattr(um, "url_retrieval_status") and um.url_retrieval_status:
                        entry["status"] = str(um.url_retrieval_status)
                    if entry:
                        urls.append(entry)
                if urls:
                    result["url_metadata"] = urls

            return result if result else None

        except Exception as e:
            logger.debug("Failed to extract URL context metadata: %s", e)
            return None

    # ── Context Caching — cache shared scenario+decision prefix ──────────

    async def _get_or_create_context_cache(
        self,
        simulation_id: str,
        scenario: Scenario,
        decisions: list[Decision],
    ) -> str | None:
        """
        Create or reuse a Gemini context cache containing the scenario summary
        and decision trace. This prefix is shared across reflection, counterfactuals,
        why, pro-comparison, coaching, etc. — saving ~5x input tokens.
        Returns the cached_content name, or None if caching fails.
        """
        if self.use_mock or not self.client:
            return None

        # Return existing cache if available
        if simulation_id in _context_cache_store:
            return _context_cache_store[simulation_id]

        shared_prefix = (
            f"SCENARIO:\n{_scenario_summary(scenario)}\n\n"
            f"USER'S DECISION TRACE:\n{_decision_trace(decisions)}"
        )

        # Gemini requires minimum token count for explicit caching:
        # - Gemini 3 Pro: 4096 tokens, Gemini 3 Flash: 1024 tokens
        # Rough estimate: ~4 chars per token. Skip caching if prefix is too short.
        MIN_CHARS_FOR_CACHING = 16000  # ~4096 tokens * 4 chars/token
        if len(shared_prefix) < MIN_CHARS_FOR_CACHING:
            logger.info(
                "Context prefix too short for caching (%d chars < %d min), using inline",
                len(shared_prefix), MIN_CHARS_FOR_CACHING,
            )
            return None

        try:
            cached_content = await asyncio.to_thread(
                self.client.caches.create,
                model=self.model_name,
                config=genai_types.CreateCachedContentConfig(
                    contents=[shared_prefix],
                    ttl=f"{settings.gemini_context_cache_ttl_minutes * 60}s",
                    display_name=f"sim-{simulation_id[:8]}",
                ),
            )
            cache_name = cached_content.name
            _context_cache_store[simulation_id] = cache_name
            logger.info("Created context cache for simulation %s: %s", simulation_id[:8], cache_name)
            return cache_name
        except Exception as e:
            logger.warning("Context cache creation failed (will use inline): %s", e)
            return None

    # ── Pydantic wrapper schemas for Gemini structured output ────────────
    # These define what we ask Gemini to return as JSON.

    class _GeminiReflection(ProcessQuality.__class__.__bases__[0]):
        """Schema we send to Gemini for the full reflection output."""
        pass

    # ── PUBLIC METHODS ───────────────────────────────────────────────────

    async def analyze_simulation(
        self,
        simulation: Simulation,
        decisions: list[Decision],
        scenario: Scenario,
    ) -> ReflectionResponse:
        """Full reflection analysis — the core Gemini call."""
        if self.use_mock or not self.client:
            return self._heuristic_analyze(simulation, decisions, scenario)

        outcome = simulation.final_outcome or {}
        profit_loss = outcome.get("profit_loss", 0)

        prompt = self._build_reflection_prompt(simulation, decisions, scenario)

        cache_k = _cache_key(str(simulation.id), "reflection")

        # Try to use context caching for shared prefix
        ctx_name = await self._get_or_create_context_cache(
            str(simulation.id), scenario, decisions
        )

        try:
            data = await self._call_gemini(
                prompt, _ReflectionGeminiOutput, cache_k,
                call_type="reflection", cached_content_name=ctx_name,
            )
            # Merge simulation_id which Gemini doesn't know
            data["simulation_id"] = str(simulation.id)
            data["counterfactuals"] = []  # generated separately
            return ReflectionResponse.model_validate(data)
        except Exception as e:
            logger.error("Gemini reflection failed, falling back to heuristic: %s", e)
            return self._heuristic_analyze(simulation, decisions, scenario)

    async def generate_counterfactuals(
        self,
        simulation: Simulation,
        decisions: list[Decision],
        scenario: Scenario,
    ) -> list[Counterfactual]:
        """Dynamic counterfactuals generated from the actual decision trace."""
        if self.use_mock or not self.client:
            return self._heuristic_counterfactuals(simulation, decisions, scenario)

        prompt = self._build_counterfactual_prompt(simulation, decisions, scenario)
        cache_k = _cache_key(str(simulation.id), "counterfactuals")
        ctx_name = await self._get_or_create_context_cache(
            str(simulation.id), scenario, decisions
        )

        try:
            data = await self._call_gemini(
                prompt, _CounterfactualGeminiOutput, cache_k,
                call_type="counterfactuals", cached_content_name=ctx_name,
            )
            return [Counterfactual.model_validate(cf) for cf in data["counterfactuals"]]
        except Exception as e:
            logger.error("Gemini counterfactuals failed, falling back: %s", e)
            return self._heuristic_counterfactuals(simulation, decisions, scenario)

    async def explain_decisions(
        self,
        simulation: Simulation,
        decisions: list[Decision],
        scenario: Scenario,
    ) -> WhyThisDecisionResponse:
        """'Why this decision?' — Gemini explains each bias detection using actual actions."""
        if self.use_mock or not self.client:
            return self._heuristic_explain_decisions(simulation, decisions, scenario)

        prompt = self._build_why_prompt(simulation, decisions, scenario)
        cache_k = _cache_key(str(simulation.id), "why")
        ctx_name = await self._get_or_create_context_cache(
            str(simulation.id), scenario, decisions
        )

        try:
            data = await self._call_gemini(
                prompt, _WhyGeminiOutput, cache_k,
                call_type="why", cached_content_name=ctx_name,
            )
            data["simulation_id"] = str(simulation.id)
            return WhyThisDecisionResponse.model_validate(data)
        except Exception as e:
            logger.error("Gemini explain_decisions failed, falling back: %s", e)
            return self._heuristic_explain_decisions(simulation, decisions, scenario)

    async def compare_with_pro(
        self,
        simulation: Simulation,
        decisions: list[Decision],
        scenario: Scenario,
    ) -> ProComparisonResponse:
        """'What would a pro do?' — side-by-side with an expert decision path."""
        if self.use_mock or not self.client:
            return self._heuristic_pro_comparison(simulation, decisions, scenario)

        prompt = self._build_pro_prompt(simulation, decisions, scenario)
        cache_k = _cache_key(str(simulation.id), "pro")
        ctx_name = await self._get_or_create_context_cache(
            str(simulation.id), scenario, decisions
        )

        try:
            data = await self._call_gemini(
                prompt, _ProGeminiOutput, cache_k,
                call_type="pro", cached_content_name=ctx_name,
            )
            data["simulation_id"] = str(simulation.id)
            return ProComparisonResponse.model_validate(data)
        except Exception as e:
            logger.error("Gemini pro comparison failed, falling back: %s", e)
            return self._heuristic_pro_comparison(simulation, decisions, scenario)

    async def generate_coaching(
        self,
        simulation: Simulation,
        decisions: list[Decision],
        scenario: Scenario,
        behavior_profile: dict | None = None,
    ) -> str:
        """Personalized coaching message that adapts using the behavior profile."""
        if self.use_mock or not self.client:
            return self._heuristic_coaching(simulation, decisions, scenario, behavior_profile)

        prompt = self._build_coaching_prompt(simulation, decisions, scenario, behavior_profile)
        cache_k = _cache_key(str(simulation.id), "coaching")
        ctx_name = await self._get_or_create_context_cache(
            str(simulation.id), scenario, decisions
        )

        try:
            data = await self._call_gemini(
                prompt, _CoachingGeminiOutput, cache_k,
                call_type="coaching", cached_content_name=ctx_name,
            )
            return data["coaching_message"]
        except Exception as e:
            logger.error("Gemini coaching failed, falling back: %s", e)
            return self._heuristic_coaching(simulation, decisions, scenario, behavior_profile)

    async def update_behavior_profile(
        self,
        user_id: str,
        simulation: Simulation,
        decisions: list[Decision],
        scenario: Scenario,
        existing_profile: dict | None = None,
    ) -> dict:
        """Use Gemini to update the compressed behavior profile with new simulation data."""
        if self.use_mock or not self.client:
            return self._heuristic_profile_update(existing_profile, decisions)

        prompt = self._build_profile_prompt(simulation, decisions, scenario, existing_profile)
        cache_k = _cache_key(str(simulation.id), "profile_update")

        try:
            data = await self._call_gemini(
                prompt, _ProfileUpdateGeminiOutput, cache_k,
                call_type="profile_update",
            )
            return data
        except Exception as e:
            logger.error("Gemini profile update failed, falling back: %s", e)
            return self._heuristic_profile_update(existing_profile, decisions)

    # ── BIAS CLASSIFICATION WITH GEMINI ────────────────────────────────

    async def classify_biases_with_gemini(
        self,
        simulation: Simulation,
        decisions: list[Decision],
        scenario: Scenario,
        decision_features: list[dict],
    ) -> dict:
        """
        Gemini-powered bias classification that combines qualitative decision analysis
        with quantitative feature data from bias_classifier.extract_decision_features().

        Args:
            simulation: The completed simulation.
            decisions: All decisions in the simulation.
            scenario: The scenario the simulation ran.
            decision_features: List of feature dicts (one per decision), from
                bias_classifier.extract_decision_features().

        Returns:
            Dict matching _BiasClassifierGeminiOutput schema.
        """
        if self.use_mock or not self.client:
            return self._heuristic_bias_classification(simulation, decisions, scenario, decision_features)

        prompt = self._build_bias_classifier_prompt(simulation, decisions, scenario, decision_features)
        cache_k = _cache_key(str(simulation.id), "bias_classifier")
        ctx_name = await self._get_or_create_context_cache(
            str(simulation.id), scenario, decisions
        )

        try:
            data = await self._call_gemini(
                prompt, _BiasClassifierGeminiOutput, cache_k,
                call_type="bias_classifier", cached_content_name=ctx_name,
            )
            data["simulation_id"] = str(simulation.id)
            return data
        except Exception as e:
            logger.error("Gemini bias classification failed, falling back: %s", e)
            return self._heuristic_bias_classification(simulation, decisions, scenario, decision_features)

    # ── CONFIDENCE CALIBRATION WITH GEMINI ────────────────────────────

    async def calibrate_pattern_confidence(
        self,
        simulation: Simulation,
        decisions: list[Decision],
        scenario: Scenario,
        patterns_detected: list[dict],
        evidence_signals: dict,
    ) -> dict:
        """
        Gemini self-evaluates its own pattern detections against observable evidence.

        Args:
            simulation: The completed simulation.
            decisions: All decisions in the simulation.
            scenario: The scenario the simulation ran.
            patterns_detected: List of pattern dicts from the reflection (pattern_name, confidence, evidence).
            evidence_signals: Dict of evidence signals from confidence_calibrator — keyed
                by pattern name, values are lists of {check, matched, weight, detail}.

        Returns:
            Dict matching _ConfidenceCalibrationGeminiOutput schema.
        """
        if self.use_mock or not self.client:
            return self._heuristic_confidence_calibration(
                simulation, decisions, scenario, patterns_detected, evidence_signals
            )

        prompt = self._build_confidence_calibration_prompt(
            simulation, decisions, scenario, patterns_detected, evidence_signals
        )
        cache_k = _cache_key(str(simulation.id), "confidence_calibration")
        ctx_name = await self._get_or_create_context_cache(
            str(simulation.id), scenario, decisions
        )

        try:
            data = await self._call_gemini(
                prompt, _ConfidenceCalibrationGeminiOutput, cache_k,
                call_type="confidence_calibration", cached_content_name=ctx_name,
            )
            data["simulation_id"] = str(simulation.id)
            return data
        except Exception as e:
            logger.error("Gemini confidence calibration failed, falling back: %s", e)
            return self._heuristic_confidence_calibration(
                simulation, decisions, scenario, patterns_detected, evidence_signals
            )

    # ── BEHAVIOR HISTORY ANALYSIS WITH GEMINI ─────────────────────────

    async def analyze_user_behavior_history(
        self,
        history_summary: list[dict],
        existing_profile: dict | None = None,
    ) -> dict:
        """
        Gemini analyzes a user's ENTIRE behavioral history across all simulations
        to discover emerging patterns, learning trajectories, and personalized
        improvement recommendations.

        Args:
            history_summary: List of dicts, one per simulation:
                {simulation_id, scenario_name, category, difficulty, profit_loss,
                 process_quality_score, decisions_count, top_biases, timestamp}.
            existing_profile: The user's current behavior profile dict, if any.

        Returns:
            Dict matching _BehaviorHistoryGeminiOutput schema.
        """
        if self.use_mock or not self.client:
            return self._heuristic_behavior_history(history_summary, existing_profile)

        prompt = self._build_behavior_history_prompt(history_summary, existing_profile)

        # Cache key based on profile + number of sims (changes when new sim is added)
        cache_seed = f"history:{len(history_summary)}:{json.dumps(sorted((existing_profile or {}).get('bias_patterns', {}).items()))}"
        cache_k = f"behavior_history:{hashlib.md5(cache_seed.encode()).hexdigest()[:12]}"

        try:
            data = await self._call_gemini(
                prompt, _BehaviorHistoryGeminiOutput, cache_k,
                call_type="behavior_history",
            )
            return data
        except Exception as e:
            logger.error("Gemini behavior history analysis failed, falling back: %s", e)
            return self._heuristic_behavior_history(history_summary, existing_profile)

    # ── LEARNING MODULE GENERATION ─────────────────────────────────────

    async def generate_learning_modules(self, profile_data: dict) -> list[dict]:
        """Generate personalized learning modules targeting the user's weaknesses."""
        if self.use_mock or not self.client:
            return self._heuristic_learning_modules(profile_data)

        weaknesses = profile_data.get("weaknesses", [])
        bias_patterns = profile_data.get("bias_patterns", {})

        # Cache key based on profile weaknesses so modules regenerate when profile changes
        cache_seed = json.dumps(sorted(weaknesses) + sorted(f"{k}:{v:.1f}" for k, v in bias_patterns.items()))
        cache_k = f"learning_modules:{hashlib.md5(cache_seed.encode()).hexdigest()[:12]}"

        if cache_k in _cache:
            return _cache[cache_k]

        prompt = f"""<role>
You are a behavioral finance educator creating personalized learning modules.
</role>

<context>
<user_behavior_profile>
Strengths: {profile_data.get("strengths", [])}
Weaknesses: {weaknesses}
Bias patterns: {bias_patterns}
Decision style: {profile_data.get("decision_style", "unknown")}
Total simulations: {profile_data.get("total_simulations_analyzed", 0)}
</user_behavior_profile>
</context>

<task>
Create 3-5 learning modules specifically targeting this user's weaknesses and bias patterns. Each module should have 2-3 lessons and 3 quiz questions.
</task>

<output_format>
Return valid JSON:
{{
  "modules": [
    {{
      "id": "gen_<short_id>",
      "title": "Module title (max 50 chars)",
      "description": "2-3 sentence description of what user will learn",
      "category": "emotional" | "technical" | "social" | "confidence" | "patience" | "risk",
      "icon": "brain" | "calculator" | "newspaper" | "target" | "clock" | "shield",
      "lessons": [
        {{
          "id": "les_gen_<short_id>",
          "title": "Lesson title",
          "content": "2-3 paragraphs of educational content with specific examples and actionable techniques",
          "key_insight": "One-sentence takeaway"
        }}
      ],
      "quiz": [
        {{
          "question": "Scenario-based question",
          "options": ["4 options"],
          "correct": 0,
          "explanation": "Why this answer is correct"
        }}
      ]
    }}
  ]
}}
</output_format>

<constraints>
- Modules MUST directly address the user's specific weaknesses and bias patterns
- Content should reference real behavioral finance research (Kahneman, Thaler, etc.)
- Quiz questions should be scenario-based, not factual recall
- Each lesson should include at least one concrete technique the user can apply
- Don't repeat content from standard modules about emotional intelligence, risk management, social media, or confidence calibration
- Make content personal — reference the user's specific patterns where possible
</constraints>"""

        try:
            data = await self._call_gemini(
                prompt, _GeneratedModulesGeminiOutput, cache_k,
                call_type="learning_modules",
            )
            return data["modules"]
        except Exception as e:
            logger.error("Learning module generation failed, falling back: %s", e)
            return self._heuristic_learning_modules(profile_data)

    def _heuristic_learning_modules(self, profile_data: dict) -> list[dict]:
        """Reorder static modules by relevance to user's weaknesses."""
        from pathlib import Path
        data_path = Path(__file__).parent.parent / "data" / "learning_modules.json"
        if not data_path.exists():
            return []
        with open(data_path) as f:
            all_modules = json.load(f)

        weaknesses = profile_data.get("weaknesses", [])
        bias_patterns = profile_data.get("bias_patterns", {})

        # Map weaknesses/biases to module categories
        relevance_map = {
            "fomo": ["emotional", "social"],
            "fomo_susceptibility": ["emotional", "social"],
            "impulsivity": ["emotional", "patience"],
            "loss_aversion": ["emotional", "risk"],
            "overconfidence": ["confidence", "technical"],
            "anchoring": ["anchoring", "technical"],
            "social_proof_reliance": ["social"],
            "social_proof": ["social"],
        }

        # Score each module
        scored = []
        for mod in all_modules:
            score = 0
            cat = mod.get("category", "")
            for weakness in weaknesses:
                if cat in relevance_map.get(weakness, []):
                    score += 3
            for bias, strength in bias_patterns.items():
                if cat in relevance_map.get(bias, []):
                    score += strength * 2
            scored.append((mod, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored]

    # ── PROMPT BUILDERS ──────────────────────────────────────────────────

    def _build_reflection_prompt(self, simulation, decisions, scenario) -> str:
        outcome = simulation.final_outcome or {}
        return f"""<role>
You are a behavioral finance expert analyzing a user's investment simulation decisions.
You are precise, evidence-based, and focused on decision PROCESS over outcomes.
</role>

<context>
<scenario>
{_scenario_summary(scenario)}
</scenario>

<decision_trace>
{_decision_trace(decisions)}
</decision_trace>

<outcome>
Profit/Loss: ${outcome.get("profit_loss", 0):.2f}
Final Portfolio Value: ${outcome.get("final_value", 10000):.2f}
Process Quality Score (heuristic): {simulation.process_quality_score or 'N/A'}
</outcome>
</context>

<task>
Analyze the user's decision-making PROCESS, not just the outcome.
</task>

<output_format>
Return valid JSON matching this exact schema:
{{
  "outcome_summary": "+$X.XX" or "-$X.XX",
  "outcome_type": "profit" | "loss" | "break_even",
  "process_quality": {{
    "score": 0-100,
    "factors": {{"timing": 0-1, "information_usage": 0-1, "risk_sizing": 0-1, "emotional_control": 0-1}},
    "summary": "one paragraph assessment"
  }},
  "patterns_detected": [
    {{
      "pattern_name": "fomo" | "impulsivity" | "loss_aversion" | "overconfidence" | "anchoring" | "social_proof_reliance" | "balanced_approach",
      "confidence": 0-1,
      "evidence": ["cite specific actions from the trace above"],
      "description": "explain what this pattern means"
    }}
  ],
  "luck_factor": 0-1,
  "skill_factor": 0-1,
  "luck_skill_explanation": "explain whether the outcome was due to luck or skill, citing evidence",
  "insights": [
    {{
      "title": "short title",
      "description": "actionable advice",
      "related_pattern": "pattern_name or null",
      "recommended_card_id": "card_id or null"
    }}
  ],
  "key_takeaway": "the single most important lesson from this simulation",
  "coaching_message": "a personalized, encouraging coaching message addressing the user directly"
}}
</output_format>

<constraints>
- Every claim MUST cite specific evidence from the decision trace (decision number, timing, actions).
- If you are not confident about a pattern, set confidence below 0.5.
- Do NOT invent evidence. If the trace doesn't support a pattern, don't include it.
- The coaching message should be warm but honest, addressing the user's specific behavior.
- Focus on PROCESS quality, not outcome. A profitable simulation with poor process should score low.
</constraints>"""

    def _build_counterfactual_prompt(self, simulation, decisions, scenario) -> str:
        outcome = simulation.final_outcome or {}
        return f"""<role>
You are a behavioral finance expert generating counterfactual "what-if" scenarios.
</role>

<context>
<scenario>
{_scenario_summary(scenario)}
</scenario>

<decision_trace>
{_decision_trace(decisions)}
</decision_trace>

<actual_outcome>
Profit/Loss: ${outcome.get("profit_loss", 0):.2f}
</actual_outcome>
</context>

<task>
Generate 3 alternate timelines where the SAME user decisions play out under DIFFERENT market conditions. Each must be plausible given the scenario category.
</task>

<output_format>
Return valid JSON:
{{
  "counterfactuals": [
    {{
      "timeline_name": "short name",
      "description": "what-if question",
      "market_changes": "what would be different in this timeline",
      "outcome": {{"profit_loss": number, "final_value": number}},
      "lesson": "what this teaches about luck vs skill, citing the user's specific decisions"
    }}
  ]
}}
</output_format>

<constraints>
- Each timeline must reference the user's SPECIFIC decisions and show how they would play out differently.
- Outcomes must be numerically plausible given the starting balance of ${outcome.get("final_value", 10000):.0f}.
- Lessons must be specific, not generic. Reference actual decision numbers or timings from the trace.
</constraints>"""

    def _build_why_prompt(self, simulation, decisions, scenario) -> str:
        outcome = simulation.final_outcome or {}
        return f"""<role>
You are a behavioral psychologist analyzing investment decisions.
</role>

<context>
<scenario>
{_scenario_summary(scenario)}
</scenario>

<decision_trace>
{_decision_trace(decisions)}
</decision_trace>

<outcome>
Profit/Loss: ${outcome.get("profit_loss", 0):.2f}
</outcome>
</context>

<task>
For each decision that shows a cognitive bias or notable pattern, explain WHY the user likely made that decision based on the available evidence (timing, information viewed, market conditions, events).
</task>

<output_format>
Return valid JSON:
{{
  "explanations": [
    {{
      "decision_index": 0,
      "decision_type": "buy",
      "timestamp_seconds": 15,
      "detected_bias": "fomo",
      "explanation": "detailed explanation of why this specific decision suggests this bias",
      "evidence_from_actions": ["specific evidence from the trace"],
      "severity": "minor" | "moderate" | "significant"
    }}
  ],
  "overall_narrative": "A cohesive 2-3 sentence story of the user's emotional and cognitive journey through the simulation"
}}
</output_format>

<constraints>
- Only include decisions that clearly show a pattern. Skip unremarkable decisions.
- Explanations must reference SPECIFIC data: timing, information panels viewed/ignored, confidence levels, events that occurred.
- The overall_narrative should read like a story, not a list.
- If a decision was genuinely good, say so. Don't force bias detection.
</constraints>"""

    def _build_pro_prompt(self, simulation, decisions, scenario) -> str:
        outcome = simulation.final_outcome or {}
        return f"""<role>
You are modeling what an experienced, disciplined professional trader would do in this scenario.
</role>

<context>
<scenario>
{_scenario_summary(scenario)}
</scenario>

<decision_trace>
{_decision_trace(decisions)}
</decision_trace>

<user_outcome>
Profit/Loss: ${outcome.get("profit_loss", 0):.2f}
</user_outcome>
</context>

<task>
Generate a professional trader's alternative decision path for this exact scenario. The pro has the same information access as the user but uses disciplined analysis.
</task>

<output_format>
Return valid JSON:
{{
  "pro_decisions": [
    {{
      "at_timestamp": 0,
      "user_action": "what the user did at this point",
      "pro_action": "what a pro would do instead",
      "pro_reasoning": "the cognitive process behind the pro's decision",
      "outcome_difference": "how this changes the trajectory",
      "skill_demonstrated": "the specific skill the pro uses (e.g., patience, risk sizing, information gathering)"
    }}
  ],
  "pro_final_outcome": {{"profit_loss": number, "final_value": number}},
  "user_final_outcome": {{"profit_loss": {outcome.get("profit_loss", 0)}, "final_value": {outcome.get("final_value", 10000)}}},
  "key_differences": ["list of 2-3 key behavioral differences"],
  "what_to_practice": ["specific skills the user should develop"]
}}
</output_format>

<constraints>
- The pro's decisions must be realistic for the scenario — no hindsight bias. The pro only knows what was available at each timestamp.
- Pro outcomes should be plausible, not guaranteed profits. Pros manage risk, they don't predict perfectly.
- Each pro_decision must contrast with a specific user decision at a similar timestamp.
- Skills demonstrated should be concrete and learnable, not vague.
</constraints>"""

    def _build_coaching_prompt(self, simulation, decisions, scenario, profile) -> str:
        outcome = simulation.final_outcome or {}
        persona, persona_instruction = self._determine_persona(profile)
        profile_str = ""
        if profile:
            profile_str = f"""
USER'S HISTORICAL BEHAVIOR PROFILE:
Strengths: {profile.get("strengths", [])}
Weaknesses: {profile.get("weaknesses", [])}
Bias patterns: {profile.get("bias_patterns", {})}
Decision style: {profile.get("decision_style", "unknown")}
Total simulations analyzed: {profile.get("total_simulations_analyzed", 0)}
"""
        return f"""<role>
You are a behavioral coach for an investor-in-training.
{persona_instruction}
</role>

<context>
<scenario>
{_scenario_summary(scenario)}
</scenario>

<decision_trace>
{_decision_trace(decisions)}
</decision_trace>

<outcome>
Profit/Loss: ${outcome.get("profit_loss", 0):.2f}
</outcome>
{profile_str}
</context>

<task>
Write a personalized coaching message (3-5 sentences). Address the user directly as "you".
</task>

<output_format>
Return valid JSON:
{{
  "coaching_message": "your personalized coaching message here",
  "persona": "{persona}"
}}
</output_format>

<constraints>
- If the user has a behavior profile, reference specific improvements or recurring patterns.
- Match the {persona} persona tone consistently.
- Give ONE specific thing they should try in their next simulation.
- If they improved on a known weakness, acknowledge the progress.
- Keep it conversational and human. No bullet points, no jargon.
</constraints>"""

    def _build_profile_prompt(self, simulation, decisions, scenario, existing_profile) -> str:
        outcome = simulation.final_outcome or {}
        existing_str = json.dumps(existing_profile or {}, indent=2)
        return f"""<role>
You are maintaining a compressed behavioral profile for an investor-in-training.
</role>

<context>
<existing_profile>
{existing_str}
</existing_profile>

<new_simulation>
Scenario: {scenario.name} (Category: {scenario.category})
Decisions: {_decision_trace(decisions)}
Outcome: ${outcome.get("profit_loss", 0):.2f}
Process Quality: {simulation.process_quality_score or 'N/A'}
</new_simulation>
</context>

<task>
Update the behavior profile by integrating observations from this new simulation. Compress the information — don't simply append.
</task>

<output_format>
Return valid JSON:
{{
  "strengths": ["list of identified strengths"],
  "weaknesses": ["list of identified weaknesses"],
  "bias_patterns": {{"pattern_name": 0.0-1.0}},
  "decision_style": "reactive" | "analytical" | "balanced",
  "stress_response": "impulsive" | "cautious" | "steady",
  "improvement_notes": "what changed from the previous profile, if anything"
}}
</output_format>

<constraints>
- Bias scores should be running averages, not just from this session.
- If this is the first simulation, establish baseline scores.
- If a previous weakness improved, lower its score but note the improvement.
- Be conservative with changes — one simulation shouldn't drastically alter the profile.
</constraints>"""

    # ── PROMPT BUILDERS (new methods) ──────────────────────────────────

    def _build_bias_classifier_prompt(self, simulation, decisions, scenario, decision_features) -> str:
        outcome = simulation.final_outcome or {}

        # Format per-decision feature tables
        feature_rows = []
        for i, (d, feats) in enumerate(zip(decisions, decision_features)):
            feat_str = ", ".join(f"{k}={v}" for k, v in feats.items())
            feature_rows.append(f"  Decision #{i+1} (t={d.simulation_time}s, {d.decision_type}): {feat_str}")
        features_block = "\n".join(feature_rows)

        return f"""<role>
You are a quantitative behavioral finance researcher specializing in cognitive bias detection.
You combine qualitative behavioral analysis with quantitative feature engineering to produce
rigorous, evidence-backed bias classifications. You are precise about uncertainty —
you never overstate confidence in a detection without strong evidence from both the
qualitative trace AND the numerical features.
</role>

<context>
<scenario>
{_scenario_summary(scenario)}
</scenario>

<decision_trace>
{_decision_trace(decisions)}
</decision_trace>

<outcome>
Profit/Loss: ${outcome.get("profit_loss", 0):.2f}
Final Portfolio Value: ${outcome.get("final_value", 10000):.2f}
Process Quality Score (heuristic): {simulation.process_quality_score or 'N/A'}
</outcome>

<extracted_numerical_features>
Each decision has been processed through a feature extraction pipeline. These features
capture timing, price context, position sizing, information usage, and behavioral signals.
{features_block}
</extracted_numerical_features>

<feature_definitions>
- time_spent_seconds: deliberation time before executing the decision
- time_pct_in_sim: position in simulation timeline (0=start, 1=end)
- is_first_decision: 1.0 if this was the first action taken
- price_change_10s: price movement in the 10 seconds before the decision (% change)
- price_change_30s: price movement in the 30 seconds before the decision (% change)
- price_vs_initial: current price vs starting price (% change)
- future_change_30s: what happened to price 30s AFTER the decision (% change, for outcome evaluation)
- info_panels_viewed: number of information panels checked before deciding
- confidence_normalized: self-reported confidence (0-1 scale, from 1-5 rating)
- position_pct_of_capital: trade size as fraction of total capital
- is_buy/is_sell/is_hold: decision type encoding
- local_volatility: price volatility in the 20s window before the decision
- time_since_last_decision: seconds since previous decision
- direction_consistency: fraction of recent decisions in the same direction
</feature_definitions>
</context>

<task>
Perform a rigorous bias classification by analyzing BOTH the qualitative decision trace
AND the quantitative features. For each decision and in aggregate:

1. Score each of the 8 biases (0-1) based on converging evidence from BOTH data sources.
   A bias score should only be high (>0.6) when BOTH qualitative and quantitative evidence agree.
   If only one source suggests the bias, cap the score at 0.4-0.5.

2. For aggregate scores, weight decisions by their trade significance (buys/sells weighted
   higher than holds).

3. Identify the top biases with detailed reasoning that cites SPECIFIC features and
   SPECIFIC decision trace elements.

4. Rank which numerical features were most diagnostic for each detected bias and explain
   WHY each feature matters for that bias.

5. Provide your analytical reasoning: HOW did you combine the qualitative and quantitative
   signals? Where did they agree? Where did they disagree?
</task>

<output_format>
Return valid JSON matching this exact schema:
{{
  "per_decision_scores": [
    {{
      "decision_index": 0,
      "decision_type": "buy",
      "timestamp_seconds": 15,
      "bias_scores": {{
        "fomo": 0.0,
        "loss_aversion": 0.0,
        "anchoring": 0.0,
        "overconfidence": 0.0,
        "impulsivity": 0.0,
        "recency_bias": 0.0,
        "confirmation_bias": 0.0,
        "social_proof_reliance": 0.0
      }},
      "primary_bias": "fomo or none",
      "qualitative_evidence": "what the decision trace reveals",
      "quantitative_evidence": "what the numerical features reveal"
    }}
  ],
  "aggregate_scores": {{
    "fomo": 0.0,
    "loss_aversion": 0.0,
    "anchoring": 0.0,
    "overconfidence": 0.0,
    "impulsivity": 0.0,
    "recency_bias": 0.0,
    "confirmation_bias": 0.0,
    "social_proof_reliance": 0.0
  }},
  "top_biases": [
    {{
      "bias": "fomo",
      "score": 0.72,
      "rank": 1,
      "reasoning": "Detailed explanation citing specific decisions and features",
      "qualitative_strength": "strong/moderate/weak",
      "quantitative_strength": "strong/moderate/weak"
    }}
  ],
  "feature_importance": {{
    "fomo": [
      {{"feature": "price_change_10s", "importance": 0.85, "explanation": "why this feature matters for FOMO"}}
    ]
  }},
  "gemini_reasoning": "A 2-3 paragraph explanation of HOW you analyzed the data, where qualitative and quantitative signals converged or diverged, and what analytical approach you used"
}}
</output_format>

<constraints>
- Every bias score MUST be justified by citing specific evidence from BOTH the decision trace AND the numerical features.
- Do NOT inflate scores. If evidence is ambiguous, keep scores moderate (0.3-0.5).
- If a bias has no supporting evidence from either source, score it at 0.0 — do not assign minimum scores to pad the analysis.
- The feature_importance section must explain the CAUSAL reasoning: why would this feature indicate this bias, not just correlation.
- top_biases should only include biases with aggregate scores above 0.15. If no biases are above 0.15, return an empty list and note the balanced approach in gemini_reasoning.
- per_decision_scores must have exactly one entry per decision, in order.
- primary_bias should be "none" if no bias exceeds 0.3 for that decision.
</constraints>"""

    def _build_confidence_calibration_prompt(
        self, simulation, decisions, scenario, patterns_detected, evidence_signals
    ) -> str:
        outcome = simulation.final_outcome or {}

        # Format patterns Gemini originally detected
        patterns_block = ""
        for p in patterns_detected:
            evidence_list = p.get("evidence", [])
            evidence_str = "; ".join(evidence_list) if evidence_list else "no evidence cited"
            patterns_block += (
                f"  - {p.get('pattern_name', '?')}: confidence={p.get('confidence', 0):.2f}, "
                f"evidence=[{evidence_str}]\n"
            )

        # Format observable evidence signals
        evidence_block = ""
        for pattern_name, signals in evidence_signals.items():
            signal_lines = []
            for sig in signals:
                matched = "MATCHED" if sig.get("matched") else "NOT matched"
                detail = sig.get("detail", "")
                signal_lines.append(
                    f"    [{matched}] {sig.get('desc', sig.get('check', '?'))} "
                    f"(weight={sig.get('weight', 0):.2f}){': ' + detail if detail else ''}"
                )
            evidence_block += f"  {pattern_name}:\n" + "\n".join(signal_lines) + "\n"

        return f"""<role>
You are a metacognitive calibration system for an AI behavioral analyst.
Your job is to SELF-EVALUATE the quality of pattern detections by checking them
against independently gathered observable evidence. You are ruthlessly honest about
epistemic uncertainty — you would rather abstain than assert a weakly-supported pattern.
</role>

<context>
<scenario>
{_scenario_summary(scenario)}
</scenario>

<decision_trace>
{_decision_trace(decisions)}
</decision_trace>

<outcome>
Profit/Loss: ${outcome.get("profit_loss", 0):.2f}
Final Portfolio Value: ${outcome.get("final_value", 10000):.2f}
</outcome>

<original_pattern_detections>
These patterns were detected by the AI during reflection analysis:
{patterns_block}
</original_pattern_detections>

<observable_evidence_signals>
These evidence checks were run independently against the raw simulation data.
Each signal tests for a specific observable behavior associated with a bias:
{evidence_block}
</observable_evidence_signals>
</context>

<task>
For EACH pattern detected in the original analysis, perform a confidence calibration:

1. COMPARE the original detection confidence against the observable evidence signals.
   Does the evidence actually support the pattern at the claimed confidence level?

2. BLEND the original AI confidence with the evidence strength:
   - If evidence strongly supports → boost confidence
   - If evidence is mixed → moderate confidence toward 0.5
   - If evidence contradicts → sharply reduce confidence
   - If evidence is insufficient (few signals matched) → flag for abstention

3. Assign a calibrated confidence label:
   - "High" (0.7+): Multiple independent evidence signals converge
   - "Medium" (0.4-0.7): Some evidence supports, some ambiguous
   - "Low" (0.2-0.4): Weak evidence, pattern is plausible but not well-supported
   - "Insufficient" (<0.2): Not enough evidence to make any claim — ABSTAIN

4. For each pattern, identify which evidence signals are MOST persuasive
   (for or against the detection), and explain why.

5. Flag patterns that should be RETRACTED — where the evidence actively contradicts
   the original detection, not just fails to support it.
</task>

<output_format>
Return valid JSON matching this exact schema:
{{
  "calibrated_patterns": [
    {{
      "pattern_name": "fomo",
      "original_confidence": 0.75,
      "evidence_strength": 0.62,
      "calibrated_confidence": 0.68,
      "confidence_label": "Medium",
      "supporting_signals": [
        {{"signal": "rapid_buy_after_spike", "persuasiveness": "high", "explanation": "why this signal is convincing"}}
      ],
      "contradicting_signals": [
        {{"signal": "info_viewed", "persuasiveness": "medium", "explanation": "why this signal argues against"}}
      ],
      "calibration_reasoning": "Detailed explanation of how original confidence was adjusted",
      "should_retract": false
    }}
  ],
  "abstained_patterns": [
    {{
      "pattern_name": "anchoring",
      "original_confidence": 0.45,
      "reason": "Why evidence is insufficient to support or deny this detection",
      "missing_evidence": ["what additional data would be needed"]
    }}
  ],
  "overall_evidence_quality": "strong",
  "calibration_summary": "A 2-3 sentence summary of the calibration: how many patterns were well-supported, how many were weak, and what the user should trust in the original analysis"
}}
</output_format>

<constraints>
- You MUST evaluate every pattern from the original_pattern_detections. Do not skip any.
- confidence_label MUST match the calibrated_confidence range (High/Medium/Low/Insufficient).
- If fewer than 2 evidence signals matched for a pattern, it MUST go into abstained_patterns, NOT calibrated_patterns.
- should_retract should be true ONLY when evidence actively contradicts the pattern, not merely when evidence is weak.
- overall_evidence_quality: "strong" if >60% of patterns are High/Medium, "moderate" if 40-60%, "weak" if <40%.
- Be specific in signal explanations — reference actual decision numbers, timestamps, and observable behaviors.
- The calibration_summary must be honest and actionable. If most detections are weakly supported, say so.
</constraints>"""

    def _build_behavior_history_prompt(self, history_summary, existing_profile) -> str:
        # Format the simulation history
        history_rows = []
        for i, sim in enumerate(history_summary):
            top_biases_str = ", ".join(
                f"{b.get('bias', '?')}({b.get('score', 0):.2f})"
                for b in sim.get("top_biases", [])[:3]
            ) or "none detected"
            history_rows.append(
                f"  Sim #{i+1}: {sim.get('scenario_name', '?')} "
                f"[{sim.get('category', '?')}, diff={sim.get('difficulty', '?')}/5] | "
                f"P/L=${sim.get('profit_loss', 0):.2f} | "
                f"Process={sim.get('process_quality_score', 'N/A')} | "
                f"Decisions={sim.get('decisions_count', 0)} | "
                f"Top biases: {top_biases_str} | "
                f"Date: {sim.get('timestamp', 'unknown')}"
            )
        history_block = "\n".join(history_rows)

        profile_str = json.dumps(existing_profile or {}, indent=2)

        return f"""<role>
You are a longitudinal behavioral data analyst specializing in learning trajectories
and behavioral change detection. You analyze patterns across an individual's ENTIRE
history, not just single sessions. You are trained to distinguish genuine behavioral
change from noise, and to identify emerging trends before they become obvious.
</role>

<context>
<simulation_history>
Complete history of the user's simulations (ordered chronologically):
{history_block}
</simulation_history>

<existing_behavior_profile>
{profile_str}
</existing_behavior_profile>

<data_summary>
Total simulations: {len(history_summary)}
Date range: {history_summary[0].get('timestamp', 'unknown') if history_summary else 'N/A'} to {history_summary[-1].get('timestamp', 'unknown') if history_summary else 'N/A'}
Average P/L: ${sum(s.get('profit_loss', 0) for s in history_summary) / max(len(history_summary), 1):.2f}
Average process quality: {sum(s.get('process_quality_score', 50) or 50 for s in history_summary) / max(len(history_summary), 1):.0f}
</data_summary>
</context>

<task>
Analyze the user's COMPLETE behavioral history to discover patterns that only emerge
from longitudinal data. Do NOT simply reformat the pre-computed profile — you must
DISCOVER insights from the raw history data.

1. EMERGING PATTERNS: Identify behavioral biases that appear across multiple simulations.
   For each, cite specific simulation numbers and how the pattern manifests across sessions.
   Distinguish between persistent biases (present in >50% of sims) and situational biases
   (appear only in specific scenario types or difficulty levels).

2. LEARNING TRAJECTORY: Analyze whether the user is improving over time.
   - Track process quality scores across simulations — is the trend up, down, or flat?
   - Track bias scores across simulations — are specific biases diminishing?
   - Identify "breakthrough moments" where a significant improvement occurred.
   - Identify "regression points" where the user reverted to old patterns.

3. STRENGTHS AND WEAKNESSES: Discover from the aggregate data (not just profile).
   - Strengths: What does this user consistently do well?
   - Weaknesses: Where do they consistently struggle?
   - Are there scenario types or difficulty levels where they perform differently?

4. DECISION STYLE CLASSIFICATION: Based on aggregate behavior, classify as:
   - "reactive": Decisions driven by market events, short deliberation, sentiment-following
   - "analytical": Longer deliberation, info-heavy, systematic approach
   - "balanced": Mix of both, adapts approach to situation

5. STRESS RESPONSE PATTERN: How does the user behave under pressure?
   - Does performance degrade at higher difficulty levels?
   - Do biases intensify when outcomes are negative?

6. SPECIFIC IMPROVEMENT RECOMMENDATIONS: Based on the trajectory analysis,
   what specific actions would produce the most improvement?
</task>

<output_format>
Return valid JSON matching this exact schema:
{{
  "emerging_patterns": [
    {{
      "pattern_name": "fomo",
      "persistence": "persistent" | "situational",
      "frequency": 0.65,
      "trend": "increasing" | "stable" | "decreasing",
      "evidence_across_sims": ["Sim #1: bought during hype...", "Sim #3: similar pattern..."],
      "situational_triggers": ["appears mainly in crypto scenarios", "intensifies at difficulty 4+"]
    }}
  ],
  "learning_trajectory": {{
    "overall_direction": "improving" | "stagnating" | "regressing",
    "process_quality_trend": [
      {{"sim_number": 1, "score": 45, "note": "baseline"}}
    ],
    "breakthrough_moments": [
      {{"sim_number": 3, "description": "First time user checked all info panels before trading"}}
    ],
    "regression_points": [
      {{"sim_number": 5, "description": "Reverted to impulsive buying after a loss in sim #4"}}
    ],
    "trajectory_summary": "2-3 sentences on the user's learning arc"
  }},
  "strengths": ["list of specific strengths discovered from the data"],
  "weaknesses": ["list of specific weaknesses discovered from the data"],
  "scenario_performance": {{
    "best_category": "category where user performs best",
    "worst_category": "category where user struggles most",
    "difficulty_impact": "how difficulty affects performance"
  }},
  "decision_style": "reactive" | "analytical" | "balanced",
  "decision_style_evidence": "Why this classification, citing specific data",
  "stress_response": "impulsive" | "cautious" | "steady",
  "stress_response_evidence": "Why this classification, citing specific data",
  "improvement_recommendations": [
    {{
      "priority": 1,
      "area": "specific area to improve",
      "recommendation": "detailed, actionable recommendation",
      "expected_impact": "what improvement to expect",
      "based_on": "which data points led to this recommendation"
    }}
  ],
  "history_analysis_reasoning": "2-3 paragraphs explaining your analytical approach, key discoveries, and confidence in findings"
}}
</output_format>

<constraints>
- All claims MUST cite specific simulation numbers from the history. Do not make generic statements.
- If there are fewer than 3 simulations, note that longitudinal analysis is limited and lower your confidence in trend claims.
- emerging_patterns frequency must be calculable: number of sims showing the pattern / total sims.
- process_quality_trend must have one entry per simulation, in chronological order.
- improvement_recommendations must be prioritized (1 = highest priority) and limited to 3-5 items.
- Do NOT simply restate the existing_behavior_profile. You must add NEW insights from the history analysis.
- If the user shows no clear biases or patterns, say so honestly rather than fabricating patterns.
- decision_style and stress_response must be supported by specific evidence, not assumed from the profile.
</constraints>"""

    # ── HEURISTIC FALLBACKS FOR NEW FEATURES ────────────────────────────

    def _heuristic_bias_classification(self, simulation, decisions, scenario, decision_features):
        """Rule-based fallback for bias classification — delegates to bias_classifier module."""
        from services.bias_classifier import classify_simulation_biases
        from services.simulation_engine import SimulationEngine

        engine = SimulationEngine(scenario)
        gemini_patterns = None
        if simulation.gemini_analysis:
            gemini_patterns = simulation.gemini_analysis.get("patterns_detected", [])

        result = classify_simulation_biases(
            decisions=decisions,
            price_timeline=engine.price_timeline,
            time_limit=engine.time_limit,
            initial_price=engine.initial_price,
            initial_balance=engine.initial_balance,
            gemini_patterns=gemini_patterns,
        )

        # Reshape to match Gemini output schema
        per_decision_scores = []
        for entry in result.get("per_decision", []):
            per_decision_scores.append({
                "decision_index": entry["decision_index"],
                "decision_type": entry["decision_type"],
                "timestamp_seconds": entry["timestamp"],
                "bias_scores": entry["bias_scores"],
                "primary_bias": entry["primary_bias"],
                "qualitative_evidence": f"Decision at t={entry['timestamp']}s",
                "quantitative_evidence": f"Features: {list(entry['features'].keys())[:5]}",
            })

        top_biases = []
        for tb in result.get("top_biases", []):
            top_biases.append({
                "bias": tb["bias"],
                "score": tb["score"],
                "rank": tb["rank"],
                "reasoning": f"Rule-based score of {tb['score']:.2f} from weighted feature analysis",
                "qualitative_strength": "moderate",
                "quantitative_strength": "moderate",
            })

        feature_importance = {}
        for bias, feats in result.get("feature_importance", {}).items():
            feature_importance[bias] = [
                {"feature": f, "importance": imp, "explanation": f"{f} correlates with {bias}"}
                for f, imp in feats[:5]
            ]

        return {
            "simulation_id": str(simulation.id),
            "per_decision_scores": per_decision_scores,
            "aggregate_scores": result.get("aggregate_scores", {}),
            "top_biases": top_biases,
            "feature_importance": feature_importance,
            "gemini_reasoning": "Analysis performed by rule-based classifier (mock/fallback mode). "
                "Scores are computed from weighted feature thresholds rather than AI reasoning.",
        }

    def _heuristic_confidence_calibration(
        self, simulation, decisions, scenario, patterns_detected, evidence_signals
    ):
        """Rule-based fallback for confidence calibration — delegates to confidence_calibrator."""
        from services.simulation_engine import SimulationEngine
        from services.confidence_calibrator import calibrate_all_patterns

        engine = SimulationEngine(scenario)
        result = calibrate_all_patterns(
            patterns=patterns_detected,
            decisions=decisions,
            price_timeline=engine.price_timeline,
            events=engine.events,
            time_limit=engine.time_limit,
            initial_price=engine.initial_price,
            initial_balance=engine.initial_balance,
        )

        # Reshape calibrated patterns to match Gemini schema
        calibrated = []
        for p in result.get("calibrated_patterns", []):
            calibrated.append({
                "pattern_name": p.get("pattern_name", ""),
                "original_confidence": p.get("gemini_confidence", 0),
                "evidence_strength": p.get("evidence_score", 0),
                "calibrated_confidence": p.get("calibrated_confidence", 0),
                "confidence_label": p.get("confidence_label", "Low"),
                "supporting_signals": [
                    {"signal": s.get("check", ""), "persuasiveness": "medium",
                     "explanation": s.get("detail", "")}
                    for s in p.get("evidence_details", []) if s.get("matched")
                ],
                "contradicting_signals": [
                    {"signal": s.get("check", ""), "persuasiveness": "medium",
                     "explanation": s.get("detail", "")}
                    for s in p.get("evidence_details", []) if not s.get("matched")
                ],
                "calibration_reasoning": f"Evidence score {p.get('evidence_score', 0):.2f} "
                    f"blended with AI confidence {p.get('gemini_confidence', 0):.2f}",
                "should_retract": False,
            })

        abstained = []
        for p in result.get("abstained_patterns", []):
            abstained.append({
                "pattern_name": p.get("pattern_name", ""),
                "original_confidence": p.get("original_confidence", 0),
                "reason": p.get("reason", "Insufficient evidence"),
                "missing_evidence": ["More behavioral data needed"],
            })

        return {
            "simulation_id": str(simulation.id),
            "calibrated_patterns": calibrated,
            "abstained_patterns": abstained,
            "overall_evidence_quality": result.get("overall_evidence_quality", "moderate"),
            "calibration_summary": result.get("summary", "Rule-based calibration (mock/fallback mode)."),
        }

    def _heuristic_behavior_history(self, history_summary, existing_profile):
        """Deterministic fallback for behavior history analysis."""
        if not history_summary:
            return {
                "emerging_patterns": [],
                "learning_trajectory": {
                    "overall_direction": "stagnating",
                    "process_quality_trend": [],
                    "breakthrough_moments": [],
                    "regression_points": [],
                    "trajectory_summary": "No simulation history available for analysis.",
                },
                "strengths": [],
                "weaknesses": [],
                "scenario_performance": {
                    "best_category": "unknown",
                    "worst_category": "unknown",
                    "difficulty_impact": "Insufficient data",
                },
                "decision_style": "unknown",
                "decision_style_evidence": "No data",
                "stress_response": "unknown",
                "stress_response_evidence": "No data",
                "improvement_recommendations": [],
                "history_analysis_reasoning": "No history data available for analysis.",
            }

        # Compute aggregate stats
        scores = [s.get("process_quality_score", 50) or 50 for s in history_summary]
        pnls = [s.get("profit_loss", 0) for s in history_summary]
        win_rate = sum(1 for p in pnls if p > 0) / max(len(pnls), 1)

        # Detect trend in process quality
        first_half = scores[:len(scores)//2] if len(scores) >= 4 else scores[:1]
        second_half = scores[len(scores)//2:] if len(scores) >= 4 else scores[-1:]
        avg_first = sum(first_half) / max(len(first_half), 1)
        avg_second = sum(second_half) / max(len(second_half), 1)

        if avg_second > avg_first + 10:
            direction = "improving"
        elif avg_second < avg_first - 10:
            direction = "regressing"
        else:
            direction = "stagnating"

        # Process quality trend
        pq_trend = [
            {"sim_number": i + 1, "score": s, "note": "baseline" if i == 0 else ""}
            for i, s in enumerate(scores)
        ]

        # Find breakthrough / regression
        breakthroughs = []
        regressions = []
        for i in range(1, len(scores)):
            if scores[i] - scores[i-1] >= 15:
                breakthroughs.append({
                    "sim_number": i + 1,
                    "description": f"Process quality jumped from {scores[i-1]} to {scores[i]}",
                })
            elif scores[i] - scores[i-1] <= -15:
                regressions.append({
                    "sim_number": i + 1,
                    "description": f"Process quality dropped from {scores[i-1]} to {scores[i]}",
                })

        # Category performance
        cat_scores: dict[str, list] = {}
        for s in history_summary:
            cat = s.get("category", "unknown")
            cat_scores.setdefault(cat, []).append(s.get("process_quality_score", 50) or 50)
        cat_avgs = {c: sum(v)/len(v) for c, v in cat_scores.items()}
        best_cat = max(cat_avgs, key=cat_avgs.get) if cat_avgs else "unknown"
        worst_cat = min(cat_avgs, key=cat_avgs.get) if cat_avgs else "unknown"

        # Strengths / weaknesses from scores
        strengths = []
        weaknesses = []
        if avg_second >= 70:
            strengths.append("consistently_high_process_quality")
        if win_rate >= 0.5:
            strengths.append("positive_win_rate")
        if direction == "improving":
            strengths.append("learning_and_improving")
        if avg_second < 40:
            weaknesses.append("low_process_quality")
        if win_rate < 0.3:
            weaknesses.append("poor_outcomes")

        # Decision style from profile or heuristic
        style = "balanced"
        if existing_profile:
            style = existing_profile.get("decision_style", "balanced")

        return {
            "emerging_patterns": [],
            "learning_trajectory": {
                "overall_direction": direction,
                "process_quality_trend": pq_trend,
                "breakthrough_moments": breakthroughs[:3],
                "regression_points": regressions[:3],
                "trajectory_summary": (
                    f"Over {len(history_summary)} simulations, the user shows "
                    f"{'improvement' if direction == 'improving' else 'mixed'} trajectory. "
                    f"Average process quality: {sum(scores)/len(scores):.0f}/100. "
                    f"Win rate: {win_rate:.0%}."
                ),
            },
            "strengths": strengths,
            "weaknesses": weaknesses,
            "scenario_performance": {
                "best_category": best_cat,
                "worst_category": worst_cat,
                "difficulty_impact": "Heuristic analysis — run with Gemini for detailed difficulty impact analysis",
            },
            "decision_style": style,
            "decision_style_evidence": "Heuristic analysis based on aggregate stats (mock/fallback mode)",
            "stress_response": existing_profile.get("stress_response", "steady") if existing_profile else "steady",
            "stress_response_evidence": "Heuristic analysis (mock/fallback mode)",
            "improvement_recommendations": [
                {
                    "priority": 1,
                    "area": weaknesses[0] if weaknesses else "process_quality",
                    "recommendation": "Focus on improving your weakest area in the next simulation",
                    "expected_impact": "Gradual improvement in process scores",
                    "based_on": f"Analysis of {len(history_summary)} simulations",
                },
            ],
            "history_analysis_reasoning": (
                f"Heuristic analysis based on {len(history_summary)} simulations. "
                f"Process quality trend: {direction}. Average score: {sum(scores)/len(scores):.0f}. "
                f"Win rate: {win_rate:.0%}. Run with Gemini enabled for deeper behavioral insights."
            ),
        }

    # ── HEURISTIC FALLBACKS (used when mock=True or Gemini fails) ────────

    def _heuristic_analyze(self, simulation, decisions, scenario) -> ReflectionResponse:
        """Deterministic heuristic fallback — same logic as original mock."""
        outcome = simulation.final_outcome or {}
        profit_loss = outcome.get("profit_loss", 0)

        if profit_loss > 100:
            outcome_type, outcome_summary = "profit", f"+${profit_loss:.2f}"
        elif profit_loss < -100:
            outcome_type, outcome_summary = "loss", f"-${abs(profit_loss):.2f}"
        else:
            outcome_type, outcome_summary = "break_even", f"${profit_loss:.2f}"

        process_quality = self._heuristic_process_quality(decisions)
        patterns = self._heuristic_patterns(decisions)
        luck, skill = self._heuristic_luck_skill(simulation)
        insights = self._heuristic_insights(patterns, process_quality)
        coaching = self._heuristic_coaching(simulation, decisions, scenario, None)

        return ReflectionResponse(
            simulation_id=simulation.id,
            outcome_summary=outcome_summary,
            outcome_type=outcome_type,
            process_quality=process_quality,
            patterns_detected=patterns,
            luck_factor=luck,
            skill_factor=skill,
            luck_skill_explanation=self._luck_skill_text(luck, outcome_type),
            counterfactuals=[],
            insights=insights,
            key_takeaway=self._heuristic_takeaway(outcome_type, process_quality.score, patterns),
            coaching_message=coaching,
        )

    def _heuristic_process_quality(self, decisions: list[Decision]) -> ProcessQuality:
        factors: dict[str, float] = {}
        if decisions:
            t0 = decisions[0].simulation_time
            factors["timing"] = 0.9 if t0 >= 20 else (0.7 if t0 >= 10 else 0.4)
        else:
            factors["timing"] = 0.5

        total_info = sum(len(d.info_viewed or []) for d in decisions)
        n = max(len(decisions), 1)
        factors["information_usage"] = 0.9 if total_info >= n * 3 else (0.6 if total_info >= n else 0.3)

        big = sum(1 for d in decisions if d.decision_type == "buy" and (d.amount or 0) > 1000)
        factors["risk_sizing"] = 0.8 if big == 0 else (0.6 if big <= 1 else 0.3)

        rapid = sum(
            1 for i in range(1, len(decisions))
            if decisions[i].simulation_time - decisions[i - 1].simulation_time < 10
        )
        factors["emotional_control"] = 0.85 if rapid == 0 else (0.6 if rapid <= 2 else 0.35)

        score = sum(factors.values()) / len(factors) * 100
        if score >= 75:
            summary = "Strong decision-making process with good information usage and emotional control."
        elif score >= 50:
            summary = "Decent process but room for improvement in information gathering or timing."
        else:
            summary = "Process shows signs of impulsivity or insufficient analysis."

        return ProcessQuality(score=score, factors=factors, summary=summary)

    def _heuristic_patterns(self, decisions: list[Decision]) -> list[PatternDetection]:
        patterns: list[PatternDetection] = []

        buys_bullish = sum(
            1 for d in decisions
            if d.decision_type == "buy"
            and _get_market_state(d).get("available_info", {}).get("market_sentiment") == "bullish"
        )
        if buys_bullish >= 2:
            patterns.append(PatternDetection(
                pattern_name="fomo",
                confidence=min(0.9, 0.4 + buys_bullish * 0.2),
                evidence=[
                    f"Made {buys_bullish} buy decisions during bullish sentiment",
                    "Timing suggests reaction to rising prices rather than analysis",
                ],
                description="Fear of Missing Out — tendency to buy when prices are already rising",
            ))

        quick = sum(1 for d in decisions if (d.time_spent_seconds or 10) < 5)
        if quick >= len(decisions) * 0.5 and len(decisions) > 1:
            patterns.append(PatternDetection(
                pattern_name="impulsivity",
                confidence=0.7,
                evidence=[
                    f"{quick}/{len(decisions)} decisions made in under 5 seconds",
                    "Limited information viewed before decisions",
                ],
                description="Tendency to make quick decisions without thorough analysis",
            ))

        sells_bearish = sum(
            1 for d in decisions
            if d.decision_type == "sell"
            and _get_market_state(d).get("available_info", {}).get("market_sentiment") == "bearish"
        )
        if sells_bearish >= 1:
            patterns.append(PatternDetection(
                pattern_name="loss_aversion",
                confidence=0.6,
                evidence=[
                    f"Sold during {sells_bearish} bearish condition(s)",
                    "Pattern suggests selling to avoid potential losses",
                ],
                description="Strong preference to avoid losses over equivalent gains",
            ))

        high_conf = sum(1 for d in decisions if (d.confidence_level or 3) >= 4)
        if high_conf >= 3:
            patterns.append(PatternDetection(
                pattern_name="overconfidence",
                confidence=0.5,
                evidence=[
                    f"{high_conf} decisions made with high confidence",
                    "Consider whether confidence matched analysis depth",
                ],
                description="Excessive confidence in predictions without proportional analysis",
            ))

        if not patterns:
            patterns.append(PatternDetection(
                pattern_name="balanced_approach",
                confidence=0.6,
                evidence=["No strong negative patterns detected", "Decision timing and information usage were reasonable"],
                description="Generally balanced decision-making approach",
            ))
        return patterns

    def _heuristic_luck_skill(self, simulation: Simulation) -> tuple[float, float]:
        outcome = simulation.final_outcome or {}
        pl = outcome.get("profit_loss", 0)
        ps = simulation.process_quality_score or 50

        if pl > 0:
            if ps >= 70: return 0.3, 0.7
            if ps >= 50: return 0.5, 0.5
            return 0.7, 0.3
        else:
            if ps >= 70: return 0.4, 0.6
            if ps >= 50: return 0.6, 0.4
            return 0.8, 0.2

    def _luck_skill_text(self, luck: float, outcome_type: str) -> str:
        if luck >= 0.7:
            if outcome_type == "profit":
                return "Your profit was largely due to favorable conditions. Similar decisions could easily result in losses."
            return "Your loss was partially due to unfavorable conditions, but process improvements would help."
        if luck >= 0.5:
            return "The outcome reflects a mix of decisions and conditions. Focus on improving your process."
        if outcome_type == "profit":
            return "Your profit reflects solid decision-making. Your analytical approach contributed significantly."
        return "Despite the loss, your process was sound. Continue refining and results will likely improve."

    def _heuristic_insights(self, patterns, pq) -> list[ActionableInsight]:
        insights: list[ActionableInsight] = []
        lowest = min(pq.factors.items(), key=lambda x: x[1])
        mapping = {
            "timing": ("Improve Decision Timing", "Take more time before your first decision.", "impulsivity", "card_patience_01"),
            "information_usage": ("Use More Information", "Review all info panels before deciding.", "information_neglect", "card_analysis_01"),
            "risk_sizing": ("Manage Position Sizes", "Start smaller and scale based on conviction.", "overconfidence", "card_risk_01"),
            "emotional_control": ("Pause Before Reacting", "Rapid decisions often indicate emotion over analysis.", "impulsivity", "card_emotions_01"),
        }
        if lowest[0] in mapping:
            t, d, p, c = mapping[lowest[0]]
            insights.append(ActionableInsight(title=t, description=d, related_pattern=p, recommended_card_id=c))
        for pat in patterns[:2]:
            if pat.pattern_name == "fomo":
                insights.append(ActionableInsight(title="Recognize FOMO Triggers", description="You bought into rising prices. Identify when hype drives your decisions.", related_pattern="fomo", recommended_card_id="card_fomo_01"))
            elif pat.pattern_name == "loss_aversion":
                insights.append(ActionableInsight(title="Reframe Loss Perspective", description="Selling during downturns locks in losses. Consider if fundamentals changed.", related_pattern="loss_aversion", recommended_card_id="card_loss_01"))
        return insights[:3]

    def _heuristic_takeaway(self, outcome_type, score, patterns) -> str:
        if outcome_type == "profit" and score < 50:
            return "You profited, but your process needs work. Don't let luck reinforce poor habits."
        if outcome_type == "loss" and score >= 70:
            return "Despite the loss, your decision-making was solid. Good process leads to good outcomes over time."
        if outcome_type == "profit" and score >= 70:
            return "Both outcome and process were strong. Keep refining your analytical approach."
        if patterns and patterns[0].pattern_name != "balanced_approach":
            name = patterns[0].pattern_name.replace("_", " ").title()
            return f"Work on addressing {name} — it can significantly impact your long-term results."
        return "Focus on gathering more information before deciding. Analysis quality determines long-term success."

    def _heuristic_counterfactuals(self, simulation, decisions, scenario) -> list[Counterfactual]:
        import random as rng
        outcome = simulation.final_outcome or {}
        fv = outcome.get("final_value", 10000)

        # Use decision trace details to make counterfactuals reactive
        buy_count = sum(1 for d in decisions if d.decision_type == "buy")
        sell_count = sum(1 for d in decisions if d.decision_type == "sell")
        max_amount = max((d.amount or 0 for d in decisions), default=0)

        crash_pct = 0.3 + (max_amount / fv) * 0.2 if fv > 0 else 0.4
        crash_loss = fv * min(crash_pct, 0.6)

        cfs = [
            Counterfactual(
                timeline_name="Market Crash",
                description=f"What if the market crashed right after your {'largest buy' if buy_count > 0 else 'decisions'}?",
                market_changes=f"Sharp {crash_pct*100:.0f}% decline triggered by unexpected regulatory news",
                outcome={"profit_loss": -crash_loss, "final_value": fv - crash_loss},
                lesson=f"With {buy_count} buy decisions totaling exposure, a crash would amplify losses proportionally. Position sizing matters."
            ),
            Counterfactual(
                timeline_name="Extended Rally",
                description="What if momentum continued and prices doubled?",
                market_changes="Sustained buying pressure pushes prices 100% higher over the remaining time",
                outcome={"profit_loss": fv * 0.35, "final_value": fv * 1.35},
                lesson=f"{'Your sells may have been premature.' if sell_count > 0 else 'Holding through volatility can pay off, but requires conviction based on analysis, not hope.'}"
            ),
            Counterfactual(
                timeline_name="Whipsaw Trap",
                description="What if volatile swings shook out weak hands?",
                market_changes="Wild +30%/-35%/+20% swings designed to trigger emotional reactions",
                outcome={"profit_loss": fv * rng.uniform(-0.2, 0.1), "final_value": fv * rng.uniform(0.8, 1.1)},
                lesson="Rapid volatility punishes impulsive reactions. The same decisions yield wildly different results — process consistency is the only edge."
            ),
        ]
        return cfs

    def _heuristic_explain_decisions(self, simulation, decisions, scenario) -> WhyThisDecisionResponse:
        explanations: list[DecisionExplanation] = []
        for i, d in enumerate(decisions):
            bias = None
            explanation = ""
            evidence = []
            severity = "minor"

            # Detect per-decision bias
            ms = _get_market_state(d)
            sentiment = ms.get("available_info", {}).get("market_sentiment", "neutral")
            time_spent = d.time_spent_seconds or 10

            if d.decision_type == "buy" and sentiment == "bullish" and time_spent < 8:
                bias = "fomo"
                explanation = (
                    f"Decision #{i+1} was a buy during bullish sentiment with only "
                    f"{time_spent:.1f}s of deliberation. The combination of rising prices "
                    f"and quick action suggests fear of missing the upward move."
                )
                evidence = [
                    f"Bullish sentiment at time of decision",
                    f"Only {time_spent:.1f}s deliberation time",
                    f"Bought at ${d.price_at_decision or 0:.2f}",
                ]
                severity = "moderate" if time_spent < 5 else "minor"

            elif d.decision_type == "sell" and sentiment == "bearish":
                bias = "loss_aversion"
                explanation = (
                    f"Decision #{i+1} was a sell during bearish conditions. "
                    f"This suggests reacting to the pain of unrealized losses rather than "
                    f"evaluating whether the fundamental scenario had changed."
                )
                evidence = [
                    f"Bearish sentiment at time of decision",
                    f"Sold at ${d.price_at_decision or 0:.2f}",
                ]
                severity = "moderate"

            elif time_spent < 3 and d.decision_type in ("buy", "sell"):
                bias = "impulsivity"
                explanation = (
                    f"Decision #{i+1} was made in {time_spent:.1f}s — barely enough time "
                    f"to read available information. This rapid action suggests emotional "
                    f"reaction rather than analytical assessment."
                )
                evidence = [
                    f"Only {time_spent:.1f}s deliberation",
                    f"Info panels viewed: {len(d.info_viewed or [])}",
                ]
                severity = "significant" if time_spent < 2 else "moderate"

            if bias:
                explanations.append(DecisionExplanation(
                    decision_index=i,
                    decision_type=d.decision_type,
                    timestamp_seconds=d.simulation_time,
                    detected_bias=bias,
                    explanation=explanation,
                    evidence_from_actions=evidence,
                    severity=severity,
                ))

        # Build narrative
        if explanations:
            biases = [e.detected_bias for e in explanations]
            narrative = (
                f"Across {len(decisions)} decisions, {len(explanations)} showed notable "
                f"cognitive patterns. "
            )
            if "fomo" in biases and "impulsivity" in biases:
                narrative += "The combination of FOMO and impulsivity suggests reacting to market excitement rather than forming independent analysis."
            elif "fomo" in biases:
                narrative += "A tendency to chase rising prices emerged, particularly when social or price signals were strong."
            elif "loss_aversion" in biases:
                narrative += "Selling during downturns dominated the pattern, suggesting difficulty tolerating unrealized losses."
            elif "impulsivity" in biases:
                narrative += "Speed of decision-making outpaced information gathering, leaving analysis gaps."
            else:
                narrative += "The patterns suggest areas for targeted improvement in future simulations."
        else:
            narrative = f"Across {len(decisions)} decisions, no strong cognitive biases were detected. Decision-making was generally measured and information-driven."

        return WhyThisDecisionResponse(
            simulation_id=simulation.id,
            explanations=explanations,
            overall_narrative=narrative,
        )

    def _heuristic_pro_comparison(self, simulation, decisions, scenario) -> ProComparisonResponse:
        outcome = simulation.final_outcome or {}
        fv = outcome.get("final_value", 10000)
        pl = outcome.get("profit_loss", 0)
        init_balance = scenario.initial_data.get("your_balance", 10000)

        pro_decisions: list[ProDecision] = []

        # Pro always waits first
        if decisions and decisions[0].simulation_time < 15:
            pro_decisions.append(ProDecision(
                at_timestamp=decisions[0].simulation_time,
                user_action=f"{decisions[0].decision_type} at t={decisions[0].simulation_time}s",
                pro_action="Wait and observe for at least 20 seconds",
                pro_reasoning="A professional gathers information before committing capital. The first seconds establish the scenario's dynamics.",
                outcome_difference="Avoids premature entry that may be driven by initial excitement",
                skill_demonstrated="Patience and information gathering",
            ))

        # Pro uses smaller position sizes
        for d in decisions:
            if d.decision_type == "buy" and (d.amount or 0) > init_balance * 0.1:
                pro_decisions.append(ProDecision(
                    at_timestamp=d.simulation_time,
                    user_action=f"Buy {d.amount} at t={d.simulation_time}s",
                    pro_action=f"Buy {min(d.amount * 0.3, init_balance * 0.05):.0f} (smaller position)",
                    pro_reasoning="Risk no more than 5% of capital per trade. Scale in gradually as conviction builds from data, not from price movement.",
                    outcome_difference="Limits downside while still participating in upside",
                    skill_demonstrated="Position sizing and risk management",
                ))
                break  # One example is enough

        # Pro checks all info panels
        for d in decisions:
            info_count = len(d.info_viewed or [])
            if info_count < 2 and d.decision_type in ("buy", "sell"):
                pro_decisions.append(ProDecision(
                    at_timestamp=d.simulation_time,
                    user_action=f"{d.decision_type} after viewing {info_count} info panels",
                    pro_action="Review all available information sources before acting",
                    pro_reasoning="Professional traders maintain a pre-decision checklist: price trend, news catalyst, social sentiment, and portfolio exposure.",
                    outcome_difference="Better-informed decisions reduce the luck component of outcomes",
                    skill_demonstrated="Systematic information gathering",
                ))
                break

        # Estimate pro outcome (conservative, risk-managed)
        pro_pl = pl * 0.6 if pl > 0 else pl * 0.4  # Pro captures less upside but much less downside
        pro_fv = init_balance + pro_pl

        key_diffs = []
        if decisions and decisions[0].simulation_time < 15:
            key_diffs.append("Pro waits to gather information; user acts immediately")
        key_diffs.append("Pro uses smaller position sizes to manage risk")
        key_diffs.append("Pro reviews all available data before each decision")

        what_to_practice = [
            "Practice waiting 20+ seconds before your first decision",
            "Use position sizes under 5% of total capital per trade",
            "Check all information panels (news, social, chart) before every trade",
        ]

        return ProComparisonResponse(
            simulation_id=simulation.id,
            pro_decisions=pro_decisions,
            pro_final_outcome={"profit_loss": pro_pl, "final_value": pro_fv},
            user_final_outcome={"profit_loss": pl, "final_value": fv},
            key_differences=key_diffs[:3],
            what_to_practice=what_to_practice,
        )

    def _heuristic_coaching(self, simulation, decisions, scenario, profile) -> str:
        outcome = simulation.final_outcome or {}
        pl = outcome.get("profit_loss", 0)
        ps = simulation.process_quality_score or 50

        # Personalize based on profile
        if profile:
            weaknesses = profile.get("weaknesses", [])
            sims = profile.get("total_simulations_analyzed", 0)

            if sims > 0 and weaknesses:
                focus = weaknesses[0].replace("_", " ")
                if ps >= 70:
                    return (
                        f"Nice work on this simulation! Your process score of {ps:.0f} shows improvement. "
                        f"I've noticed you've been working on {focus} — keep that focus going. "
                        f"For your next session, try extending your initial observation period by 10 more seconds "
                        f"before making any decisions."
                    )
                else:
                    return (
                        f"This simulation highlighted some familiar patterns, especially around {focus}. "
                        f"{'You made money, but the process had gaps.' if pl > 0 else 'The outcome was tough, but the real win comes from improving process.'} "
                        f"Next time, before each decision, pause and ask: 'What information haven't I looked at yet?'"
                    )

        # Generic coaching without profile
        if pl > 0 and ps >= 70:
            return (
                "Solid performance — both your outcome and process were strong. "
                "You're building good habits. Next challenge: try a harder scenario "
                "to test your discipline under more pressure."
            )
        if pl > 0 and ps < 50:
            return (
                "You made money, but your process suggests it was more luck than skill. "
                "Don't let a good outcome hide a weak process. "
                "In your next simulation, try to view all information panels before every decision."
            )
        if pl <= 0 and ps >= 70:
            return (
                "The loss stings, but your decision-making process was actually solid. "
                "This is exactly the kind of session that builds real skill — "
                "good process eventually leads to good outcomes. Keep going."
            )
        return (
            "Tough session. The good news: every simulation teaches you something. "
            "Focus on one thing next time — slow down your first decision. "
            "Give yourself at least 20 seconds to observe before acting."
        )

    # ── LIVE COACH NUDGE (Phase 1.3) ─────────────────────────────────

    async def generate_live_nudge(
        self,
        decisions: list[Decision],
        scenario: Scenario,
        current_time: int,
    ) -> dict | None:
        """
        Quick bias check on latest decisions during live simulation.
        Returns a nudge message if bias intensity >= medium, else None.
        """
        if not decisions:
            return None

        # Run heuristic bias check on recent decisions (last 3)
        recent = decisions[-3:]
        bias_data = self._heuristic_bias_heatmap(recent)
        timeline = bias_data.get("timeline", [])

        # Check if any recent decision has medium+ intensity
        high_bias = [e for e in timeline if e["intensity"] in ("medium", "high")]
        if not high_bias:
            return None

        worst = max(high_bias, key=lambda e: sum(e["biases"].values()))
        dominant = max(worst["biases"], key=worst["biases"].get)

        if self.use_mock or not self.client:
            return self._heuristic_nudge(dominant, worst)

        prompt = f"""<role>
You are a real-time trading coach.
</role>

<context>
The user just made a decision that shows signs of {dominant} bias.
Evidence: {worst['evidence']}
Time: {current_time}s into simulation
</context>

<task>
Write ONE sentence of coaching (max 20 words). Be direct, specific, and actionable. No jargon.
</task>

<output_format>
Return JSON: {{"message": "your coaching nudge", "bias": "{dominant}"}}
</output_format>

<constraints>
- Maximum 20 words
- Be direct, specific, and actionable
- No jargon
</constraints>"""

        try:
            data = await self._call_gemini(
                prompt, _LiveNudgeOutput, None,
                call_type="nudge",
            )
            return data
        except Exception as e:
            logger.warning("Live nudge Gemini failed: %s", e)
            return self._heuristic_nudge(dominant, worst)

    def _heuristic_nudge(self, bias: str, entry: dict) -> dict:
        nudges = {
            "fomo": "Pause — are you buying because of analysis or because you're afraid of missing out?",
            "impulsivity": "Slow down. What information haven't you checked yet?",
            "loss_aversion": "Before selling, ask: did the fundamentals change, or just the price?",
            "overconfidence": "High confidence with limited data? Consider what you might be missing.",
            "anchoring": "Are you anchored to the entry price? Focus on current conditions instead.",
            "social_proof_reliance": "Following the crowd? Check if the data supports this move independently.",
        }
        return {
            "message": nudges.get(bias, "Take a breath. Review your reasoning before confirming."),
            "bias": bias,
        }

    # ── DECISION CHALLENGE (Phase 2.4) ────────────────────────────────

    async def challenge_reasoning(
        self,
        decision_type: str,
        amount: float | None,
        rationale: str,
        scenario: Scenario,
        current_state: dict,
        decisions_so_far: list[Decision],
    ) -> dict:
        """Rate the user's reasoning before they confirm a decision."""
        if self.use_mock or not self.client:
            return self._heuristic_challenge(decision_type, amount, rationale, current_state)

        prompt = f"""<role>
You are a trading coach evaluating reasoning BEFORE a decision is made.
</role>

<context>
<scenario>
{_scenario_summary(scenario)}
</scenario>

<current_state>
Price: ${current_state.get('current_price', 0):.2f}
Sentiment: {current_state.get('available_info', {}).get('market_sentiment', 'unknown')}
</current_state>

<previous_decisions>
{_decision_trace(decisions_so_far) if decisions_so_far else 'None yet'}
</previous_decisions>

<proposed_decision>
Action: {decision_type.upper()} {f'${amount}' if amount else ''}
User's reasoning: "{rationale}"
</proposed_decision>
</context>

<task>
Rate the reasoning quality 1-5 and give ONE sentence of feedback.
</task>

<output_format>
Return JSON: {{"reasoning_score": 1-5, "feedback": "one sentence"}}
</output_format>

<constraints>
- Score based on analytical depth, not whether the decision is correct
- Feedback must be exactly one sentence
</constraints>"""

        try:
            return await self._call_gemini(
                prompt, _ChallengeOutput, None,
                call_type="challenge",
            )
        except Exception as e:
            logger.warning("Challenge Gemini failed: %s", e)
            return self._heuristic_challenge(decision_type, amount, rationale, current_state)

    def _heuristic_challenge(self, decision_type: str, amount: float | None, rationale: str, state: dict) -> dict:
        rat_lower = rationale.lower()
        score = 3

        # Check for analytical keywords
        if any(w in rat_lower for w in ["data", "trend", "analysis", "risk", "information", "news"]):
            score = 4
        elif any(w in rat_lower for w in ["everyone", "moon", "hype", "fomo", "scared", "panic"]):
            score = 2
        elif len(rationale) < 10:
            score = 2

        feedback_map = {
            5: "Excellent reasoning — clear analysis with specific data points.",
            4: "Good reasoning. Consider also checking contrary signals before confirming.",
            3: "Decent but generic. What specific data supports this decision?",
            2: "Your reasoning seems emotional rather than analytical. What does the data say?",
            1: "No clear reasoning provided. Pause and think about why you're making this move.",
        }
        return {"reasoning_score": score, "feedback": feedback_map[score]}

    # ── ADAPTIVE SCENARIO GENERATOR (Phase 4) ─────────────────────────

    async def generate_adaptive_scenario(self, profile_data: dict) -> dict:
        """Generate a custom scenario targeting the user's weakest bias."""
        if self.use_mock or not self.client:
            return self._heuristic_adaptive_scenario(profile_data)

        prompt = f"""<role>
You are designing a trading simulation scenario to train a specific cognitive weakness.
</role>

<context>
<user_behavior_profile>
{json.dumps(profile_data, indent=2)}
</user_behavior_profile>

<market_realism_features>
Available features (pick appropriate ones for the difficulty):
- Transaction costs: "fixed_fee": 1.50, "pct_fee": 0.001 (commission per trade)
- Bid-ask spread: "base_spread_pct": 0.003 (wider = less liquid market)
- Liquidity constraints: "avg_volume": 500, "max_trade_size": 300 (limits order fills)
- Volatility clustering (GARCH): "volatility_clustering": true with "vol_params": {{"base_vol": 0.02, "persistence": 0.90, "shock_probability": 0.02, "shock_multiplier": 3.0}}
- Circuit breaker halts: "halts_enabled": true, "halt_threshold_pct": 0.08, "halt_duration": 15
- Order types: "order_types_enabled": true (enables limit/stop orders beyond market orders)
- News latency: "news_latency_enabled": true (news arrives with realistic delay)
- Crowd behavior model: "crowd_model_enabled": true (social sentiment affects price)
- Margin/leverage: "margin_enabled": true, "max_leverage": 2.0, "margin_call_threshold": 0.25
- Drawdown limit: "drawdown_limit": 0.20 (max portfolio loss before forced close)
- Correlated assets: "correlated_assets": [{{"symbol": "MKTIDX", "correlation": 0.7, "initial_price": 1000}}]

For difficulty 2: use 3-4 features (spread, news latency, crowd model)
For difficulty 3: use 5-7 features (add volatility, halts, order types, fees)
For difficulty 4: use 8-10 features (add margin, liquidity, correlated assets)
</market_realism_features>
</context>

<task>
Create a scenario that specifically targets the user's weakest area. The scenario should create situations where their dominant bias will be tested. Use a difficulty between 2-4 based on the user's experience level. Include a "market_params" object in initial_data to enable realistic market features. Choose features that make sense for the scenario's difficulty and target bias.
</task>

<output_format>
Return valid JSON:
{{
  "name": "Creative scenario name (max 50 chars)",
  "description": "2-3 sentence description of the scenario setup",
  "difficulty": 3,
  "category": "fomo_trap" | "patience_test" | "loss_aversion" | "social_proof" | "risk_management" | "contrarian",
  "time_pressure_seconds": 180,
  "initial_data": {{
    "asset": "Creative asset name",
    "price": 100,
    "your_balance": 10000,
    "market_sentiment": "neutral" | "bullish" | "bearish",
    "market_params": {{
      "base_spread_pct": 0.003,
      "news_latency_enabled": true,
      "crowd_model_enabled": true
    }},
    "news_headlines": [{{"time": 0, "content": "string", "type": "news"}}],
    "social_signals": [{{"time": 0, "content": "string", "sentiment": "neutral"}}]
  }},
  "events": [
    {{"time": 30, "type": "news", "content": "headline"}},
    {{"time": 60, "type": "price", "change": 0.1}},
    {{"time": 90, "type": "social", "content": "social post"}}
  ],
  "target_bias": "the bias this scenario tests"
}}
</output_format>

<constraints>
- Create 6-8 events spread across the timeline
- Include at least 2 price events, 2 news events, and 2 social events
- Difficulty should be between 2-4 based on the user's experience level
- Scenario should create situations where the user's dominant bias will be tested
</constraints>"""

        try:
            return await self._call_gemini(
                prompt, _AdaptiveScenarioOutput, None,
                call_type="adaptive_scenario",
            )
        except Exception as e:
            logger.error("Adaptive scenario generation failed: %s", e)
            return self._heuristic_adaptive_scenario(profile_data)

    def _heuristic_adaptive_scenario(self, profile: dict) -> dict:
        """Generate a scenario from templates based on weakest bias, with variety."""
        import random

        bp = profile.get("bias_patterns", {})
        weakest = max(bp, key=bp.get) if bp else "fomo"

        # Multiple templates per bias for variety
        templates = {
            "fomo": [
                {
                    "name": "The Viral Coin Rush",
                    "description": "A new cryptocurrency is trending on social media. Influencers are posting massive gains. Can you stay disciplined while everyone else is buying?",
                    "category": "fomo_trap",
                    "initial_data": {
                        "asset": "VIRALCOIN",
                        "price": 50,
                        "your_balance": 10000,
                        "market_sentiment": "bullish",
                        "market_params": {
                            "base_spread_pct": 0.005,
                            "news_latency_enabled": True,
                            "crowd_model_enabled": True,
                            "volatility_clustering": True,
                            "vol_params": {"base_vol": 0.025, "persistence": 0.88, "shock_probability": 0.03, "shock_multiplier": 3.0},
                            "order_types_enabled": True,
                        },
                        "news_headlines": [{"time": 0, "content": "New crypto gaining traction among retail investors", "type": "news"}],
                        "social_signals": [{"time": 0, "content": "Just bought VIRALCOIN! To the moon!", "sentiment": "bullish"}],
                    },
                    "events": [
                        {"time": 15, "type": "social", "content": "My friend made 500% on VIRALCOIN last week!"},
                        {"time": 30, "type": "price", "change": 0.15},
                        {"time": 45, "type": "social", "content": "Don't miss out! VIRALCOIN is the next big thing!"},
                        {"time": 60, "type": "price", "change": 0.08},
                        {"time": 90, "type": "news", "content": "Analysts warn of potential bubble in meme coins"},
                        {"time": 120, "type": "price", "change": -0.25},
                        {"time": 150, "type": "news", "content": "SEC investigating VIRALCOIN for market manipulation"},
                    ],
                },
                {
                    "name": "The IPO Frenzy",
                    "description": "A hot tech company just went public and shares are surging. Everyone on social media is celebrating gains. Do you chase the momentum or wait for a pullback?",
                    "category": "fomo_trap",
                    "initial_data": {
                        "asset": "HYPETEK",
                        "price": 85,
                        "your_balance": 12000,
                        "market_sentiment": "bullish",
                        "market_params": {
                            "base_spread_pct": 0.006,
                            "news_latency_enabled": True,
                            "crowd_model_enabled": True,
                            "avg_volume": 400,
                            "max_trade_size": 200,
                            "order_types_enabled": True,
                        },
                        "news_headlines": [{"time": 0, "content": "HYPETEK IPO opens 40% above offering price", "type": "news"}],
                        "social_signals": [{"time": 0, "content": "HYPETEK is the next trillion-dollar company!", "sentiment": "bullish"}],
                    },
                    "events": [
                        {"time": 10, "type": "social", "content": "Already up 15% since I bought at open!"},
                        {"time": 25, "type": "price", "change": 0.12},
                        {"time": 40, "type": "news", "content": "Analysts set price target 50% above current levels"},
                        {"time": 55, "type": "social", "content": "If you're not in HYPETEK you're missing generational wealth"},
                        {"time": 75, "type": "price", "change": 0.10},
                        {"time": 100, "type": "news", "content": "Insider lock-up period ends next week, insiders expected to sell"},
                        {"time": 130, "type": "price", "change": -0.30},
                        {"time": 155, "type": "news", "content": "HYPETEK reports disappointing user growth metrics"},
                    ],
                },
            ],
            "impulsivity": [
                {
                    "name": "Flash Crash Recovery",
                    "description": "A stable stock suddenly drops 20% on unclear news. Will you panic sell, buy the dip, or wait for clarity?",
                    "category": "patience_test",
                    "initial_data": {
                        "asset": "STEADYCORP",
                        "price": 200,
                        "your_balance": 10000,
                        "market_sentiment": "neutral",
                        "market_params": {
                            "base_spread_pct": 0.004,
                            "news_latency_enabled": True,
                            "crowd_model_enabled": True,
                            "halts_enabled": True,
                            "halt_threshold_pct": 0.10,
                            "halt_duration": 15,
                            "volatility_clustering": True,
                            "vol_params": {"base_vol": 0.03, "persistence": 0.90, "shock_probability": 0.04, "shock_multiplier": 3.5},
                        },
                        "news_headlines": [{"time": 0, "content": "STEADYCORP reports solid Q3 earnings", "type": "news"}],
                        "social_signals": [{"time": 0, "content": "STEADYCORP is a reliable hold", "sentiment": "neutral"}],
                    },
                    "events": [
                        {"time": 20, "type": "news", "content": "Breaking: Unconfirmed rumor about STEADYCORP accounting issues"},
                        {"time": 25, "type": "price", "change": -0.20},
                        {"time": 30, "type": "social", "content": "STEADYCORP is crashing! Sell everything!"},
                        {"time": 60, "type": "news", "content": "STEADYCORP denies accounting allegations"},
                        {"time": 90, "type": "price", "change": 0.10},
                        {"time": 120, "type": "news", "content": "Independent audit confirms STEADYCORP financials are clean"},
                        {"time": 150, "type": "price", "change": 0.12},
                    ],
                },
                {
                    "name": "The Whipsaw Trap",
                    "description": "Prices are swinging wildly on conflicting headlines. Every move seems urgent but reversals are fast. Can you resist reacting to every tick?",
                    "category": "patience_test",
                    "initial_data": {
                        "asset": "SWINGCO",
                        "price": 120,
                        "your_balance": 15000,
                        "market_sentiment": "neutral",
                        "market_params": {
                            "base_spread_pct": 0.005,
                            "news_latency_enabled": True,
                            "crowd_model_enabled": True,
                            "volatility_clustering": True,
                            "vol_params": {"base_vol": 0.035, "persistence": 0.92, "shock_probability": 0.05, "shock_multiplier": 3.0},
                            "fixed_fee": 2.00,
                            "pct_fee": 0.001,
                        },
                        "news_headlines": [{"time": 0, "content": "SWINGCO faces mixed signals from regulators", "type": "news"}],
                        "social_signals": [{"time": 0, "content": "SWINGCO is impossible to predict right now", "sentiment": "neutral"}],
                    },
                    "events": [
                        {"time": 15, "type": "price", "change": -0.08},
                        {"time": 25, "type": "social", "content": "Sell now before it drops more!"},
                        {"time": 35, "type": "price", "change": 0.12},
                        {"time": 50, "type": "news", "content": "Conflicting analyst reports: one upgrades, one downgrades"},
                        {"time": 70, "type": "price", "change": -0.10},
                        {"time": 90, "type": "social", "content": "This is the bottom, loading up!"},
                        {"time": 110, "type": "price", "change": 0.06},
                        {"time": 140, "type": "news", "content": "SWINGCO announces strategic review, shares stabilize"},
                    ],
                },
            ],
            "loss_aversion": [
                {
                    "name": "Death by a Thousand Cuts",
                    "description": "Your portfolio is slowly bleeding value day after day. Small losses compound but hope keeps whispering 'it'll come back'. When do you pull the plug?",
                    "category": "loss_aversion",
                    "initial_data": {
                        "asset": "SLOWBLEED",
                        "price": 180,
                        "your_balance": 8000,
                        "holdings": 50,
                        "market_sentiment": "bearish",
                        "market_params": {
                            "base_spread_pct": 0.004,
                            "news_latency_enabled": True,
                            "crowd_model_enabled": True,
                            "fixed_fee": 1.50,
                            "pct_fee": 0.001,
                            "order_types_enabled": True,
                        },
                        "news_headlines": [{"time": 0, "content": "Sector rotation out of tech continues", "type": "news"}],
                        "social_signals": [{"time": 0, "content": "Still holding, it'll bounce back eventually", "sentiment": "neutral"}],
                    },
                    "events": [
                        {"time": 15, "type": "price", "change": -0.03},
                        {"time": 35, "type": "news", "content": "Management lowers guidance for next quarter"},
                        {"time": 55, "type": "price", "change": -0.05},
                        {"time": 75, "type": "social", "content": "Diamond hands! Selling at a loss is for losers!"},
                        {"time": 95, "type": "price", "change": -0.04},
                        {"time": 120, "type": "news", "content": "Competitor announces superior product launch"},
                        {"time": 145, "type": "price", "change": -0.06},
                        {"time": 165, "type": "news", "content": "Buyout rumor emerges, brief spike then denial"},
                    ],
                },
                {
                    "name": "The Sunk Cost Spiral",
                    "description": "You've been averaging down on a declining stock. Every purchase has lost value. Do you keep doubling down or accept the loss and move on?",
                    "category": "loss_aversion",
                    "initial_data": {
                        "asset": "BAGHOLD",
                        "price": 95,
                        "your_balance": 12000,
                        "holdings": 80,
                        "market_sentiment": "bearish",
                        "market_params": {
                            "base_spread_pct": 0.005,
                            "news_latency_enabled": True,
                            "crowd_model_enabled": True,
                            "volatility_clustering": True,
                            "vol_params": {"base_vol": 0.02, "persistence": 0.85, "shock_probability": 0.02, "shock_multiplier": 2.5},
                        },
                        "news_headlines": [{"time": 0, "content": "BAGHOLD misses revenue estimates for third straight quarter", "type": "news"}],
                        "social_signals": [{"time": 0, "content": "I've put too much in to sell now", "sentiment": "bearish"}],
                    },
                    "events": [
                        {"time": 20, "type": "price", "change": -0.06},
                        {"time": 40, "type": "social", "content": "Average down, this is a gift at these prices!"},
                        {"time": 60, "type": "news", "content": "CFO unexpectedly resigns"},
                        {"time": 80, "type": "price", "change": -0.10},
                        {"time": 100, "type": "social", "content": "I'm down 40% but selling means locking in losses"},
                        {"time": 125, "type": "news", "content": "New CEO appointed from outside the industry"},
                        {"time": 150, "type": "price", "change": 0.08},
                        {"time": 170, "type": "news", "content": "Restructuring plan announced with layoffs"},
                    ],
                },
            ],
            "anchoring": [
                {
                    "name": "The All-Time High Anchor",
                    "description": "A stock that once traded at $300 is now at $120. Is it cheap or still overvalued? Your memory of the old price may cloud your judgment.",
                    "category": "contrarian",
                    "initial_data": {
                        "asset": "PEAKFALL",
                        "price": 120,
                        "your_balance": 15000,
                        "market_sentiment": "bearish",
                        "market_params": {
                            "base_spread_pct": 0.003,
                            "news_latency_enabled": True,
                            "crowd_model_enabled": True,
                            "order_types_enabled": True,
                        },
                        "news_headlines": [{"time": 0, "content": "PEAKFALL down 60% from all-time high of $300", "type": "news"}],
                        "social_signals": [{"time": 0, "content": "PEAKFALL was $300 last year, this is a steal!", "sentiment": "bullish"}],
                    },
                    "events": [
                        {"time": 20, "type": "news", "content": "Analysts say fair value is $80-100, not the old $300"},
                        {"time": 40, "type": "price", "change": -0.05},
                        {"time": 60, "type": "social", "content": "It was $300! Even $150 would be 25% gains from here!"},
                        {"time": 80, "type": "news", "content": "Business model fundamentally changed since peak"},
                        {"time": 100, "type": "price", "change": -0.08},
                        {"time": 130, "type": "social", "content": "Buying more, this will definitely go back to $200"},
                        {"time": 160, "type": "price", "change": 0.05},
                    ],
                },
            ],
            "social_proof": [
                {
                    "name": "The Echo Chamber",
                    "description": "Every trader you follow is bullish on the same stock. Forums are unanimous. But what if the crowd is wrong? Can you think independently?",
                    "category": "social_proof",
                    "initial_data": {
                        "asset": "CROWDFAV",
                        "price": 75,
                        "your_balance": 10000,
                        "market_sentiment": "bullish",
                        "market_params": {
                            "base_spread_pct": 0.004,
                            "news_latency_enabled": True,
                            "crowd_model_enabled": True,
                            "volatility_clustering": True,
                            "vol_params": {"base_vol": 0.02, "persistence": 0.88, "shock_probability": 0.03, "shock_multiplier": 3.0},
                        },
                        "news_headlines": [{"time": 0, "content": "CROWDFAV trending #1 on trading forums", "type": "news"}],
                        "social_signals": [{"time": 0, "content": "Everyone is buying CROWDFAV, join the movement!", "sentiment": "bullish"}],
                    },
                    "events": [
                        {"time": 15, "type": "social", "content": "98% of traders on this forum are long CROWDFAV"},
                        {"time": 30, "type": "price", "change": 0.08},
                        {"time": 50, "type": "social", "content": "If you're not in CROWDFAV you must hate money"},
                        {"time": 70, "type": "news", "content": "Short seller publishes critical report on CROWDFAV"},
                        {"time": 85, "type": "social", "content": "Ignore the haters! Short squeeze incoming!"},
                        {"time": 110, "type": "price", "change": -0.18},
                        {"time": 140, "type": "news", "content": "SEC opens investigation into CROWDFAV social media campaigns"},
                        {"time": 165, "type": "price", "change": -0.12},
                    ],
                },
            ],
        }

        # Get templates for this bias, fall back to fomo
        bias_templates = templates.get(weakest, templates["fomo"])
        template = random.choice(bias_templates)

        return {
            **template,
            "difficulty": 3,
            "time_pressure_seconds": 180,
            "target_bias": weakest,
        }

    # ── BATCH ANALYSIS ─────────────────────────────────────────────────

    async def batch_analyze(
        self,
        simulation: Simulation,
        decisions: list[Decision],
        scenario: Scenario,
        behavior_profile: dict | None = None,
    ) -> dict:
        """Single Gemini call returning reflection + counterfactuals + coaching."""
        if self.use_mock or not self.client:
            reflection = self._heuristic_analyze(simulation, decisions, scenario)
            counterfactuals = self._heuristic_counterfactuals(simulation, decisions, scenario)
            coaching = self._heuristic_coaching(simulation, decisions, scenario, behavior_profile)
            return {
                "reflection": reflection.model_dump(),
                "counterfactuals": [cf.model_dump() for cf in counterfactuals],
                "coaching_message": coaching,
            }

        prompt = self._build_batch_prompt(simulation, decisions, scenario, behavior_profile)
        cache_k = _cache_key(str(simulation.id), "batch")
        ctx_name = await self._get_or_create_context_cache(
            str(simulation.id), scenario, decisions
        )

        try:
            data = await self._call_gemini(
                prompt, _BatchAnalysisGeminiOutput, cache_k,
                call_type="batch", cached_content_name=ctx_name,
            )
            reflection_data = {
                "simulation_id": str(simulation.id),
                "outcome_summary": data["outcome_summary"],
                "outcome_type": data["outcome_type"],
                "process_quality": data["process_quality"],
                "patterns_detected": data["patterns_detected"],
                "luck_factor": data["luck_factor"],
                "skill_factor": data["skill_factor"],
                "luck_skill_explanation": data["luck_skill_explanation"],
                "insights": data["insights"],
                "key_takeaway": data["key_takeaway"],
                "coaching_message": data["coaching_message"],
                "counterfactuals": data["counterfactuals"],
            }
            return {
                "reflection": reflection_data,
                "counterfactuals": data["counterfactuals"],
                "coaching_message": data["coaching_message"],
            }
        except Exception as e:
            logger.error("Batch analyze failed, falling back: %s", e)
            reflection = self._heuristic_analyze(simulation, decisions, scenario)
            counterfactuals = self._heuristic_counterfactuals(simulation, decisions, scenario)
            coaching = self._heuristic_coaching(simulation, decisions, scenario, behavior_profile)
            return {
                "reflection": reflection.model_dump(),
                "counterfactuals": [cf.model_dump() for cf in counterfactuals],
                "coaching_message": coaching,
            }

    def _build_batch_prompt(self, simulation, decisions, scenario, profile) -> str:
        outcome = simulation.final_outcome or {}
        profile_str = ""
        if profile:
            profile_str = f"\nUSER'S BEHAVIOR PROFILE:\n{json.dumps(profile, indent=2)}\n"

        return f"""<role>
You are a behavioral finance expert analyzing a trading simulation comprehensively.
</role>

<context>
<scenario>
{_scenario_summary(scenario)}
</scenario>

<decision_trace>
{_decision_trace(decisions)}
</decision_trace>

<outcome>
Profit/Loss: ${outcome.get("profit_loss", 0):.2f} | Final Value: ${outcome.get("final_value", 10000):.2f}
</outcome>
{profile_str}
</context>

<task>
Provide a COMPLETE analysis in a single response covering: reflection, counterfactuals, and coaching.
</task>

<output_format>
Return valid JSON matching this schema:
{{
  "outcome_summary": "+$X.XX" or "-$X.XX",
  "outcome_type": "profit" | "loss" | "break_even",
  "process_quality": {{"score": 0-100, "factors": {{"timing": 0-1, "information_usage": 0-1, "risk_sizing": 0-1, "emotional_control": 0-1}}, "summary": "one paragraph"}},
  "patterns_detected": [{{"pattern_name": "string", "confidence": 0-1, "evidence": ["strings"], "description": "string"}}],
  "luck_factor": 0-1,
  "skill_factor": 0-1,
  "luck_skill_explanation": "string",
  "insights": [{{"title": "string", "description": "string", "related_pattern": "string or null", "recommended_card_id": "string or null"}}],
  "key_takeaway": "string",
  "coaching_message": "personalized 3-5 sentence coaching message",
  "counterfactuals": [
    {{"timeline_name": "string", "description": "string", "market_changes": "string", "outcome": {{"profit_loss": number, "final_value": number}}, "lesson": "string"}}
  ]
}}
</output_format>

<constraints>
- Generate exactly 3 counterfactuals
- All evidence must cite specific decision numbers and timings
- Focus on PROCESS quality, not just outcome
</constraints>"""

    # ── BIAS HEATMAP ─────────────────────────────────────────────────

    async def analyze_bias_timeline(
        self,
        simulation: Simulation,
        decisions: list[Decision],
        scenario: Scenario,
    ) -> dict:
        """Bias intensity at each decision point."""
        if self.use_mock or not self.client:
            return self._heuristic_bias_heatmap(decisions)

        prompt = self._build_bias_heatmap_prompt(simulation, decisions, scenario)
        cache_k = _cache_key(str(simulation.id), "bias_heatmap")
        ctx_name = await self._get_or_create_context_cache(
            str(simulation.id), scenario, decisions
        )

        try:
            return await self._call_gemini(
                prompt, _BiasHeatmapGeminiOutput, cache_k,
                call_type="bias_heatmap", cached_content_name=ctx_name,
            )
        except Exception as e:
            logger.error("Bias heatmap failed, falling back: %s", e)
            return self._heuristic_bias_heatmap(decisions)

    def _build_bias_heatmap_prompt(self, simulation, decisions, scenario) -> str:
        return f"""<role>
You are analyzing decision-making biases across time in a trading simulation.
</role>

<context>
<scenario>
{_scenario_summary(scenario)}
</scenario>

<decisions>
{_decision_trace(decisions)}
</decisions>
</context>

<task>
Analyze EVERY decision for these biases: fomo, impulsivity, loss_aversion, overconfidence, anchoring, social_proof_reliance. Set score to 0 if not present.
</task>

<output_format>
Return JSON: {{"timeline": [{{"timestamp_seconds": int, "decision_index": int, "biases": {{"bias_name": 0-1}}, "evidence": "string", "intensity": "low"|"medium"|"high"}}], "peak_bias_moment": int, "dominant_bias": "string"}}
</output_format>

<constraints>
- Every decision must be analyzed, even if no bias is detected
- Set bias scores to 0 if not present
- Evidence must reference specific decision data (timing, action, market conditions)
</constraints>"""

    def _heuristic_bias_heatmap(self, decisions: list[Decision]) -> dict:
        timeline = []
        max_intensity = 0
        peak_moment = 0
        bias_sums: dict[str, float] = {}

        for i, d in enumerate(decisions):
            ms = _get_market_state(d)
            sentiment = ms.get("available_info", {}).get("market_sentiment", "neutral")
            time_spent = d.time_spent_seconds or 10
            biases: dict[str, float] = {}

            # FOMO
            if d.decision_type == "buy" and sentiment == "bullish":
                biases["fomo"] = min(0.9, 0.5 + (1 / max(time_spent, 1)) * 2)
            else:
                biases["fomo"] = 0.0

            # Impulsivity
            if time_spent < 5 and d.decision_type in ("buy", "sell"):
                biases["impulsivity"] = min(0.9, (5 - time_spent) / 5)
            else:
                biases["impulsivity"] = 0.0

            # Loss aversion
            if d.decision_type == "sell" and sentiment == "bearish":
                biases["loss_aversion"] = 0.6
            else:
                biases["loss_aversion"] = 0.0

            # Overconfidence
            if (d.confidence_level or 3) >= 4 and len(d.info_viewed or []) < 2:
                biases["overconfidence"] = 0.5
            else:
                biases["overconfidence"] = 0.0

            total = sum(biases.values())
            intensity = "high" if total > 1.0 else ("medium" if total > 0.4 else "low")

            if total > max_intensity:
                max_intensity = total
                peak_moment = d.simulation_time

            for k, v in biases.items():
                bias_sums[k] = bias_sums.get(k, 0) + v

            evidence_parts = []
            if biases.get("fomo", 0) > 0:
                evidence_parts.append(f"bought during bullish sentiment")
            if biases.get("impulsivity", 0) > 0:
                evidence_parts.append(f"decided in {time_spent:.1f}s")
            if biases.get("loss_aversion", 0) > 0:
                evidence_parts.append(f"sold during bearish conditions")
            evidence = "; ".join(evidence_parts) or "No strong bias signals"

            timeline.append({
                "timestamp_seconds": d.simulation_time,
                "decision_index": i,
                "biases": biases,
                "evidence": evidence,
                "intensity": intensity,
            })

        dominant = max(bias_sums, key=bias_sums.get) if bias_sums else "none"

        return {
            "timeline": timeline,
            "peak_bias_moment": peak_moment,
            "dominant_bias": dominant,
        }

    # ── RATIONALE REVIEW ─────────────────────────────────────────────

    async def review_rationales(
        self,
        simulation: Simulation,
        decisions: list[Decision],
        scenario: Scenario,
    ) -> dict:
        """Critique user's stated rationales for each decision."""
        if self.use_mock or not self.client:
            return self._heuristic_rationale_review(decisions)

        prompt = self._build_rationale_prompt(simulation, decisions, scenario)
        cache_k = _cache_key(str(simulation.id), "rationale_review")
        ctx_name = await self._get_or_create_context_cache(
            str(simulation.id), scenario, decisions
        )

        try:
            return await self._call_gemini(
                prompt, _RationaleReviewOutput, cache_k,
                call_type="rationale_review", cached_content_name=ctx_name,
            )
        except Exception as e:
            logger.error("Rationale review failed, falling back: %s", e)
            return self._heuristic_rationale_review(decisions)

    def _build_rationale_prompt(self, simulation, decisions, scenario) -> str:
        rationales = []
        for i, d in enumerate(decisions):
            if d.rationale:
                rationales.append(f"Decision #{i+1} ({d.decision_type}): \"{d.rationale}\"")
        return f"""<role>
You are a behavioral coach evaluating an investor's stated reasoning.
</role>

<context>
<scenario>
{_scenario_summary(scenario)}
</scenario>

<decisions>
{_decision_trace(decisions)}
</decisions>

<user_rationales>
{chr(10).join(rationales)}
</user_rationales>
</context>

<task>
Critique each rationale for analytical quality. Identify specific factors the user missed (data, contrary signals, risk assessment).
</task>

<output_format>
Return JSON: {{"reviews": [{{"decision_index": int, "user_rationale": "string", "critique": "string", "quality_score": 1-5, "missed_factors": ["strings"], "reasoning_bias": "string or null"}}], "summary": "string", "overall_reasoning_quality": 1-5}}
</output_format>

<constraints>
- Score quality: 5=exceptional reasoning, 1=emotional/no reasoning
- Identify specific factors they missed (data, contrary signals, risk assessment)
- Critiques must reference the actual scenario context and market conditions
</constraints>"""

    def _heuristic_rationale_review(self, decisions: list[Decision]) -> dict:
        reviews = []
        total_quality = 0
        count = 0
        for i, d in enumerate(decisions):
            if not d.rationale:
                continue
            rat = d.rationale.lower()
            quality = 3
            missed = []
            bias = None

            if any(w in rat for w in ["everyone", "trending", "hype", "moon"]):
                quality = 2
                bias = "social_proof_reliance"
                missed.append("Independent analysis of fundamentals")
            elif any(w in rat for w in ["falling", "scared", "losing", "panic"]):
                quality = 2
                bias = "loss_aversion"
                missed.append("Assessment of whether scenario fundamentals changed")
            elif any(w in rat for w in ["analysis", "data", "information", "trend"]):
                quality = 4
                missed.append("Specific data points cited") if "price" not in rat else None
            else:
                missed.append("Specific reasoning for chosen action")
                missed.append("Risk assessment")

            missed = [m for m in missed if m]
            reviews.append({
                "decision_index": i,
                "user_rationale": d.rationale,
                "critique": f"Your reasoning {'shows analytical thinking' if quality >= 4 else 'could be more specific and data-driven'}.",
                "quality_score": quality,
                "missed_factors": missed,
                "reasoning_bias": bias,
            })
            total_quality += quality
            count += 1

        overall = round(total_quality / max(count, 1))
        return {
            "reviews": reviews,
            "summary": f"Across {count} rationales, average quality was {overall}/5." if count > 0 else "No rationales provided.",
            "overall_reasoning_quality": max(1, min(5, overall)) if count > 0 else 3,
        }

    # ── COUNTERFACTUAL ISOLATION ─────────────────────────────────────

    async def isolate_counterfactual(
        self,
        simulation: Simulation,
        decisions: list[Decision],
        scenario: Scenario,
        target_decision_index: int,
    ) -> dict:
        """Show causal impact of changing a single decision."""
        if self.use_mock or not self.client:
            return self._heuristic_isolate_counterfactual(simulation, decisions, scenario, target_decision_index)

        target = decisions[target_decision_index]
        prompt = f"""<role>
You are analyzing the causal impact of changing ONE decision in an investment simulation.
</role>

<context>
<scenario>
{_scenario_summary(scenario)}
</scenario>

<all_decisions>
{_decision_trace(decisions)}
</all_decisions>

<target_decision>
Decision #{target_decision_index + 1} — {target.decision_type} at t={target.simulation_time}s
</target_decision>
</context>

<task>
What would happen if ONLY this decision changed (to the opposite action)? Trace the ripple effects through subsequent decisions and the final outcome.
</task>

<output_format>
Return JSON: {{"original_decision": "string", "alternative_decision": "string", "ripple_effects": ["strings"], "original_outcome": {{"profit_loss": number, "final_value": number}}, "alternative_outcome": {{"profit_loss": number, "final_value": number}}, "causal_impact": number, "lesson": "string"}}
</output_format>

<constraints>
- Only change the target decision; keep all other decisions the same
- Ripple effects must be logically plausible given the scenario timeline
- The lesson must be specific to this decision, not generic advice
</constraints>"""

        cache_k = _cache_key(str(simulation.id), f"isolate_{target_decision_index}")
        ctx_name = await self._get_or_create_context_cache(
            str(simulation.id), scenario, decisions
        )
        try:
            return await self._call_gemini(
                prompt, _IsolatedCounterfactualOutput, cache_k,
                call_type="isolate", cached_content_name=ctx_name,
            )
        except Exception as e:
            logger.error("Counterfactual isolation failed: %s", e)
            return self._heuristic_isolate_counterfactual(simulation, decisions, scenario, target_decision_index)

    def _heuristic_isolate_counterfactual(self, simulation, decisions, scenario, idx) -> dict:
        outcome = simulation.final_outcome or {}
        target = decisions[idx]
        init_balance = scenario.initial_data.get("your_balance", 10000)
        price = target.price_at_decision or scenario.initial_data.get("price", 100)
        amount = target.amount or 0

        # Calculate what would happen with opposite action
        if target.decision_type == "buy":
            alt_action = "hold (skip this buy)"
            impact = amount * price * 0.1  # Rough estimate
            alt_pl = outcome.get("profit_loss", 0) + impact
        elif target.decision_type == "sell":
            alt_action = "hold (keep the position)"
            impact = -amount * price * 0.15
            alt_pl = outcome.get("profit_loss", 0) + impact
        else:
            alt_action = f"buy small position ({init_balance * 0.05:.0f})"
            impact = init_balance * 0.05 * 0.1
            alt_pl = outcome.get("profit_loss", 0) + impact

        return {
            "original_decision": f"{target.decision_type} {amount} at t={target.simulation_time}s, price=${price:.2f}",
            "alternative_decision": alt_action,
            "ripple_effects": [
                f"Portfolio allocation changes from this point forward",
                f"Subsequent decisions would have different context",
                f"Final exposure different by ~${abs(impact):.2f}",
            ],
            "original_outcome": {"profit_loss": outcome.get("profit_loss", 0), "final_value": outcome.get("final_value", init_balance)},
            "alternative_outcome": {"profit_loss": alt_pl, "final_value": init_balance + alt_pl},
            "causal_impact": impact,
            "lesson": f"This single {'buy' if target.decision_type == 'buy' else 'sell'} decision accounts for ~${abs(impact):.2f} of your result. {'Skipping impulsive buys' if target.decision_type == 'buy' else 'Holding through volatility'} can significantly change outcomes.",
        }

    # ── ADAPTIVE COACH PERSONA ───────────────────────────────────────

    def _determine_persona(self, profile: dict | None) -> tuple[str, str]:
        """Determine coaching persona based on behavior profile."""
        if not profile:
            return "supportive", ""

        sims = profile.get("total_simulations_analyzed", 0)
        bias_patterns = profile.get("bias_patterns", {})
        repeat_biases = [k for k, v in bias_patterns.items() if v > 0.6]

        if sims <= 2:
            return "encouraging", (
                "PERSONA: You are an encouraging, warm mentor. "
                "The user is new. Celebrate small wins, be gentle with criticism, "
                "use phrases like 'great start' and 'you're building good habits'."
            )
        elif sims > 5 and len(repeat_biases) >= 2:
            return "strict", (
                f"PERSONA: You are a direct, no-nonsense coach. "
                f"The user has done {sims} simulations and STILL shows strong "
                f"{', '.join(repeat_biases)} patterns. Be honest and firm. "
                f"Challenge them. Use phrases like 'we need to talk about...' "
                f"and 'this pattern keeps appearing'."
            )
        elif sims > 3:
            return "analytical", (
                "PERSONA: You are an analytical coach who speaks in data. "
                "Reference specific metrics and trends. Compare this session "
                "to their historical averages."
            )
        return "supportive", ""

    # ── PLAYBOOK ─────────────────────────────────────────────────────

    async def generate_playbook(self, profile_data: dict) -> dict:
        """Generate a personal do/don't playbook from the behavior profile."""
        if self.use_mock or not self.client:
            return self._heuristic_playbook(profile_data)

        prompt = f"""<role>
You are generating a personal trading playbook based on behavioral analysis.
</role>

<context>
<behavior_profile>
{json.dumps(profile_data, indent=2)}
</behavior_profile>
</context>

<task>
Create a personalized do/don't playbook with specific rules tailored to this user's behavioral patterns.
</task>

<output_format>
Return JSON: {{"dos": ["3-5 specific things to do"], "donts": ["3-5 specific things to avoid"], "key_rules": ["2-3 personal rules"], "generated_from": {profile_data.get("total_simulations_analyzed", 1)}}}
</output_format>

<constraints>
- Rules must be specific to THIS user's patterns, not generic advice
- Reference the user's actual bias patterns and weaknesses
- Keep rules actionable and measurable
</constraints>"""

        try:
            return await self._call_gemini(
                prompt, _PlaybookOutput, "playbook",
                call_type="playbook",
            )
        except Exception as e:
            logger.error("Playbook generation failed: %s", e)
            return self._heuristic_playbook(profile_data)

    async def check_playbook_adherence(self, playbook: dict, decisions: list[Decision], simulation: Simulation) -> dict:
        """Check how well a user followed their playbook in a simulation."""
        if self.use_mock or not self.client:
            return self._heuristic_adherence(playbook, decisions)

        prompt = f"""<role>
You are a trading coach checking playbook adherence after a simulation.
</role>

<context>
<playbook>
DOs: {playbook.get("dos", [])}
DON'Ts: {playbook.get("donts", [])}
Rules: {playbook.get("key_rules", [])}
</playbook>

<decisions>
{_decision_trace(decisions)}
</decisions>
</context>

<task>
Check if the user followed their personal trading playbook. Identify which rules were followed and which were violated, with specific evidence from the decisions.
</task>

<output_format>
Return JSON: {{"adherence_score": 0-100, "followed": ["rules that were followed"], "violated": ["rules that were violated"], "specific_examples": ["evidence from decisions"]}}
</output_format>

<constraints>
- Every rule in the playbook must be evaluated
- Specific examples must reference actual decision data (timing, amounts, actions)
- Score should reflect the proportion of rules followed vs violated
</constraints>"""

        cache_k = _cache_key(str(simulation.id), "adherence")
        try:
            return await self._call_gemini(
                prompt, _PlaybookAdherenceOutput, cache_k,
                call_type="adherence",
            )
        except Exception as e:
            logger.error("Adherence check failed: %s", e)
            return self._heuristic_adherence(playbook, decisions)

    def _heuristic_playbook(self, profile: dict) -> dict:
        weaknesses = profile.get("weaknesses", [])
        strengths = profile.get("strengths", [])
        bp = profile.get("bias_patterns", {})

        dos = ["Wait at least 20 seconds before your first decision"]
        donts = ["Don't trade more than 10% of capital in a single decision"]
        rules = ["Process over outcome — judge yourself by HOW you decided, not the result"]

        if "fomo_susceptibility" in weaknesses or bp.get("fomo", 0) > 0.5:
            dos.append("Check all information panels before any buy decision")
            donts.append("Don't buy just because prices are rising")
        if "impulsivity" in weaknesses or bp.get("impulsivity", 0) > 0.5:
            dos.append("Spend at least 5 seconds deliberating each decision")
            donts.append("Don't make back-to-back trades within 10 seconds")
        if "patience" in strengths:
            dos.append("Continue using your observation period — it's working")
        if bp.get("loss_aversion", 0) > 0.5:
            donts.append("Don't sell purely because of bearish sentiment")
            rules.append("Ask: 'Did the fundamentals change, or just the price?'")

        if len(dos) < 3:
            dos.append("Review your portfolio exposure after each trade")
        if len(donts) < 3:
            donts.append("Don't ignore contrary information")

        return {
            "dos": dos[:5],
            "donts": donts[:5],
            "key_rules": rules[:3],
            "generated_from": profile.get("total_simulations_analyzed", 1),
        }

    def _heuristic_adherence(self, playbook: dict, decisions: list[Decision]) -> dict:
        followed = []
        violated = []
        examples = []
        score = 70  # Base score

        dos = playbook.get("dos", [])
        donts = playbook.get("donts", [])

        # Check wait time rule
        if decisions and decisions[0].simulation_time >= 20:
            followed.append("Waited before first decision")
        elif decisions and any("wait" in d.lower() for d in dos):
            violated.append("First decision was too quick")
            examples.append(f"First decision at t={decisions[0].simulation_time}s (rule: wait 20s)")
            score -= 15

        # Check position sizing
        for d in decisions:
            if d.decision_type == "buy" and (d.amount or 0) > 1000:
                violated.append("Position size too large")
                examples.append(f"Bought {d.amount} at t={d.simulation_time}s")
                score -= 10
                break

        # Check deliberation time
        quick = sum(1 for d in decisions if (d.time_spent_seconds or 10) < 5)
        if quick == 0:
            followed.append("Took time on every decision")
        elif quick > 0:
            violated.append("Some decisions made too quickly")
            examples.append(f"{quick} decisions in under 5 seconds")
            score -= quick * 5

        if not violated:
            followed.append("Good overall adherence to playbook rules")

        return {
            "adherence_score": max(0, min(100, score)),
            "followed": followed,
            "violated": violated,
            "specific_examples": examples,
        }

    def _heuristic_profile_update(self, existing: dict | None, decisions: list[Decision]) -> dict:
        profile = existing or {
            "strengths": [],
            "weaknesses": [],
            "bias_patterns": {},
            "decision_style": "unknown",
            "stress_response": "unknown",
        }

        # Simple heuristic updates
        quick = sum(1 for d in decisions if (d.time_spent_seconds or 10) < 5)
        total = max(len(decisions), 1)

        bp = profile.get("bias_patterns", {})
        old_imp = bp.get("impulsivity", 0.5)
        new_imp = quick / total
        bp["impulsivity"] = old_imp * 0.7 + new_imp * 0.3  # Running average

        if new_imp < 0.3:
            profile.setdefault("strengths", [])
            if "patience" not in profile["strengths"]:
                profile["strengths"].append("patience")

        profile["bias_patterns"] = bp
        profile["improvement_notes"] = "Updated from latest simulation"
        return profile

    # ── GOOGLE SEARCH GROUNDING — credibility verification ───────────

    async def verify_claim_credibility(
        self,
        claim: str,
        source_type: str = "news",
    ) -> dict:
        """
        Use Google Search grounding to fact-check a news claim or social post.
        Returns a credibility assessment with grounding sources.
        """
        if self.use_mock or not self.client:
            return self._heuristic_credibility(claim, source_type)

        prompt = f"""<role>
You are a financial news credibility analyst with access to Google Search.
</role>

<context>
<claim>{claim}</claim>
<source_type>{source_type}</source_type>
</context>

<task>
Use Google Search to verify this claim. Check if reputable sources confirm or deny it.
</task>

<output_format>
Return valid JSON:
{{
  "credibility_score": 0.0-1.0,
  "verdict": "verified" | "partially_verified" | "unverified" | "likely_false",
  "supporting_sources": ["list of confirming source names/outlets"],
  "contradicting_sources": ["list of contradicting source names/outlets"],
  "key_finding": "one-sentence summary of what search revealed",
  "risk_level": "low" | "medium" | "high"
}}
</output_format>

<constraints>
- Score 0.9+ only if multiple reputable outlets confirm.
- Score below 0.3 if no credible sources found or sources contradict.
- Be specific about which sources you found.
</constraints>"""

        try:
            data = await self._call_gemini(
                prompt, _CredibilityCheckOutput, None,
                call_type="nudge",  # Low thinking — needs to be fast
                use_search_grounding=True,
            )

            # Merge real grounding metadata into response
            gm = data.pop("_grounding_metadata", None)
            if gm:
                # Add real source URLs from grounding chunks
                chunks = gm.get("grounding_chunks", [])
                if chunks:
                    data["grounding_source_urls"] = chunks

                # Add search queries used
                queries = gm.get("search_queries", [])
                if queries:
                    data["search_queries_used"] = queries

            return data
        except Exception as e:
            logger.warning("Search grounding credibility check failed: %s", e)
            return self._heuristic_credibility(claim, source_type)

    def _heuristic_credibility(self, claim: str, source_type: str) -> dict:
        """Deterministic credibility scoring based on claim characteristics."""
        claim_lower = claim.lower()

        # Red flags
        red_flags = ["guaranteed", "moon", "100x", "insider", "secret", "manipulation"]
        flag_count = sum(1 for flag in red_flags if flag in claim_lower)

        # Credibility indicators
        credible_signals = ["sec", "filing", "earnings", "quarterly", "official", "reuters", "bloomberg"]
        credible_count = sum(1 for sig in credible_signals if sig in claim_lower)

        if source_type == "social":
            base_score = 0.3
        elif source_type == "news":
            base_score = 0.6
        else:
            base_score = 0.5

        score = max(0.0, min(1.0, base_score + credible_count * 0.15 - flag_count * 0.2))

        if score >= 0.7:
            verdict = "verified"
        elif score >= 0.4:
            verdict = "partially_verified"
        elif score >= 0.2:
            verdict = "unverified"
        else:
            verdict = "likely_false"

        return {
            "credibility_score": round(score, 2),
            "verdict": verdict,
            "supporting_sources": [],
            "contradicting_sources": [],
            "key_finding": f"Heuristic assessment based on {source_type} source characteristics.",
            "risk_level": "high" if score < 0.3 else ("medium" if score < 0.6 else "low"),
        }

    # ── URL CONTEXT — generate scenarios from article URLs ───────────

    async def generate_scenario_from_url(
        self,
        url: str,
        difficulty: int = 3,
    ) -> dict:
        """
        Use Gemini URL Context to read an article and generate a trading scenario.
        The user pastes a URL, Gemini extracts the key facts, and creates a sim.
        """
        if self.use_mock or not self.client:
            return self._heuristic_url_scenario(url, difficulty)

        if not settings.gemini_enable_url_context:
            return self._heuristic_url_scenario(url, difficulty)

        prompt = f"""<role>
You are designing a trading simulation scenario based on a real article.
</role>

<context>
<article_url>{url}</article_url>
<target_difficulty>{difficulty}</target_difficulty>
</context>

<task>
Read the article at the URL above and extract:
1. The main company/asset involved
2. Key claims and events
3. Market impact potential
4. Conflicting viewpoints or risks

Then create a simulation scenario where the user must navigate decisions based on events inspired by this article.
Include "market_params" with appropriate realism features for difficulty {difficulty}.
</task>

<output_format>
Return valid JSON:
{{
  "name": "Scenario name (max 50 chars)",
  "description": "2-3 sentence description of the scenario",
  "category": "fomo_trap" | "loss_aversion" | "risk_management" | "social_proof" | "patience_test",
  "difficulty": {difficulty},
  "time_pressure_seconds": 120-240,
  "source_url": "{url}",
  "source_summary": "2-3 sentence summary of the article",
  "initial_data": {{
    "asset": "TICKER (fictional, inspired by article)",
    "price": 10-500,
    "your_balance": 10000,
    "market_sentiment": "bullish" | "bearish" | "neutral",
    "market_params": {{
      "fixed_fee": 1.50,
      "pct_fee": 0.001,
      "base_spread_pct": 0.003,
      "volatility_clustering": true,
      "vol_params": {{"base_vol": 0.02, "persistence": 0.90, "shock_probability": 0.02, "shock_multiplier": 3.0}}
    }}
  }},
  "events": [
    {{"time": 10, "type": "news"|"social"|"price", "content": "string", "change": null}},
    ...6-8 events spread across the timeline
  ]
}}
</output_format>

<constraints>
- Events should mirror the real article's narrative arc (buildup, climax, resolution)
- Include at least 2 conflicting signals to test decision-making
- Asset ticker should be fictional (not the real company)
- Scenario should be playable in 2-4 minutes
</constraints>"""

        try:
            data = await self._call_gemini(
                prompt, _URLScenarioOutput, None,
                call_type="adaptive_scenario",
                use_url_context=True,
            )

            # Merge URL context metadata (retrieval statuses) into response
            url_meta = data.pop("_url_context_metadata", None)
            if url_meta:
                data["url_retrieval_metadata"] = url_meta.get("url_metadata", [])

            return data
        except Exception as e:
            logger.error("URL-based scenario generation failed: %s", e)
            return self._heuristic_url_scenario(url, difficulty)

    def _heuristic_url_scenario(self, url: str, difficulty: int) -> dict:
        """Fallback scenario when URL context is unavailable."""
        return {
            "name": "Breaking News Challenge",
            "description": "Navigate a scenario inspired by breaking financial news. "
                         "Evaluate claims, check sources, and make disciplined decisions.",
            "category": "risk_management",
            "difficulty": difficulty,
            "time_pressure_seconds": 180,
            "source_url": url,
            "source_summary": "Scenario generated from heuristic fallback — URL context unavailable.",
            "initial_data": {
                "asset": "NEWSX",
                "price": 75.00,
                "your_balance": 10000,
                "market_sentiment": "neutral",
                "market_params": {
                    "fixed_fee": 1.50,
                    "pct_fee": 0.001,
                    "base_spread_pct": 0.003,
                },
            },
            "events": [
                {"time": 10, "type": "news", "content": "Breaking: Major announcement expected from company", "change": None},
                {"time": 25, "type": "social", "content": "Everyone is talking about this stock!", "change": None},
                {"time": 40, "type": "price", "content": None, "change": 0.08},
                {"time": 60, "type": "news", "content": "Analyst downgrades to HOLD, cites valuation concerns", "change": None},
                {"time": 80, "type": "price", "content": None, "change": -0.12},
                {"time": 100, "type": "social", "content": "Panic selling everywhere — time to buy the dip?", "change": None},
                {"time": 130, "type": "news", "content": "Company issues official statement clarifying situation", "change": None},
                {"time": 160, "type": "price", "content": None, "change": 0.05},
            ],
        }


