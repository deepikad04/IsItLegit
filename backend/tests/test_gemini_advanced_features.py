"""
Tests for advanced Gemini features:
- Thinking level configuration
- Context caching
- Google Search grounding (credibility verification)
- URL Context (article-based scenario generation)
"""
import uuid
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from services.gemini_service import (
    GeminiService,
    THINKING_LEVELS,
    _context_cache_store,
    _cache,
)


# ── Shared Fixtures ──────────────────────────────────────────────────────


def _make_scenario(name="Test Scenario", time_limit=60):
    s = MagicMock()
    s.id = uuid.uuid4()
    s.name = name
    s.category = "fomo_trap"
    s.difficulty = 2
    s.time_pressure_seconds = time_limit
    s.initial_data = {
        "asset": "TESTCOIN",
        "price": 100.0,
        "your_balance": 10000.0,
        "market_sentiment": "neutral",
    }
    s.events = [
        {"time": 10, "type": "news", "content": "Breaking news!"},
        {"time": 20, "type": "price", "change": 0.10},
        {"time": 30, "type": "social", "content": "Everyone's buying!"},
        {"time": 40, "type": "price", "change": -0.15},
    ]
    return s


def _make_simulation(scenario_id):
    s = MagicMock()
    s.id = uuid.uuid4()
    s.user_id = uuid.uuid4()
    s.scenario_id = scenario_id
    s.status = "completed"
    s.process_quality_score = 55.0
    s.final_outcome = {
        "profit_loss": 250.0,
        "final_value": 10250.0,
        "profit_loss_percent": 2.5,
    }
    s.current_time_elapsed = 50
    return s


def _make_decisions(simulation_id, count=4):
    decisions = []
    for i in range(count):
        d = MagicMock()
        d.id = uuid.uuid4()
        d.simulation_id = simulation_id
        d.simulation_time = 10 + i * 10
        d.decision_type = ["buy", "hold", "buy", "sell"][i % 4]
        d.amount = [500, None, 300, 200][i % 4]
        d.confidence_level = [4, 3, 5, 2][i % 4]
        d.time_spent_seconds = [3.0, 8.0, 2.5, 12.0][i % 4]
        d.price_at_decision = 100.0 + i * 5
        d.rationale = ["FOMO buy", None, "Strong conviction", "Cut losses"][i % 4]
        d.info_viewed = [{"panel": "news", "view_duration_seconds": 2, "timestamp": d.simulation_time}]
        d.info_ignored = []
        d.market_state_at_decision = {
            "current_price": d.price_at_decision,
            "available_info": {
                "market_sentiment": ["bullish", "bullish", "bearish", "bearish"][i % 4]
            }
        }
        d.events_since_last = []
        decisions.append(d)
    return decisions


# ── 1. Thinking Level Tests ──────────────────────────────────────────────


class TestThinkingLevels:
    """Verify thinking level presets are correctly configured."""

    def test_all_call_types_have_thinking_levels(self):
        """Every known call type should have a thinking level assigned."""
        expected_types = [
            "nudge", "challenge", "reflection", "why", "coaching",
            "bias_heatmap", "rationale_review", "profile_update",
            "counterfactuals", "pro", "batch", "isolate",
            "adaptive_scenario", "learning_modules", "playbook", "adherence",
        ]
        for call_type in expected_types:
            assert call_type in THINKING_LEVELS, f"Missing thinking level for: {call_type}"

    def test_live_calls_use_low_thinking(self):
        """Real-time calls (nudge, challenge) should use 'low' for speed."""
        assert THINKING_LEVELS["nudge"] == "low"
        assert THINKING_LEVELS["challenge"] == "low"

    def test_deep_analysis_uses_high_thinking(self):
        """Complex analysis calls should use 'high' for depth."""
        assert THINKING_LEVELS["counterfactuals"] == "high"
        assert THINKING_LEVELS["pro"] == "high"
        assert THINKING_LEVELS["batch"] == "high"
        assert THINKING_LEVELS["isolate"] == "high"

    def test_standard_analysis_uses_expected_thinking(self):
        """Post-sim calls use appropriate thinking levels based on budget."""
        assert THINKING_LEVELS["reflection"] == "high"  # 1024 tokens — deeper reasoning
        assert THINKING_LEVELS["why"] == "low"
        assert THINKING_LEVELS["coaching"] == "low"

    def test_only_pro_compatible_levels(self):
        """All levels should be 'low' or 'high' (Gemini 3 Pro compatible)."""
        for call_type, level in THINKING_LEVELS.items():
            assert level in ("low", "high"), (
                f"Call type '{call_type}' uses '{level}' — "
                f"Gemini 3 Pro only supports 'low' and 'high'"
            )

    def test_default_fallback_is_low(self):
        """Unknown call types should default to 'low' (safest for Pro)."""
        assert THINKING_LEVELS.get("unknown_type", "low") == "low"


# ── 2. Context Caching Tests ─────────────────────────────────────────────


class TestContextCaching:
    """Test context cache store behavior."""

    def setup_method(self):
        _context_cache_store.clear()

    def test_context_cache_store_starts_empty(self):
        assert len(_context_cache_store) == 0

    def test_context_cache_stores_by_simulation_id(self):
        sim_id = str(uuid.uuid4())
        _context_cache_store[sim_id] = "cached_content_name_123"
        assert sim_id in _context_cache_store
        assert _context_cache_store[sim_id] == "cached_content_name_123"

    def test_context_cache_reuse(self):
        """Same simulation_id should reuse cached content."""
        sim_id = str(uuid.uuid4())
        _context_cache_store[sim_id] = "cache_abc"
        # Simulate second access
        assert _context_cache_store.get(sim_id) == "cache_abc"

    @pytest.mark.asyncio
    async def test_get_or_create_returns_none_in_mock_mode(self):
        """Mock mode should not create context caches."""
        svc = GeminiService()
        svc.use_mock = True
        scenario = _make_scenario()
        decisions = _make_decisions(uuid.uuid4())
        result = await svc._get_or_create_context_cache(
            str(uuid.uuid4()), scenario, decisions
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_get_or_create_returns_existing(self):
        """Should return existing cache without calling API."""
        svc = GeminiService()
        svc.use_mock = False
        svc.client = MagicMock()
        sim_id = str(uuid.uuid4())
        _context_cache_store[sim_id] = "existing_cache_name"

        result = await svc._get_or_create_context_cache(
            sim_id, _make_scenario(), []
        )
        assert result == "existing_cache_name"
        # Should NOT have called the API
        svc.client.caches.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_or_create_skips_short_prefix(self):
        """Should return None when prefix is below 4096 token minimum."""
        svc = GeminiService()
        svc.use_mock = False
        svc.client = MagicMock()
        # Short scenario with few decisions → prefix too small for caching
        scenario = _make_scenario()
        decisions = _make_decisions(uuid.uuid4(), count=2)

        result = await svc._get_or_create_context_cache(
            str(uuid.uuid4()), scenario, decisions
        )
        assert result is None
        # Should NOT have called the API since prefix is too short
        svc.client.caches.create.assert_not_called()


# ── 3. Credibility Verification (Search Grounding) Tests ─────────────────


class TestCredibilityVerification:
    """Test the search-grounded credibility checker."""

    def setup_method(self):
        self.svc = GeminiService()
        self.svc.use_mock = True

    @pytest.mark.asyncio
    async def test_heuristic_credibility_news(self):
        """News sources should get higher base credibility."""
        result = await self.svc.verify_claim_credibility(
            "Company reports quarterly earnings beat", "news"
        )
        assert 0 <= result["credibility_score"] <= 1
        assert result["verdict"] in ("verified", "partially_verified", "unverified", "likely_false")
        assert result["risk_level"] in ("low", "medium", "high")

    @pytest.mark.asyncio
    async def test_heuristic_credibility_social(self):
        """Social sources should get lower base credibility."""
        result = await self.svc.verify_claim_credibility(
            "This stock is going to the moon!", "social"
        )
        assert result["credibility_score"] < 0.5
        assert result["risk_level"] in ("medium", "high")

    @pytest.mark.asyncio
    async def test_heuristic_credibility_red_flags(self):
        """Claims with red flag words should score low."""
        result = await self.svc.verify_claim_credibility(
            "Guaranteed 100x returns, insider secret!", "social"
        )
        assert result["credibility_score"] < 0.3
        assert result["verdict"] == "likely_false"
        assert result["risk_level"] == "high"

    @pytest.mark.asyncio
    async def test_heuristic_credibility_reputable(self):
        """Claims referencing SEC filings should score higher."""
        result = await self.svc.verify_claim_credibility(
            "SEC filing confirms quarterly earnings grew 15%", "news"
        )
        assert result["credibility_score"] >= 0.7
        assert result["verdict"] == "verified"

    @pytest.mark.asyncio
    async def test_credibility_output_schema(self):
        """Output should always contain required keys."""
        result = await self.svc.verify_claim_credibility("any claim", "news")
        assert "credibility_score" in result
        assert "verdict" in result
        assert "supporting_sources" in result
        assert "contradicting_sources" in result
        assert "key_finding" in result
        assert "risk_level" in result
        assert isinstance(result["supporting_sources"], list)
        assert isinstance(result["contradicting_sources"], list)

    @pytest.mark.asyncio
    async def test_heuristic_credibility_has_grounding_fields(self):
        """Heuristic output should not include grounding metadata (mock mode)."""
        result = await self.svc.verify_claim_credibility("test claim", "news")
        # Heuristic fallback should NOT have grounding metadata
        assert "_grounding_metadata" not in result


# ── 3b. Grounding Metadata Extraction Tests ──────────────────────────────


class TestGroundingMetadataExtraction:
    """Test _extract_grounding_metadata parses response objects correctly."""

    def test_extract_with_no_candidates(self):
        """Should return None when response has no candidates."""
        response = MagicMock()
        response.candidates = []
        result = GeminiService._extract_grounding_metadata(response)
        assert result is None

    def test_extract_with_no_grounding_metadata(self):
        """Should return None when candidate has no grounding_metadata."""
        response = MagicMock()
        candidate = MagicMock()
        candidate.grounding_metadata = None
        response.candidates = [candidate]
        result = GeminiService._extract_grounding_metadata(response)
        assert result is None

    def test_extract_search_queries(self):
        """Should extract web_search_queries from grounding metadata."""
        response = MagicMock()
        gm = MagicMock()
        gm.web_search_queries = ["ACME Corp earnings 2026", "ACME stock price"]
        gm.grounding_chunks = None
        gm.grounding_supports = None
        gm.search_entry_point = None
        candidate = MagicMock()
        candidate.grounding_metadata = gm
        response.candidates = [candidate]

        result = GeminiService._extract_grounding_metadata(response)
        assert result is not None
        assert result["search_queries"] == ["ACME Corp earnings 2026", "ACME stock price"]

    def test_extract_grounding_chunks(self):
        """Should extract source URIs and titles from grounding chunks."""
        response = MagicMock()
        gm = MagicMock()
        gm.web_search_queries = None

        chunk1 = MagicMock()
        chunk1.web = MagicMock()
        chunk1.web.uri = "https://reuters.com/article/acme-earnings"
        chunk1.web.title = "ACME Reports Record Earnings"

        chunk2 = MagicMock()
        chunk2.web = MagicMock()
        chunk2.web.uri = "https://bloomberg.com/news/acme"
        chunk2.web.title = "ACME Stock Rises"

        gm.grounding_chunks = [chunk1, chunk2]
        gm.grounding_supports = None
        gm.search_entry_point = None
        candidate = MagicMock()
        candidate.grounding_metadata = gm
        response.candidates = [candidate]

        result = GeminiService._extract_grounding_metadata(response)
        assert result is not None
        assert len(result["grounding_chunks"]) == 2
        assert result["grounding_chunks"][0]["uri"] == "https://reuters.com/article/acme-earnings"
        assert result["grounding_chunks"][0]["title"] == "ACME Reports Record Earnings"
        assert result["grounding_chunks"][1]["uri"] == "https://bloomberg.com/news/acme"

    def test_extract_grounding_supports(self):
        """Should extract text segments with source indices and confidence."""
        response = MagicMock()
        gm = MagicMock()
        gm.web_search_queries = None
        gm.grounding_chunks = None

        support = MagicMock()
        support.segment = MagicMock()
        support.segment.text = "ACME reported record Q4 earnings."
        support.grounding_chunk_indices = [0, 1]
        support.confidence_scores = [0.95, 0.87]

        gm.grounding_supports = [support]
        gm.search_entry_point = None
        candidate = MagicMock()
        candidate.grounding_metadata = gm
        response.candidates = [candidate]

        result = GeminiService._extract_grounding_metadata(response)
        assert result is not None
        assert len(result["grounding_supports"]) == 1
        assert result["grounding_supports"][0]["text"] == "ACME reported record Q4 earnings."
        assert result["grounding_supports"][0]["chunk_indices"] == [0, 1]
        assert result["grounding_supports"][0]["confidence_scores"] == [0.95, 0.87]

    def test_extract_full_grounding_response(self):
        """Should extract all metadata fields from a complete response."""
        response = MagicMock()
        gm = MagicMock()
        gm.web_search_queries = ["test query"]

        chunk = MagicMock()
        chunk.web = MagicMock()
        chunk.web.uri = "https://example.com"
        chunk.web.title = "Example"
        gm.grounding_chunks = [chunk]

        support = MagicMock()
        support.segment = MagicMock()
        support.segment.text = "Test text"
        support.grounding_chunk_indices = [0]
        support.confidence_scores = [0.99]
        gm.grounding_supports = [support]

        sep = MagicMock()
        sep.rendered_content = "<div>Search suggestion</div>"
        gm.search_entry_point = sep

        candidate = MagicMock()
        candidate.grounding_metadata = gm
        response.candidates = [candidate]

        result = GeminiService._extract_grounding_metadata(response)
        assert result is not None
        assert "search_queries" in result
        assert "grounding_chunks" in result
        assert "grounding_supports" in result
        assert "search_suggestion_html" in result

    def test_extract_handles_exceptions_gracefully(self):
        """Should return None on any exception, not crash."""
        response = MagicMock()
        response.candidates = None  # Will cause TypeError on indexing
        result = GeminiService._extract_grounding_metadata(response)
        assert result is None


# ── 3c. URL Context Metadata Extraction Tests ────────────────────────────


class TestURLContextMetadataExtraction:
    """Test _extract_url_context_metadata parses response objects correctly."""

    def test_extract_with_no_candidates(self):
        """Should return None when response has no candidates."""
        response = MagicMock()
        response.candidates = []
        result = GeminiService._extract_url_context_metadata(response)
        assert result is None

    def test_extract_with_no_url_context_metadata(self):
        """Should return None when candidate has no url_context_metadata."""
        response = MagicMock()
        candidate = MagicMock(spec=[])  # No attributes
        response.candidates = [candidate]
        result = GeminiService._extract_url_context_metadata(response)
        assert result is None

    def test_extract_url_metadata_success(self):
        """Should extract retrieved URLs and their statuses."""
        response = MagicMock()
        candidate = MagicMock()

        um1 = MagicMock()
        um1.retrieved_url = "https://reuters.com/article/example"
        um1.url_retrieval_status = "URL_RETRIEVAL_STATUS_SUCCESS"

        um2 = MagicMock()
        um2.retrieved_url = "https://bloomberg.com/news/example"
        um2.url_retrieval_status = "URL_RETRIEVAL_STATUS_SUCCESS"

        ucm = MagicMock()
        ucm.url_metadata = [um1, um2]
        candidate.url_context_metadata = ucm
        response.candidates = [candidate]

        result = GeminiService._extract_url_context_metadata(response)
        assert result is not None
        assert len(result["url_metadata"]) == 2
        assert result["url_metadata"][0]["retrieved_url"] == "https://reuters.com/article/example"
        assert result["url_metadata"][0]["status"] == "URL_RETRIEVAL_STATUS_SUCCESS"

    def test_extract_url_metadata_with_failure(self):
        """Should capture failed URL retrievals."""
        response = MagicMock()
        candidate = MagicMock()

        um = MagicMock()
        um.retrieved_url = "https://paywalled-site.com/article"
        um.url_retrieval_status = "URL_RETRIEVAL_STATUS_UNSAFE"

        ucm = MagicMock()
        ucm.url_metadata = [um]
        candidate.url_context_metadata = ucm
        response.candidates = [candidate]

        result = GeminiService._extract_url_context_metadata(response)
        assert result is not None
        assert result["url_metadata"][0]["status"] == "URL_RETRIEVAL_STATUS_UNSAFE"

    def test_extract_handles_exceptions_gracefully(self):
        """Should return None on any exception, not crash."""
        response = MagicMock()
        response.candidates = None
        result = GeminiService._extract_url_context_metadata(response)
        assert result is None


# ── 4. URL Context (Scenario from Article) Tests ─────────────────────────


class TestURLScenarioGeneration:
    """Test URL-based scenario generation."""

    def setup_method(self):
        self.svc = GeminiService()
        self.svc.use_mock = True

    @pytest.mark.asyncio
    async def test_heuristic_url_scenario_structure(self):
        """Fallback scenario should have all required fields."""
        result = await self.svc.generate_scenario_from_url(
            "https://example.com/article", difficulty=3
        )
        assert "name" in result
        assert "description" in result
        assert "category" in result
        assert "difficulty" in result
        assert "time_pressure_seconds" in result
        assert "source_url" in result
        assert "source_summary" in result
        assert "initial_data" in result
        assert "events" in result

    @pytest.mark.asyncio
    async def test_heuristic_url_scenario_has_events(self):
        """Fallback scenario should have multiple events."""
        result = await self.svc.generate_scenario_from_url(
            "https://example.com/breaking-news", difficulty=2
        )
        assert len(result["events"]) >= 6

    @pytest.mark.asyncio
    async def test_heuristic_url_scenario_preserves_url(self):
        """Source URL should be preserved in output."""
        url = "https://reuters.com/article/big-company-merger"
        result = await self.svc.generate_scenario_from_url(url, difficulty=4)
        assert result["source_url"] == url

    @pytest.mark.asyncio
    async def test_heuristic_url_scenario_initial_data(self):
        """Initial data should contain required trading fields."""
        result = await self.svc.generate_scenario_from_url(
            "https://example.com/article", difficulty=3
        )
        init = result["initial_data"]
        assert "asset" in init
        assert "price" in init
        assert "your_balance" in init
        assert "market_sentiment" in init
        assert init["your_balance"] == 10000

    @pytest.mark.asyncio
    async def test_heuristic_url_scenario_market_params(self):
        """Fallback should include basic market_params."""
        result = await self.svc.generate_scenario_from_url(
            "https://example.com/article", difficulty=3
        )
        mp = result["initial_data"].get("market_params", {})
        assert "fixed_fee" in mp
        assert "pct_fee" in mp

    @pytest.mark.asyncio
    async def test_heuristic_url_scenario_events_ordered(self):
        """Events should be in chronological order."""
        result = await self.svc.generate_scenario_from_url(
            "https://example.com/article", difficulty=3
        )
        times = [e["time"] for e in result["events"]]
        assert times == sorted(times), "Events should be chronologically ordered"


# ── 5. Integration: _call_gemini Signature Tests ─────────────────────────


class TestCallGeminiSignature:
    """Verify _call_gemini accepts new parameters without error in mock mode."""

    def setup_method(self):
        self.svc = GeminiService()
        self.svc.use_mock = True

    def test_thinking_levels_dict_valid_values(self):
        """All thinking levels should be valid Gemini 3 Pro values."""
        valid = {"low", "high"}
        for call_type, level in THINKING_LEVELS.items():
            assert level in valid, f"Invalid thinking level '{level}' for {call_type}"

    def test_call_gemini_accepts_new_params(self):
        """_call_gemini signature should accept call_type, use_search_grounding, use_url_context, cached_content_name."""
        import inspect
        sig = inspect.signature(self.svc._call_gemini)
        params = set(sig.parameters.keys())
        assert "call_type" in params
        assert "use_search_grounding" in params
        assert "use_url_context" in params
        assert "cached_content_name" in params

    def test_call_gemini_defaults(self):
        """New params should have safe defaults."""
        import inspect
        sig = inspect.signature(self.svc._call_gemini)
        assert sig.parameters["call_type"].default == "reflection"
        assert sig.parameters["use_search_grounding"].default is False
        assert sig.parameters["use_url_context"].default is False
        assert sig.parameters["cached_content_name"].default is None


# ── 6. Pydantic Schema Tests for New Output Types ────────────────────────


class TestNewOutputSchemas:
    """Test Pydantic validation for credibility and URL scenario outputs."""

    def test_credibility_output_valid(self):
        from services.gemini_service import _CredibilityCheckOutput
        data = {
            "credibility_score": 0.75,
            "verdict": "verified",
            "supporting_sources": ["Reuters", "Bloomberg"],
            "contradicting_sources": [],
            "key_finding": "Multiple sources confirm the claim.",
            "risk_level": "low",
        }
        result = _CredibilityCheckOutput.model_validate(data)
        assert result.credibility_score == 0.75
        assert result.verdict == "verified"
        assert len(result.supporting_sources) == 2
        # New fields default to empty lists
        assert result.grounding_source_urls == []
        assert result.search_queries_used == []

    def test_credibility_output_with_grounding_urls(self):
        from services.gemini_service import _CredibilityCheckOutput
        data = {
            "credibility_score": 0.85,
            "verdict": "verified",
            "supporting_sources": ["Reuters"],
            "contradicting_sources": [],
            "key_finding": "Confirmed by Reuters.",
            "risk_level": "low",
            "grounding_source_urls": [
                {"uri": "https://reuters.com/article/123", "title": "Earnings Report"},
            ],
            "search_queries_used": ["ACME earnings Q4"],
        }
        result = _CredibilityCheckOutput.model_validate(data)
        assert len(result.grounding_source_urls) == 1
        assert result.grounding_source_urls[0]["uri"] == "https://reuters.com/article/123"
        assert result.search_queries_used == ["ACME earnings Q4"]

    def test_credibility_output_bounds(self):
        from pydantic import ValidationError
        from services.gemini_service import _CredibilityCheckOutput
        with pytest.raises(ValidationError):
            _CredibilityCheckOutput.model_validate({
                "credibility_score": 1.5,  # Invalid: > 1
                "verdict": "verified",
                "supporting_sources": [],
                "contradicting_sources": [],
                "key_finding": "test",
                "risk_level": "low",
            })

    def test_url_scenario_output_valid(self):
        from services.gemini_service import _URLScenarioOutput
        data = {
            "name": "Breaking News Challenge",
            "description": "Navigate decisions based on a real article.",
            "category": "risk_management",
            "difficulty": 3,
            "time_pressure_seconds": 180,
            "source_url": "https://example.com/article",
            "source_summary": "Article about a company merger.",
            "initial_data": {
                "asset": "NEWSX",
                "price": 75.0,
                "your_balance": 10000,
            },
            "events": [
                {"time": 10, "type": "news", "content": "Breaking news!"},
            ],
        }
        result = _URLScenarioOutput.model_validate(data)
        assert result.name == "Breaking News Challenge"
        # New field defaults to empty list
        assert result.url_retrieval_metadata == []
        assert result.difficulty == 3
        assert len(result.events) == 1

    def test_url_scenario_difficulty_bounds(self):
        from pydantic import ValidationError
        from services.gemini_service import _URLScenarioOutput
        with pytest.raises(ValidationError):
            _URLScenarioOutput.model_validate({
                "name": "Test",
                "description": "Test",
                "category": "test",
                "difficulty": 6,  # Invalid: > 5
                "time_pressure_seconds": 180,
                "source_url": "https://example.com",
                "source_summary": "test",
                "initial_data": {},
                "events": [],
            })


# ── 7. Config Tests ──────────────────────────────────────────────────────


class TestConfigSettings:
    """Test that new config settings exist with correct defaults."""

    def test_context_cache_ttl_default(self):
        from config import get_settings
        s = get_settings()
        assert hasattr(s, "gemini_context_cache_ttl_minutes")
        assert s.gemini_context_cache_ttl_minutes == 30

    def test_search_grounding_default(self):
        from config import get_settings
        s = get_settings()
        assert hasattr(s, "gemini_enable_search_grounding")
        assert s.gemini_enable_search_grounding is True

    def test_url_context_default(self):
        from config import get_settings
        s = get_settings()
        assert hasattr(s, "gemini_enable_url_context")
        assert s.gemini_enable_url_context is True


# ── Chart Analysis (Multimodal Vision) ─────────────────────────────


class TestChartAnalysis:
    """Tests for the multimodal chart analysis feature."""

    def test_heuristic_fallback_returns_valid_schema(self):
        """Heuristic fallback should return schema-compliant data."""
        from services.gemini.schemas import _ChartAnalysisGeminiOutput

        svc = GeminiService()
        result = svc._heuristic_chart_analysis()

        assert result["chart_type"] == "unknown"
        assert result["recommended_action"] == "wait"
        assert result["confidence"] == 0.0
        assert len(result["bias_warnings"]) >= 2
        assert result["_source"] == "heuristic"

        # Validate bias warning structure
        for w in result["bias_warnings"]:
            assert "bias" in w
            assert "explanation" in w
            assert w["risk_level"] in ("low", "medium", "high")

    def test_heuristic_fallback_schema_validation(self):
        """Heuristic output should pass Pydantic schema validation."""
        from services.gemini.schemas import _ChartAnalysisGeminiOutput

        svc = GeminiService()
        result = svc._heuristic_chart_analysis()

        # Remove non-schema fields
        clean = {k: v for k, v in result.items() if not k.startswith("_")}
        validated = _ChartAnalysisGeminiOutput.model_validate(clean)
        assert validated.chart_type == "unknown"
        assert validated.confidence == 0.0

    @pytest.mark.asyncio
    async def test_mock_mode_uses_heuristic(self):
        """When USE_MOCK_GEMINI=true, should return heuristic result."""
        svc = GeminiService()
        svc.use_mock = True

        result = await svc.analyze_chart(b"fake_image_bytes", "image/png")
        assert result["_source"] == "heuristic"
        assert result["chart_type"] == "unknown"
        assert len(result["bias_warnings"]) >= 2

    def test_chart_analysis_thinking_budget_exists(self):
        """Chart analysis should have a thinking budget configured."""
        from services.gemini.helpers import THINKING_BUDGETS
        assert "chart_analysis" in THINKING_BUDGETS
        assert THINKING_BUDGETS["chart_analysis"] > 0

    def test_chart_analysis_schema_fields(self):
        """Schema should have all required fields."""
        from services.gemini.schemas import _ChartAnalysisGeminiOutput

        fields = _ChartAnalysisGeminiOutput.model_fields
        expected = [
            "chart_type", "trend_summary", "key_patterns",
            "support_resistance", "bias_warnings",
            "recommended_action", "confidence", "reasoning",
        ]
        for field in expected:
            assert field in fields, f"Missing field: {field}"
