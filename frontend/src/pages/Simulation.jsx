import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { scenariosApi, simulationsApi, streamApi } from '../api/client';
import {
  ComposedChart, Line, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, CartesianGrid
} from 'recharts';
import {
  Clock, TrendingUp, TrendingDown, DollarSign, AlertCircle,
  MessageSquare, Users, Newspaper, ChevronUp, ChevronDown, Minus,
  ShieldCheck, ShieldAlert, ShieldQuestion, Zap, CheckCircle, XCircle,
  Pause, Eye, Globe, MessageCircle, Twitter, Hash, FastForward,
  Activity, Ban, BarChart3, AlertTriangle, Target, Layers,
  Play, Shield, Gauge, Radio, Heart, Repeat2, ArrowRight, X
} from 'lucide-react';
import clsx from 'clsx';
import CoachNudge from '../components/CoachNudge';

/* ─── Helper components ─────────────────────────────────────────── */

const CredibilityBadge = ({ credibility }) => {
  if (credibility == null) return null;
  if (credibility >= 0.7) return (
    <span className="inline-flex items-center gap-1 ml-2 text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 flex-shrink-0">
      <ShieldCheck className="h-3 w-3" /> Reliable
    </span>
  );
  if (credibility >= 0.4) return (
    <span className="inline-flex items-center gap-1 ml-2 text-xs font-semibold px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 flex-shrink-0">
      <ShieldQuestion className="h-3 w-3" /> Uncertain
    </span>
  );
  return (
    <span className="inline-flex items-center gap-1 ml-2 text-xs font-semibold px-2 py-0.5 rounded-full bg-red-100 text-red-800 flex-shrink-0">
      <ShieldAlert className="h-3 w-3" /> Unreliable
    </span>
  );
};

const TOAST_CONFIG = {
  buy:  { icon: TrendingUp,   color: 'bg-emerald-600', label: 'Buy Order Placed' },
  sell: { icon: TrendingDown,  color: 'bg-red-600',    label: 'Sell Order Placed' },
  hold: { icon: Pause,         color: 'bg-blue-600',   label: 'Holding Position' },
  wait: { icon: Eye,           color: 'bg-amber-600',  label: 'Waiting for Info' },
  end:  { icon: CheckCircle,   color: 'bg-purple-600', label: 'Simulation Ended' },
  partial: { icon: AlertTriangle, color: 'bg-orange-600', label: 'Partial Fill' },
  halted: { icon: Ban, color: 'bg-red-700', label: 'Market Halted' },
  limit_placed: { icon: Target, color: 'bg-indigo-600', label: 'Limit Order Placed' },
  stop_placed: { icon: ShieldAlert, color: 'bg-amber-700', label: 'Stop Order Placed' },
  order_filled: { icon: CheckCircle, color: 'bg-emerald-600', label: 'Order Filled' },
};

const SentimentTag = ({ sentiment }) => {
  if (!sentiment || sentiment === 'event') return null;
  const colors = {
    bullish: 'text-emerald-800 bg-emerald-100',
    bearish: 'text-red-800 bg-red-100',
    neutral: 'text-gray-700 bg-gray-100',
    fearful: 'text-orange-800 bg-orange-100',
  };
  return (
    <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${colors[sentiment] || colors.neutral}`}>
      {sentiment}
    </span>
  );
};

const volLabel = (vol) => {
  if (vol == null) return null;
  if (vol < 0.01) return { text: 'Low', color: 'text-emerald-700', bg: 'bg-emerald-50' };
  if (vol < 0.03) return { text: 'Med', color: 'text-amber-700', bg: 'bg-amber-50' };
  return { text: 'High', color: 'text-red-700', bg: 'bg-red-50' };
};

/* ─── News/Social helpers ────────────────────────────────────────── */

const NEWS_SOURCES = ['Reuters', 'Bloomberg', 'MarketWatch', 'CNBC', 'WSJ'];
const UNVERIFIED_SOURCES = ['Anonymous Tip', 'Forum Post', 'Unverified Source', 'Social Media'];
const USERNAMES = ['@trader_mike', '@crypto_whale', '@market_guru', '@wall_st_wolf', '@penny_picker',
  '@bull_runner', '@bear_hunter', '@options_queen', '@value_victor', '@swing_king',
  '@diamond_hands', '@chart_master', '@risk_taker', '@steady_eddie', '@alpha_seeker'];
const AVATAR_COLORS = ['bg-blue-500', 'bg-purple-500', 'bg-pink-500', 'bg-indigo-500',
  'bg-teal-500', 'bg-orange-500', 'bg-cyan-500', 'bg-rose-500', 'bg-emerald-500', 'bg-violet-500'];

function hashStr(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash) + str.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

function getNewsSource(item) {
  if (item.unverified) {
    return UNVERIFIED_SOURCES[hashStr(item.content) % UNVERIFIED_SOURCES.length];
  }
  return NEWS_SOURCES[hashStr(item.content) % NEWS_SOURCES.length];
}

function getSocialUser(content) {
  const h = hashStr(content);
  const username = USERNAMES[h % USERNAMES.length];
  const initials = username.slice(1, 3).toUpperCase();
  const color = AVATAR_COLORS[h % AVATAR_COLORS.length];
  return { username, initials, color };
}

function getEngagement(content, sentiment) {
  const h = hashStr(content);
  const base = sentiment === 'bullish' || sentiment === 'bearish' ? 3 : 1;
  const likes = (h % 40) * base + 2;
  const reposts = Math.floor(likes * 0.3) + (h % 5);
  return { likes, reposts };
}

function relativeTime(itemTime, currentTime) {
  const diff = currentTime - itemTime;
  if (diff <= 0) return 'Just now';
  if (diff < 60) return `${diff}s ago`;
  return `${Math.floor(diff / 60)}m ago`;
}

function isBreaking(itemTime, currentTime) {
  return currentTime - itemTime <= 10;
}

/* ─── Feature badges for briefing ────────────────────────────────── */

const FEATURE_LABELS = {
  order_types_enabled: { label: 'Order Types', icon: Target, desc: 'Limit & stop orders available' },
  halts_enabled: { label: 'Circuit Breakers', icon: Ban, desc: 'Trading halts on big moves' },
  volatility_clustering: { label: 'Volatility Clustering', icon: Activity, desc: 'GARCH-style vol surges' },
  crowd_model_enabled: { label: 'Crowd Behavior', icon: Users, desc: 'Herd sentiment affects prices' },
  margin_enabled: { label: 'Margin Trading', icon: Layers, desc: 'Leverage & margin calls' },
  news_latency_enabled: { label: 'News Delays', icon: Clock, desc: 'Breaking news arrives late' },
  time_pressure_fills: { label: 'Time Pressure', icon: Gauge, desc: 'Prices move while you decide' },
  secondary_asset: { label: 'Correlated Asset', icon: BarChart3, desc: 'A second asset moves in tandem' },
};

function getActiveFeatures(marketParams) {
  if (!marketParams) return [];
  const features = [];
  for (const [key, meta] of Object.entries(FEATURE_LABELS)) {
    if (marketParams[key]) features.push(meta);
  }
  if (marketParams.base_spread_pct > 0.003) {
    features.push({ label: 'Wide Spreads', icon: ArrowRight, desc: 'Bid-ask spread is significant' });
  }
  if (marketParams.fixed_fee > 1 || marketParams.pct_fee > 0.001) {
    features.push({ label: 'High Fees', icon: DollarSign, desc: 'Transaction costs eat into profits' });
  }
  if (marketParams.max_drawdown_pct) {
    features.push({ label: 'Risk Limits', icon: Shield, desc: `Max drawdown: ${(marketParams.max_drawdown_pct * 100).toFixed(0)}%` });
  }
  return features;
}

const DIFFICULTY_LABELS = ['', 'Beginner', 'Easy', 'Medium', 'Hard', 'Extreme'];
const DIFFICULTY_COLORS = ['', 'text-emerald-600 bg-emerald-100', 'text-blue-600 bg-blue-100',
  'text-amber-600 bg-amber-100', 'text-orange-600 bg-orange-100', 'text-red-600 bg-red-100'];

/* ─── Main component ────────────────────────────────────────────── */

export default function Simulation() {
  const { scenarioId } = useParams();
  const navigate = useNavigate();

  const [scenario, setScenario] = useState(null);
  const [simulation, setSimulation] = useState(null);
  const [state, setState] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Phase: briefing → running → completed
  const [phase, setPhase] = useState('briefing');

  const [timeElapsed, setTimeElapsed] = useState(0);
  const [isRunning, setIsRunning] = useState(false);

  const [decisionAmount, setDecisionAmount] = useState(100);
  const [confidence, setConfidence] = useState(3);
  const [rationale, setRationale] = useState('');
  const [activePanel, setActivePanel] = useState('news');
  const [infoViewTimes, setInfoViewTimes] = useState({});

  // Order type state
  const [orderType, setOrderType] = useState('market');
  const [limitPrice, setLimitPrice] = useState('');
  const [stopPrice, setStopPrice] = useState('');

  // Coach nudge state
  const [coachNudge, setCoachNudge] = useState(null);

  // Challenge mode state
  const [challengeResult, setChallengeResult] = useState(null);
  const [challenging, setChallenging] = useState(false);

  // Wait/skip state
  const [waitPickerOpen, setWaitPickerOpen] = useState(false);
  const [skipping, setSkipping] = useState(false);

  // Toast notifications
  const [toasts, setToasts] = useState([]);

  // Decision history
  const [decisionHistory, setDecisionHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);

  const decisionStartTime = useRef(Date.now());
  const panelSwitchTime = useRef(Date.now());
  const sseReaderRef = useRef(null);

  /* ─── Effects ───────────────────────────────────────────────── */

  // Load scenario data on mount (but don't start simulation yet)
  useEffect(() => {
    loadScenario();
    return () => {
      if (sseReaderRef.current) sseReaderRef.current.cancel();
    };
  }, [scenarioId]);

  // Connect SSE when simulation is running (with auto-reconnect)
  useEffect(() => {
    if (!simulation || !isRunning) return;
    let cancelled = false;
    let retryCount = 0;
    const MAX_RETRIES = 3;

    const connectSSE = async () => {
      try {
        const response = await streamApi.simulationStream(simulation.id);
        const reader = response.body.getReader();
        sseReaderRef.current = reader;
        const decoder = new TextDecoder();
        let buffer = '';
        retryCount = 0; // reset on successful connect

        while (!cancelled) {
          const { done, value } = await reader.read();
          if (done) {
            // Stream ended unexpectedly — try to reconnect
            if (!cancelled && retryCount < MAX_RETRIES) {
              retryCount++;
              console.warn(`SSE stream ended, reconnecting (attempt ${retryCount}/${MAX_RETRIES})...`);
              await new Promise(r => setTimeout(r, 1000 * retryCount));
              return connectSSE();
            }
            break;
          }
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop();

          let eventType = 'message';
          for (const line of lines) {
            if (line.startsWith('event: ')) {
              eventType = line.slice(7).trim();
            } else if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                if (eventType === 'coach') {
                  setCoachNudge(data);
                } else if (data.status === 'time_expired' || eventType === 'complete') {
                  setIsRunning(false);
                  handleComplete();
                } else if (data.status === 'in_progress') {
                  setState(data);
                  setTimeElapsed(data.time_elapsed);
                } else if (data.status && data.status !== 'in_progress') {
                  setIsRunning(false);
                }
              } catch {}
              eventType = 'message';
            }
          }
        }
      } catch (err) {
        if (!cancelled) {
          if (retryCount < MAX_RETRIES) {
            retryCount++;
            console.warn(`SSE connection failed, reconnecting (attempt ${retryCount}/${MAX_RETRIES})...`);
            await new Promise(r => setTimeout(r, 1000 * retryCount));
            return connectSSE();
          }
          console.error('SSE failed after retries, falling back to polling:', err);
          startPollingFallback();
        }
      }
    };
    connectSSE();
    return () => {
      cancelled = true;
      if (sseReaderRef.current) sseReaderRef.current.cancel();
    };
  }, [simulation, isRunning]);

  const startPollingFallback = () => {
    const timer = setInterval(() => {
      setTimeElapsed((prev) => {
        const newTime = prev + 1;
        if (newTime >= scenario.time_pressure_seconds) {
          clearInterval(timer);
          setIsRunning(false);
          handleComplete();
          return prev;
        }
        return newTime;
      });
    }, 1000);
    return () => clearInterval(timer);
  };

  useEffect(() => {
    if (!simulation || !isRunning || !state) return;
    if (state.time_elapsed !== undefined && Math.abs(state.time_elapsed - timeElapsed) < 3) return;
    const fetchState = async () => {
      try {
        const res = await simulationsApi.getState(simulation.id, timeElapsed);
        setState(res.data);
      } catch (err) {
        console.error('Failed to fetch state:', err);
      }
    };
    fetchState();
  }, [timeElapsed, simulation, isRunning]);

  useEffect(() => {
    return () => {
      const duration = (Date.now() - panelSwitchTime.current) / 1000;
      setInfoViewTimes((prev) => ({
        ...prev,
        [activePanel]: (prev[activePanel] || 0) + duration
      }));
    };
  }, [activePanel]);

  /* ─── Keyboard Shortcuts ─────────────────────────────────────── */

  useEffect(() => {
    if (phase !== 'running' || !simulation) return;
    const handler = (e) => {
      // Ignore if user is typing in an input/textarea
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      switch (e.key.toLowerCase()) {
        case 'b': makeDecision('buy'); break;
        case 's': makeDecision('sell'); break;
        case 'h': makeDecision('hold'); break;
        case 'w': case ' ': e.preventDefault(); setWaitPickerOpen(v => !v); break;
        case 'd': setShowHistory(v => !v); break;
        default: break;
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [phase, simulation, decisionAmount, confidence, rationale, orderType, limitPrice, stopPrice, state]);

  /* ─── Actions ───────────────────────────────────────────────── */

  const loadScenario = async () => {
    try {
      const scenarioRes = await scenariosApi.get(scenarioId);
      setScenario(scenarioRes.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load scenario');
    } finally {
      setLoading(false);
    }
  };

  const startSimulation = async () => {
    try {
      setLoading(true);
      const simRes = await simulationsApi.start(scenarioId);
      setSimulation(simRes.data);
      setState(simRes.data);
      setPhase('running');
      setIsRunning(true);
      decisionStartTime.current = Date.now();
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to start simulation');
    } finally {
      setLoading(false);
    }
  };

  const makeDecision = async (decisionType) => {
    if (!simulation) return;
    const mc = state?.market_conditions;
    if (mc?.halted && (decisionType === 'buy' || decisionType === 'sell')) {
      showToast('halted', ' — Trading suspended');
      return;
    }
    const timeSpent = (Date.now() - decisionStartTime.current) / 1000;
    const payload = {
      decision_type: decisionType,
      amount: decisionType !== 'hold' && decisionType !== 'wait' ? decisionAmount : null,
      confidence_level: confidence,
      time_spent_seconds: timeSpent,
      rationale: rationale.trim() || null,
      info_viewed: Object.entries(infoViewTimes).map(([panel, duration]) => ({
        panel, view_duration_seconds: duration, timestamp: timeElapsed
      })),
      order_type: orderType,
      limit_price: orderType === 'limit' && limitPrice ? parseFloat(limitPrice) : null,
      stop_price: orderType === 'stop' && stopPrice ? parseFloat(stopPrice) : null,
      time_elapsed: timeElapsed,
    };
    try {
      await simulationsApi.makeDecision(simulation.id, payload);
      // Track in decision history
      setDecisionHistory(prev => [...prev, {
        type: decisionType,
        amount: decisionAmount,
        price: state?.current_price,
        time: timeElapsed,
        orderType,
        rationale: rationale.trim() || null,
      }]);
      decisionStartTime.current = Date.now();
      setInfoViewTimes({});
      setRationale('');
      setChallengeResult(null);
      if (orderType === 'limit' && (decisionType === 'buy' || decisionType === 'sell')) {
        showToast('limit_placed', ` — ${decisionType.toUpperCase()} @ $${limitPrice}`);
      } else if (orderType === 'stop' && (decisionType === 'buy' || decisionType === 'sell')) {
        showToast('stop_placed', ` — trigger @ $${stopPrice}`);
      } else {
        const amtLabel = (decisionType === 'buy' || decisionType === 'sell') ? ` — $${decisionAmount}` : '';
        showToast(decisionType, amtLabel);
      }
      setOrderType('market');
      setLimitPrice('');
      setStopPrice('');
    } catch (err) {
      console.error('Failed to make decision:', err);
      showToast('halted', err.response?.data?.detail || 'Decision failed — try again');
    }
  };

  const handleWaitWithDuration = async (seconds) => {
    if (!simulation) return;
    setWaitPickerOpen(false);
    setSkipping(true);
    const timeSpent = (Date.now() - decisionStartTime.current) / 1000;
    try {
      await simulationsApi.makeDecision(simulation.id, {
        decision_type: 'wait', amount: null, confidence_level: confidence,
        time_spent_seconds: timeSpent,
        rationale: rationale.trim() || `Waiting ${seconds}s for more data`,
        info_viewed: Object.entries(infoViewTimes).map(([panel, duration]) => ({
          panel, view_duration_seconds: duration, timestamp: timeElapsed
        }))
      });
    } catch {}
    if (sseReaderRef.current) { sseReaderRef.current.cancel(); sseReaderRef.current = null; }
    setIsRunning(false);
    try {
      const res = await simulationsApi.skipTime(simulation.id, seconds);
      setState(res.data);
      setTimeElapsed(res.data.time_elapsed);
      if (res.data.time_remaining <= 0) { handleComplete(); return; }
    } catch (err) { console.error('Skip time failed:', err); }
    decisionStartTime.current = Date.now();
    setInfoViewTimes({});
    setRationale('');
    setChallengeResult(null);
    setSkipping(false);
    showToast('wait', ` — skipped ${seconds}s`);
    setTimeout(() => setIsRunning(true), 100);
  };

  const handleChallenge = async (decisionType) => {
    if (!simulation || !rationale.trim()) return;
    setChallenging(true);
    try {
      const res = await simulationsApi.challenge(simulation.id, {
        decision_type: decisionType,
        amount: decisionType !== 'hold' && decisionType !== 'wait' ? decisionAmount : null,
        rationale: rationale.trim(),
      });
      setChallengeResult({ ...res.data, decisionType });
    } catch (err) {
      console.error('Challenge failed:', err);
      setChallengeResult({
        reasoning_score: 0,
        feedback: 'AI analysis is temporarily unavailable. You can still proceed with your decision.',
        decisionType,
      });
    } finally { setChallenging(false); }
  };

  const handleComplete = async () => {
    if (!simulation) return;
    try {
      showToast('end', '');
      await simulationsApi.complete(simulation.id);
      navigate(`/reflection/${simulation.id}`);
    } catch (err) { console.error('Failed to complete:', err); }
  };

  const handleAbandon = async () => {
    if (!simulation) return;
    try {
      await simulationsApi.abandon(simulation.id);
      navigate('/dashboard');
    } catch (err) { console.error('Failed to abandon:', err); }
  };

  const switchPanel = (panel) => {
    const duration = (Date.now() - panelSwitchTime.current) / 1000;
    setInfoViewTimes((prev) => ({ ...prev, [activePanel]: (prev[activePanel] || 0) + duration }));
    panelSwitchTime.current = Date.now();
    setActivePanel(panel);
  };

  const showToast = (type, detail) => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, type, detail }]);
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 3500);
  };

  /* ─── Loading / Error states ────────────────────────────────── */

  if (loading && !scenario) {
    return (
      <div className="min-h-screen bg-brand-cream flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-brand-navy mx-auto mb-4" />
          <p className="text-gray-800 text-lg font-medium">Loading scenario...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-brand-cream flex items-center justify-center px-4">
        <div className="card max-w-md text-center">
          <div className="w-16 h-16 rounded-2xl bg-red-100 flex items-center justify-center mx-auto mb-4">
            <AlertCircle className="h-8 w-8 text-red-600" />
          </div>
          <h2 className="text-xl font-bold text-gray-900 mb-2">Unable to Load</h2>
          <p className="text-gray-500 mb-6 text-sm" role="alert">{error}</p>
          <div className="flex justify-center gap-3">
            <button
              onClick={() => { setError(null); setLoading(true); loadScenario(); }}
              className="btn btn-secondary"
            >
              Try Again
            </button>
            <button onClick={() => navigate('/dashboard')} className="btn btn-primary">
              Return to Dashboard
            </button>
          </div>
        </div>
      </div>
    );
  }

  /* ─── Briefing Screen ──────────────────────────────────────── */

  if (phase === 'briefing' && scenario) {
    const mp = scenario.initial_data?.market_params || {};
    const features = getActiveFeatures(mp);
    const initialData = scenario.initial_data || {};
    const holdings = initialData.holdings || {};
    const hasHoldings = Object.keys(holdings).length > 0;

    return (
      <div className="min-h-screen bg-brand-cream flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl border border-gray-200 shadow-xl max-w-2xl w-full overflow-hidden">
          {/* Header */}
          <div className="bg-brand-navy p-6 text-white">
            <div className="flex items-center justify-between mb-3">
              <h1 className="text-2xl font-black">{scenario.name}</h1>
              <span className={clsx(
                'text-xs font-black px-3 py-1.5 rounded-full uppercase tracking-wide',
                DIFFICULTY_COLORS[scenario.difficulty] || 'text-gray-600 bg-gray-100'
              )}>
                {DIFFICULTY_LABELS[scenario.difficulty] || 'Unknown'}
              </span>
            </div>
            <p className="text-white/80 text-sm leading-relaxed">{scenario.description}</p>
          </div>

          {/* Starting Conditions */}
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-4">Starting Conditions</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-gray-50 rounded-xl p-3.5 text-center">
                <p className="text-xs font-bold text-gray-400 uppercase mb-1">Asset</p>
                <p className="text-lg font-black text-gray-900">{initialData.asset}</p>
              </div>
              <div className="bg-gray-50 rounded-xl p-3.5 text-center">
                <p className="text-xs font-bold text-gray-400 uppercase mb-1">Price</p>
                <p className="text-lg font-black text-gray-900">${initialData.price?.toFixed(2)}</p>
              </div>
              <div className="bg-gray-50 rounded-xl p-3.5 text-center">
                <p className="text-xs font-bold text-gray-400 uppercase mb-1">Cash</p>
                <p className="text-lg font-black text-emerald-700">${initialData.your_balance?.toLocaleString()}</p>
              </div>
              <div className="bg-gray-50 rounded-xl p-3.5 text-center">
                <p className="text-xs font-bold text-gray-400 uppercase mb-1">Time</p>
                <p className="text-lg font-black text-gray-900">{Math.floor(scenario.time_pressure_seconds / 60)}:{(scenario.time_pressure_seconds % 60).toString().padStart(2, '0')}</p>
              </div>
            </div>
            {hasHoldings && (
              <div className="mt-3 p-3 bg-blue-50 rounded-xl border border-blue-200">
                <p className="text-sm font-bold text-blue-800">
                  You already own: {Object.entries(holdings).map(([asset, qty]) => `${qty} shares of ${asset}`).join(', ')}
                </p>
              </div>
            )}
          </div>

          {/* Market Features */}
          {features.length > 0 && (
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-4">Active Market Features</h2>
              <div className="flex flex-wrap gap-2">
                {features.map((feat, i) => (
                  <div key={i} className="group relative flex items-center gap-2 bg-gray-50 border border-gray-200 rounded-xl px-3.5 py-2.5 hover:bg-gray-100 transition-colors cursor-default">
                    <feat.icon className="h-4 w-4 text-brand-navy flex-shrink-0" />
                    <span className="text-sm font-bold text-gray-800">{feat.label}</span>
                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-1.5 bg-gray-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                      {feat.desc}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Market Sentiment */}
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-3">Market Mood</h2>
            <div className="flex items-center gap-3">
              <span className={clsx(
                'text-sm font-black px-4 py-2 rounded-full capitalize',
                initialData.market_sentiment === 'bullish' ? 'bg-emerald-100 text-emerald-800' :
                initialData.market_sentiment === 'bearish' ? 'bg-red-100 text-red-800' :
                'bg-gray-100 text-gray-700'
              )}>
                {initialData.market_sentiment || 'Neutral'}
              </span>
              {initialData.news_headlines?.[0] && (
                <p className="text-sm text-gray-600 italic">"{initialData.news_headlines[0].content}"</p>
              )}
            </div>
          </div>

          {/* Start Button */}
          <div className="p-6 flex items-center justify-between">
            <button onClick={() => navigate('/dashboard')} className="text-sm font-semibold text-gray-400 hover:text-gray-700 transition-colors">
              Back to Dashboard
            </button>
            <button onClick={startSimulation} disabled={loading}
              className="flex items-center gap-3 bg-brand-navy text-white px-8 py-4 rounded-xl font-bold text-lg hover:bg-brand-navy-light transition-all shadow-lg hover:shadow-xl active:scale-[0.98]">
              <Play className="h-6 w-6" />
              {loading ? 'Starting...' : 'Begin Simulation'}
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!state || !scenario) return null;

  /* ─── Derived values ────────────────────────────────────────── */

  const timeRemaining = scenario.time_pressure_seconds - timeElapsed;
  const priceChange = state.price_history.length > 1
    ? ((state.current_price - state.price_history[0]) / state.price_history[0]) * 100 : 0;
  const entryPrice = state.price_history[0] || state.current_price;
  const mc = state.market_conditions;
  const isHalted = mc?.halted;
  const portfolio = state.portfolio || {};
  const pendingOrders = state.pending_orders || portfolio.pending_orders || [];
  const fees = portfolio.cumulative_fees || 0;
  const drawdownPct = mc?.drawdown_pct || 0;
  const marginStatus = mc?.margin_status;
  const crowdSentiment = mc?.crowd_sentiment;
  const macro = mc?.macro;
  const orderTypesEnabled = scenario.initial_data?.market_params?.order_types_enabled;

  // Fees preview
  const feePreview = mc ? (() => {
    const spread = mc.spread_pct || 0;
    const estSpread = decisionAmount * spread / 2;
    const fixedFee = scenario.initial_data?.market_params?.fixed_fee || 0;
    const pctFee = scenario.initial_data?.market_params?.pct_fee || 0;
    const totalFee = fixedFee + decisionAmount * pctFee;
    return { spread: estSpread, fees: totalFee, total: estSpread + totalFee };
  })() : null;

  // Chart data
  const historical = (state.historical_prices || []).map((price, i, arr) => ({
    time: `H${i + 1}`, price: Number(price.toFixed(2)), label: `Day -${arr.length - i}`,
  }));
  const live = state.price_history.map((price, i) => ({
    time: `${i}s`, price: Number(price.toFixed(2)), label: `${i}s`,
  }));
  const chartData = [...historical, ...live];

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.[0]) return null;
    const price = payload[0].value;
    const change = ((price - entryPrice) / entryPrice * 100);
    const item = chartData.find(d => d.time === label);
    return (
      <div className="bg-white border border-gray-300 rounded-lg px-3 py-2 shadow-xl">
        <p className="text-xs text-gray-500 mb-0.5">{item?.label || label}</p>
        <p className="text-base font-bold text-gray-900">${price.toFixed(2)}</p>
        <p className={`text-sm font-semibold ${change >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
          {change >= 0 ? '+' : ''}{change.toFixed(2)}%
        </p>
      </div>
    );
  };

  /* ─── Render ────────────────────────────────────────────────── */

  return (
    <div className="min-h-screen bg-brand-cream p-4 lg:p-6">
      {/* Coach Nudge */}
      <CoachNudge nudge={coachNudge} onDismiss={() => setCoachNudge(null)} />

      {/* Toast Notifications */}
      <div className="fixed top-4 right-4 z-50 space-y-2 pointer-events-none">
        {toasts.map((toast) => {
          const cfg = TOAST_CONFIG[toast.type] || TOAST_CONFIG.hold;
          const Icon = cfg.icon;
          return (
            <div key={toast.id} className={`${cfg.color} text-white px-5 py-3 rounded-xl shadow-xl flex items-center gap-3 animate-slide-in-right pointer-events-auto min-w-[240px]`}>
              <Icon className="h-5 w-5 flex-shrink-0" />
              <div>
                <p className="font-bold text-sm">{cfg.label}</p>
                {toast.detail && <p className="text-xs opacity-90">{toast.detail}</p>}
              </div>
            </div>
          );
        })}
      </div>

      {/* Skip overlay */}
      {skipping && (
        <div className="fixed inset-0 bg-black/30 backdrop-blur-sm z-40 flex items-center justify-center">
          <div className="bg-white rounded-2xl px-10 py-8 shadow-2xl flex items-center gap-5">
            <FastForward className="h-8 w-8 text-brand-navy animate-pulse" />
            <p className="text-gray-900 font-bold text-xl">Fast-forwarding...</p>
          </div>
        </div>
      )}

      {/* Halt banner */}
      {isHalted && (
        <div role="alert" className="fixed top-0 left-0 right-0 z-30 bg-red-600 text-white py-3 px-4 text-center font-bold text-base flex items-center justify-center gap-3 shadow-xl">
          <Ban className="h-6 w-6" />
          TRADING HALTED — Circuit breaker triggered
        </div>
      )}

      {/* ─── Header ─────────────────────────────────────────────── */}
      <div className={clsx('flex items-center justify-between gap-3 mb-5 flex-wrap', isHalted && 'mt-12')}>
        <div className="min-w-0">
          <h1 className="text-lg sm:text-2xl font-bold text-gray-900 truncate">{scenario.name}</h1>
          <p className="text-gray-500 text-xs sm:text-sm mt-0.5">
            <span className="font-semibold text-gray-700">${portfolio.cash?.toFixed(2) || '0.00'}</span> cash available
          </p>
        </div>
        <div className={clsx(
          'flex items-center gap-2 px-4 sm:px-5 py-2 sm:py-3 rounded-xl font-mono shadow-md flex-shrink-0',
          timeRemaining <= 30
            ? 'bg-red-600 text-white'
            : 'bg-white text-gray-900 border border-gray-200'
        )}>
          <Clock className="h-4 w-4 sm:h-5 sm:w-5" />
          <span className="text-xl sm:text-2xl font-bold tracking-tight">
            {Math.floor(timeRemaining / 60)}:{(timeRemaining % 60).toString().padStart(2, '0')}
          </span>
        </div>
      </div>

      {/* ─── Market Conditions Ticker ───────────────────────────── */}
      {mc && (
        <div className="flex flex-wrap items-center gap-2 mb-5 p-3 bg-white rounded-xl border border-gray-200 shadow-md">
          {/* Bid / Ask */}
          {mc.bid != null && (
            <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-50 rounded-lg">
              <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">Bid</span>
              <span className="text-sm font-bold text-emerald-700">${mc.bid.toFixed(2)}</span>
              <span className="text-gray-300">/</span>
              <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">Ask</span>
              <span className="text-sm font-bold text-red-700">${mc.ask.toFixed(2)}</span>
            </div>
          )}

          {/* Spread */}
          {mc.spread_pct != null && (
            <div className={clsx(
              'px-3 py-1.5 rounded-lg text-xs font-bold',
              mc.spread_pct < 0.005 ? 'bg-emerald-100 text-emerald-800' :
              mc.spread_pct < 0.015 ? 'bg-amber-100 text-amber-800' :
              'bg-red-100 text-red-800'
            )}>
              Spread {(mc.spread_pct * 100).toFixed(2)}%
            </div>
          )}

          {/* Volatility */}
          {mc.volatility != null && (() => {
            const vl = volLabel(mc.volatility);
            return vl && (
              <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold ${vl.bg} ${vl.color}`}>
                <Activity className="h-3.5 w-3.5" />
                {vl.text} Vol
              </div>
            );
          })()}

          {/* Crowd */}
          {crowdSentiment != null && (
            <div className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-50 rounded-lg">
              <Users className="h-3.5 w-3.5 text-gray-500" />
              <span className={clsx(
                'text-xs font-bold',
                crowdSentiment > 0.6 ? 'text-emerald-700' :
                crowdSentiment < 0.4 ? 'text-red-700' : 'text-amber-700'
              )}>
                {Math.round(crowdSentiment * 100)}% buying
              </span>
            </div>
          )}

          {/* Fees */}
          {fees > 0.01 && (
            <div className="flex items-center gap-1 px-3 py-1.5 bg-gray-50 rounded-lg text-xs font-bold text-gray-700">
              <DollarSign className="h-3.5 w-3.5 text-gray-500" />
              Fees: ${fees.toFixed(2)}
            </div>
          )}

          {/* Drawdown */}
          {drawdownPct > 0.02 && (
            <div className={clsx(
              'px-3 py-1.5 rounded-lg text-xs font-bold',
              drawdownPct < 0.10 ? 'bg-amber-100 text-amber-800' :
              drawdownPct < 0.20 ? 'bg-orange-100 text-orange-800' :
              'bg-red-100 text-red-800'
            )}>
              Drawdown {(drawdownPct * 100).toFixed(1)}%
            </div>
          )}

          {/* Margin */}
          {marginStatus && marginStatus !== 'ok' && (
            <div className={clsx(
              'px-3 py-1.5 rounded-lg text-xs font-black uppercase tracking-wide',
              marginStatus === 'margin_call' ? 'bg-red-600 text-white' : 'bg-orange-100 text-orange-800'
            )}>
              {marginStatus === 'margin_call' ? 'MARGIN CALL' : 'Margin Warning'}
            </div>
          )}

          {isHalted && (
            <div className="px-3 py-1.5 rounded-lg text-xs font-black bg-red-600 text-white uppercase tracking-wide animate-pulse">
              HALTED
            </div>
          )}
        </div>
      )}

      {/* ─── Main Grid ──────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">

        {/* ─── Price Chart Card ──────────────────────────────────── */}
        <div className="lg:col-span-2 bg-white rounded-2xl p-4 sm:p-6 border border-gray-200 shadow-md">
          <div className="flex items-center justify-between mb-5 flex-wrap gap-3">
            <div className="min-w-0">
              <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-1">{scenario.initial_data?.asset}</p>
              <div className="flex items-center gap-2 sm:gap-3">
                <span className="text-2xl sm:text-4xl font-black text-gray-900">${state.current_price.toFixed(2)}</span>
                <span className={clsx(
                  'flex items-center text-xs sm:text-sm font-bold px-2 sm:px-3 py-1 rounded-full',
                  priceChange >= 0 ? 'text-emerald-800 bg-emerald-100' : 'text-red-800 bg-red-100'
                )}>
                  {priceChange >= 0 ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                  {Math.abs(priceChange).toFixed(2)}%
                </span>
              </div>
            </div>
            <div className="text-right">
              <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-1">Portfolio</p>
              <p className="text-lg sm:text-2xl font-black text-gray-900">${(portfolio.total_value || 0).toFixed(2)}</p>
              {Object.keys(portfolio.holdings || {}).length > 0 && (
                <p className="text-xs text-gray-500 mt-0.5">
                  {Object.entries(portfolio.holdings).map(([k, v]) => `${v} ${k}`).join(', ')}
                </p>
              )}
            </div>
          </div>

          <div className="h-48 sm:h-72">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData}>
                <defs>
                  <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={priceChange >= 0 ? '#10b981' : '#ef4444'} stopOpacity={0.2} />
                    <stop offset="95%" stopColor={priceChange >= 0 ? '#10b981' : '#ef4444'} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis
                  dataKey="time" stroke="#9ca3af"
                  tick={{ fontSize: 12, fill: '#6b7280', fontWeight: 500 }}
                  interval="preserveStartEnd"
                  axisLine={{ stroke: '#d1d5db' }}
                />
                <YAxis
                  stroke="#9ca3af"
                  domain={['auto', 'auto']}
                  tick={{ fontSize: 12, fill: '#6b7280', fontWeight: 500 }}
                  axisLine={{ stroke: '#d1d5db' }}
                  tickFormatter={(val) => `$${val}`}
                />
                <Tooltip content={<CustomTooltip />} />
                {historical.length > 0 && (
                  <ReferenceLine x="0s" stroke="#374151" strokeDasharray="3 3"
                    label={{ value: 'Start', fill: '#374151', fontSize: 11, fontWeight: 600 }} />
                )}
                <ReferenceLine y={entryPrice} stroke="#9ca3af" strokeDasharray="5 5"
                  label={{ value: `Entry $${entryPrice.toFixed(2)}`, fill: '#9ca3af', fontSize: 11, position: 'right' }} />
                {mc?.bid != null && (
                  <>
                    <ReferenceLine y={mc.bid} stroke="#10b981" strokeDasharray="2 4" strokeOpacity={0.5} />
                    <ReferenceLine y={mc.ask} stroke="#ef4444" strokeDasharray="2 4" strokeOpacity={0.5} />
                  </>
                )}
                <Area type="monotone" dataKey="price" fill="url(#priceGradient)" stroke="none" />
                <Line type="monotone" dataKey="price"
                  stroke={priceChange >= 0 ? '#059669' : '#dc2626'}
                  strokeWidth={2.5} dot={false}
                  activeDot={{ r: 5, stroke: '#1f2937', strokeWidth: 2, fill: 'white' }} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {/* Pending orders */}
          {pendingOrders.length > 0 && (
            <div className="mt-4 pt-4 border-t border-gray-200">
              <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">Pending Orders</p>
              <div className="flex flex-wrap gap-2">
                {pendingOrders.map((ord, i) => (
                  <div key={i} className="text-xs bg-indigo-50 border border-indigo-200 rounded-lg px-3 py-2 flex items-center gap-2 font-semibold text-indigo-800">
                    <Target className="h-3.5 w-3.5" />
                    {ord.order_type?.toUpperCase()} {ord.decision_type?.toUpperCase()} ${ord.amount?.toFixed(0)} @ ${(ord.limit_price || ord.stop_price || 0).toFixed(2)}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ─── Info Panels Card ──────────────────────────────────── */}
        <div className="bg-white rounded-2xl border border-gray-200 shadow-md flex flex-col">
          {/* Tabs */}
          <div className="flex border-b border-gray-200">
            {[
              { id: 'news', icon: Newspaper, label: 'News' },
              { id: 'social', icon: Users, label: 'Social' },
              ...(macro ? [{ id: 'macro', icon: BarChart3, label: 'Macro' }] : []),
            ].map((tab) => (
              <button key={tab.id} onClick={() => switchPanel(tab.id)}
                className={clsx(
                  'flex-1 flex items-center justify-center gap-2 py-3.5 text-sm font-semibold border-b-2 transition-colors',
                  activePanel === tab.id
                    ? 'border-brand-navy text-brand-navy bg-gray-50'
                    : 'border-transparent text-gray-400 hover:text-gray-700'
                )}
              >
                <tab.icon className="h-4 w-4" />
                {tab.label}
              </button>
            ))}
          </div>

          {/* Panel content */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3 max-h-[260px] sm:max-h-[360px]">

            {/* NEWS — Enhanced */}
            {activePanel === 'news' && (
              state.available_info?.news?.map((item, i) => {
                const source = getNewsSource(item);
                const breaking = isBreaking(item.time, timeElapsed);
                const timeAgo = relativeTime(item.time, timeElapsed);
                return (
                  <div key={i} className={clsx(
                    'p-3.5 rounded-xl border',
                    breaking ? 'bg-red-50 border-red-300' : 'bg-gray-50 border-gray-200'
                  )}>
                    <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                      <Globe className="h-4 w-4 text-blue-600 flex-shrink-0" />
                      <span className="text-xs font-bold text-blue-700 uppercase tracking-wide">{source}</span>
                      {breaking && (
                        <span className="text-xs font-black px-2 py-0.5 rounded-full bg-red-600 text-white uppercase animate-pulse">
                          Breaking
                        </span>
                      )}
                      {item.unverified && (
                        <span className="text-xs font-black px-2 py-0.5 rounded-full bg-orange-500 text-white uppercase">
                          Unverified
                        </span>
                      )}
                      <CredibilityBadge credibility={item.credibility} />
                    </div>
                    <p className={clsx(
                      'text-sm leading-relaxed',
                      breaking ? 'text-gray-900 font-bold' : 'text-gray-800 font-medium'
                    )}>{item.content}</p>
                    <div className="flex items-center gap-3 mt-2">
                      <span className="text-xs font-semibold text-gray-400">{timeAgo}</span>
                      {item.delayed && (
                        <span className="text-xs font-bold text-orange-600 flex items-center gap-1">
                          <Clock className="h-3 w-3" /> Delayed
                        </span>
                      )}
                    </div>
                  </div>
                );
              })
            )}

            {/* SOCIAL — Enhanced */}
            {activePanel === 'social' && (
              <>
                {crowdSentiment != null && (
                  <div className="p-3.5 bg-gray-50 rounded-xl border border-gray-200">
                    <div className="flex items-center gap-3">
                      <Users className="h-5 w-5 text-gray-500 flex-shrink-0" />
                      <div className="flex-1">
                        <p className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-1">Market Crowd</p>
                        <div className="flex items-center gap-3">
                          <div className="flex-1 bg-gray-200 rounded-full h-3 overflow-hidden">
                            <div
                              className={clsx(
                                'h-full rounded-full transition-all duration-700',
                                crowdSentiment > 0.6 ? 'bg-emerald-500' :
                                crowdSentiment < 0.4 ? 'bg-red-500' : 'bg-amber-500'
                              )}
                              style={{ width: `${crowdSentiment * 100}%` }}
                            />
                          </div>
                          <span className="text-sm font-black text-gray-900 w-20 text-right">
                            {Math.round(crowdSentiment * 100)}% buy
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                {state.available_info?.social?.map((item, i) => {
                  const user = getSocialUser(item.content);
                  const engagement = getEngagement(item.content, item.sentiment);
                  const timeAgo = relativeTime(item.time, timeElapsed);
                  const isPlatformX = hashStr(item.content) % 3 !== 0; // ~66% X, ~33% Reddit
                  return (
                    <div key={i} className="p-3.5 bg-gray-50 rounded-xl border border-gray-200">
                      <div className="flex items-start gap-3">
                        {/* Avatar */}
                        <div className={clsx(
                          'w-9 h-9 rounded-full flex items-center justify-center text-white text-xs font-black flex-shrink-0',
                          user.color
                        )}>
                          {user.initials}
                        </div>
                        <div className="flex-1 min-w-0">
                          {/* Username + platform */}
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-sm font-bold text-gray-900">{user.username}</span>
                            {isPlatformX ? (
                              <Twitter className="h-3.5 w-3.5 text-gray-400" />
                            ) : (
                              <Hash className="h-3.5 w-3.5 text-orange-500" />
                            )}
                            <span className="text-xs text-gray-400">{timeAgo}</span>
                            {item.unverified && (
                              <span className="text-xs font-black px-2 py-0.5 rounded-full bg-orange-500 text-white uppercase">
                                Unverified
                              </span>
                            )}
                          </div>
                          {/* Content */}
                          <p className="text-sm text-gray-800 font-medium leading-relaxed">{item.content}</p>
                          {/* Engagement + sentiment */}
                          <div className="flex items-center gap-4 mt-2">
                            <span className="flex items-center gap-1 text-xs text-gray-400 hover:text-red-500 cursor-default">
                              <Heart className="h-3.5 w-3.5" /> {engagement.likes}
                            </span>
                            <span className="flex items-center gap-1 text-xs text-gray-400 hover:text-emerald-500 cursor-default">
                              <Repeat2 className="h-3.5 w-3.5" /> {engagement.reposts}
                            </span>
                            <SentimentTag sentiment={item.sentiment} />
                            <CredibilityBadge credibility={item.credibility} />
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </>
            )}

            {/* MACRO */}
            {activePanel === 'macro' && macro && (
              <div className="space-y-3">
                {macro.interest_rate_direction && (
                  <div className="p-4 bg-gray-50 rounded-xl border border-gray-200">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <BarChart3 className="h-5 w-5 text-gray-500" />
                        <span className="text-sm font-bold text-gray-800">Interest Rates</span>
                      </div>
                      <span className={clsx(
                        'text-sm font-black px-3 py-1 rounded-full',
                        macro.interest_rate_direction === 'up' ? 'bg-red-100 text-red-800' :
                        macro.interest_rate_direction === 'down' ? 'bg-emerald-100 text-emerald-800' :
                        'bg-gray-100 text-gray-700'
                      )}>
                        {macro.interest_rate_direction === 'up' ? 'Rising' :
                         macro.interest_rate_direction === 'down' ? 'Falling' : 'Flat'}
                      </span>
                    </div>
                  </div>
                )}
                {macro.market_breadth != null && (
                  <div className="p-4 bg-gray-50 rounded-xl border border-gray-200">
                    <div className="flex items-center gap-2 mb-3">
                      <Layers className="h-5 w-5 text-gray-500" />
                      <span className="text-sm font-bold text-gray-800">Market Breadth</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="flex-1 bg-gray-200 rounded-full h-3 overflow-hidden">
                        <div
                          className={clsx(
                            'h-full rounded-full',
                            macro.market_breadth > 0.6 ? 'bg-emerald-500' :
                            macro.market_breadth < 0.4 ? 'bg-red-500' : 'bg-amber-500'
                          )}
                          style={{ width: `${macro.market_breadth * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-black text-gray-900 w-20 text-right">
                        {Math.round(macro.market_breadth * 100)}% adv
                      </span>
                    </div>
                  </div>
                )}
                {macro.vix != null && (
                  <div className="p-4 bg-gray-50 rounded-xl border border-gray-200">
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-2">
                        <Activity className="h-5 w-5 text-gray-500" />
                        <span className="text-sm font-bold text-gray-800">Volatility Index</span>
                      </div>
                      <span className={clsx(
                        'text-2xl font-black',
                        macro.vix < 15 ? 'text-emerald-600' :
                        macro.vix < 25 ? 'text-amber-600' : 'text-red-600'
                      )}>
                        {macro.vix.toFixed(1)}
                      </span>
                    </div>
                    <p className="text-xs font-medium text-gray-500">
                      {macro.vix < 15 ? 'Low fear — calm markets' :
                       macro.vix < 25 ? 'Moderate uncertainty' : 'High fear — elevated risk'}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Recent events */}
          {state.recent_events?.length > 0 && (
            <div className="p-4 border-t border-gray-200">
              <div className="flex items-center gap-2 mb-2">
                <Radio className="h-3.5 w-3.5 text-red-500 animate-pulse" />
                <p className="text-xs font-bold text-red-600 uppercase tracking-wider">Live</p>
              </div>
              {state.recent_events.slice(-2).map((event, i) => (
                <p key={i} className="text-sm text-amber-700 font-bold leading-relaxed">{event.content}</p>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ─── Decision Panel ─────────────────────────────────────── */}
      <div className="bg-white rounded-2xl p-4 sm:p-6 border border-gray-200 shadow-md mt-5">

        {/* Amount + Confidence row */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-5">
          <div>
            <label htmlFor="trade-amount" className="text-sm font-bold text-gray-700 mb-2 block">Trade Amount</label>
            <div className="flex items-center gap-3">
              <input id="trade-amount" type="range" min="10" max={portfolio.cash || 1000}
                value={decisionAmount}
                onChange={(e) => setDecisionAmount(Number(e.target.value))}
                aria-label={`Trade amount: ${decisionAmount} dollars`}
                className="flex-1 accent-brand-navy h-2"
              />
              <span className="text-lg font-black text-gray-900 font-mono w-28 text-right">${decisionAmount}</span>
            </div>
            {feePreview && feePreview.total > 0.01 && (
              <p className="text-xs text-gray-500 mt-1.5 font-medium">
                Est. cost: ${feePreview.spread.toFixed(2)} spread + ${feePreview.fees.toFixed(2)} fee ={' '}
                <span className="font-bold text-gray-700">${feePreview.total.toFixed(2)}</span>
              </p>
            )}
          </div>
          <div>
            <label id="confidence-label" className="text-sm font-bold text-gray-700 mb-2 block">Confidence</label>
            <div className="flex items-center gap-2" role="group" aria-labelledby="confidence-label">
              {[1, 2, 3, 4, 5].map((level) => (
                <button key={level} onClick={() => setConfidence(level)}
                  aria-label={`Confidence level ${level}`}
                  aria-pressed={confidence === level}
                  className={clsx(
                    'w-11 h-11 rounded-xl font-bold text-base transition-all',
                    confidence === level
                      ? 'bg-brand-navy text-white shadow-md scale-110'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  )}>
                  {level}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Order Type Selector */}
        {orderTypesEnabled && (
          <div className="mb-5">
            <label className="text-sm font-bold text-gray-700 mb-2 block">Order Type</label>
            <div className="flex items-center gap-3 flex-wrap">
              {['market', 'limit', 'stop'].map((type) => (
                <button key={type} onClick={() => setOrderType(type)}
                  className={clsx(
                    'px-5 py-2.5 rounded-xl text-sm font-bold transition-all',
                    orderType === type
                      ? 'bg-brand-navy text-white shadow-md'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  )}>
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </button>
              ))}
              {orderType === 'limit' && (
                <div className="flex items-center gap-2 ml-1">
                  <span className="text-sm font-semibold text-gray-500">Limit Price:</span>
                  <input type="number" value={limitPrice}
                    onChange={(e) => setLimitPrice(e.target.value)}
                    placeholder={state.current_price.toFixed(2)} step="0.01"
                    className="w-32 px-3 py-2.5 rounded-xl border-2 border-gray-200 text-sm font-bold text-gray-900 bg-white focus:border-brand-navy focus:outline-none"
                  />
                </div>
              )}
              {orderType === 'stop' && (
                <div className="flex items-center gap-2 ml-1">
                  <span className="text-sm font-semibold text-gray-500">Trigger:</span>
                  <input type="number" value={stopPrice}
                    onChange={(e) => setStopPrice(e.target.value)}
                    placeholder={state.current_price.toFixed(2)} step="0.01"
                    className="w-32 px-3 py-2.5 rounded-xl border-2 border-gray-200 text-sm font-bold text-gray-900 bg-white focus:border-brand-navy focus:outline-none"
                  />
                </div>
              )}
            </div>
          </div>
        )}

        {/* Rationale */}
        <div className="mb-5">
          <label className="text-sm font-bold text-gray-700 mb-2 block">Why are you making this decision? (optional)</label>
          <textarea value={rationale} onChange={(e) => setRationale(e.target.value)}
            placeholder="Type your reasoning... This will be reviewed after the simulation."
            aria-label="Decision rationale"
            className="w-full bg-gray-50 text-gray-900 rounded-xl p-4 text-sm font-medium resize-none h-20 border-2 border-gray-200 focus:border-brand-navy focus:outline-none placeholder-gray-400"
            maxLength={500}
          />
          <div className="flex items-center justify-between mt-1.5">
            <p className="text-xs font-semibold text-gray-400">{rationale.length}/500</p>
            {rationale.trim().length >= 5 && (
              <button onClick={() => handleChallenge('buy')} disabled={challenging}
                className="text-xs text-brand-navy hover:text-brand-navy-light flex items-center gap-1.5 font-bold">
                <Zap className="h-3.5 w-3.5" />
                {challenging ? 'Analyzing...' : 'Challenge My Reasoning'}
              </button>
            )}
          </div>
        </div>

        {/* Challenge Result */}
        {challengeResult && (
          <div className={clsx(
            'mb-5 p-4 rounded-xl border-2',
            challengeResult.reasoning_score >= 4 ? 'bg-emerald-50 border-emerald-300' :
            challengeResult.reasoning_score >= 3 ? 'bg-amber-50 border-amber-300' :
            'bg-red-50 border-red-300'
          )}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-bold text-gray-900">AI Challenge Result</span>
              <span className={clsx(
                'text-lg font-black',
                challengeResult.reasoning_score >= 4 ? 'text-emerald-600' :
                challengeResult.reasoning_score >= 3 ? 'text-amber-600' : 'text-red-600'
              )}>
                {challengeResult.reasoning_score}/5
              </span>
            </div>
            <p className="text-sm text-gray-700 font-medium leading-relaxed">{challengeResult.feedback}</p>
            <div className="flex gap-2 mt-3">
              <button onClick={() => makeDecision(challengeResult.decisionType)} className="btn btn-primary text-xs py-2 px-4">Confirm Anyway</button>
              <button onClick={() => setChallengeResult(null)} className="btn btn-secondary text-xs py-2 px-4">Reconsider</button>
            </div>
          </div>
        )}

        {/* Wait picker */}
        {waitPickerOpen && (
          <div className="mb-5 flex items-center gap-3 p-4 bg-gray-50 rounded-xl border-2 border-gray-200">
            <FastForward className="h-5 w-5 text-brand-navy flex-shrink-0" />
            <span className="text-sm text-gray-800 font-bold">Skip ahead:</span>
            {[10, 30, 60].map((s) => (
              <button key={s} onClick={() => handleWaitWithDuration(s)} disabled={skipping}
                className="px-4 py-2 text-sm font-bold bg-white border-2 border-gray-200 rounded-xl text-gray-800 hover:bg-gray-100 hover:border-gray-300 transition-all">
                {s}s
              </button>
            ))}
            <button onClick={() => setWaitPickerOpen(false)} className="text-xs font-bold text-gray-400 hover:text-gray-700 ml-auto">Cancel</button>
          </div>
        )}

        {/* Action Buttons */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-3">
          <button onClick={() => makeDecision('buy')} disabled={isHalted}
            aria-label={isHalted ? 'Trading halted' : `Buy ${decisionAmount} dollars`}
            className={clsx(
              'flex items-center justify-center gap-2 py-3 sm:py-3.5 rounded-xl font-bold text-sm sm:text-base transition-all shadow-sm',
              isHalted
                ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                : 'bg-emerald-600 text-white hover:bg-emerald-500 active:scale-[0.98]'
            )}>
            <TrendingUp className="h-4 w-4 sm:h-5 sm:w-5" />
            {isHalted ? 'HALTED' : `BUY $${decisionAmount}`}
          </button>
          <button onClick={() => makeDecision('sell')} disabled={isHalted}
            aria-label={isHalted ? 'Trading halted' : 'Sell position'}
            className={clsx(
              'flex items-center justify-center gap-2 py-3 sm:py-3.5 rounded-xl font-bold text-sm sm:text-base transition-all shadow-sm',
              isHalted
                ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                : 'bg-red-600 text-white hover:bg-red-500 active:scale-[0.98]'
            )}>
            <TrendingDown className="h-4 w-4 sm:h-5 sm:w-5" />
            {isHalted ? 'HALTED' : 'SELL'}
          </button>
          <button onClick={() => makeDecision('hold')}
            aria-label="Hold current position"
            className="flex items-center justify-center gap-2 py-3 sm:py-3.5 rounded-xl font-bold text-sm sm:text-base bg-gray-100 text-gray-700 hover:bg-gray-200 transition-all">
            <Minus className="h-4 w-4 sm:h-5 sm:w-5" />
            HOLD
          </button>
          <button onClick={() => setWaitPickerOpen(!waitPickerOpen)}
            aria-label="Wait and skip time"
            className={clsx(
              'flex items-center justify-center gap-2 py-3 sm:py-3.5 rounded-xl font-bold text-sm sm:text-base transition-all',
              waitPickerOpen
                ? 'bg-brand-navy text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            )}>
            <Clock className="h-4 w-4 sm:h-5 sm:w-5" />
            WAIT
          </button>
        </div>

        {/* Keyboard hints — hidden on touch devices */}
        <div className="hidden sm:flex items-center justify-center gap-3 mt-4 text-xs text-gray-400">
          <span className="flex items-center gap-1"><kbd className="px-1.5 py-0.5 bg-gray-100 rounded border border-gray-200 font-mono text-gray-500">B</kbd> Buy</span>
          <span className="flex items-center gap-1"><kbd className="px-1.5 py-0.5 bg-gray-100 rounded border border-gray-200 font-mono text-gray-500">S</kbd> Sell</span>
          <span className="flex items-center gap-1"><kbd className="px-1.5 py-0.5 bg-gray-100 rounded border border-gray-200 font-mono text-gray-500">H</kbd> Hold</span>
          <span className="flex items-center gap-1"><kbd className="px-1.5 py-0.5 bg-gray-100 rounded border border-gray-200 font-mono text-gray-500">W</kbd> Wait</span>
          <span className="flex items-center gap-1"><kbd className="px-1.5 py-0.5 bg-gray-100 rounded border border-gray-200 font-mono text-gray-500">D</kbd> History</span>
        </div>

        {/* Footer actions */}
        <div className="flex items-center justify-between mt-5 pt-5 border-t border-gray-200">
          <button onClick={handleAbandon} className="text-sm font-semibold text-gray-400 hover:text-red-600 transition-colors">
            Abandon Simulation
          </button>
          <button onClick={handleComplete} className="btn btn-primary">End Simulation Early</button>
        </div>
      </div>

      {/* Decision History Panel */}
      {showHistory && (
        <div className="fixed right-0 top-0 bottom-0 w-full sm:w-80 bg-white border-l border-gray-200 shadow-2xl z-40 flex flex-col">
          <div className="flex items-center justify-between p-4 border-b border-gray-200">
            <h3 className="text-sm font-bold text-gray-900 uppercase tracking-wider">Your Moves</h3>
            <button onClick={() => setShowHistory(false)} className="text-gray-400 hover:text-gray-700">
              <X className="h-5 w-5" />
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {decisionHistory.length === 0 ? (
              <div className="text-center py-8">
                <Minus className="h-8 w-8 text-gray-300 mx-auto mb-2" />
                <p className="text-sm text-gray-400">No decisions yet</p>
                <p className="text-xs text-gray-300 mt-1">Your moves will appear here</p>
              </div>
            ) : (
              decisionHistory.map((d, i) => (
                <div key={i} className={clsx(
                  'p-3 rounded-xl border text-sm',
                  d.type === 'buy' ? 'bg-emerald-50 border-emerald-200' :
                  d.type === 'sell' ? 'bg-red-50 border-red-200' :
                  d.type === 'hold' ? 'bg-blue-50 border-blue-200' :
                  'bg-amber-50 border-amber-200'
                )}>
                  <div className="flex items-center justify-between mb-1">
                    <span className={clsx(
                      'font-bold uppercase text-xs tracking-wide',
                      d.type === 'buy' ? 'text-emerald-700' :
                      d.type === 'sell' ? 'text-red-700' :
                      d.type === 'hold' ? 'text-blue-700' :
                      'text-amber-700'
                    )}>
                      #{i + 1} {d.type}
                    </span>
                    <span className="text-xs text-gray-400">{Math.floor(d.time / 60)}:{(d.time % 60).toString().padStart(2, '0')}</span>
                  </div>
                  {(d.type === 'buy' || d.type === 'sell') && (
                    <p className="text-xs text-gray-600">
                      ${d.amount} @ ${d.price?.toFixed(2)}
                      {d.orderType !== 'market' && ` (${d.orderType})`}
                    </p>
                  )}
                  {d.rationale && (
                    <p className="text-xs text-gray-500 mt-1 italic truncate">"{d.rationale}"</p>
                  )}
                </div>
              ))
            )}
          </div>
          <div className="p-4 border-t border-gray-200 bg-gray-50">
            <p className="text-xs text-gray-400 text-center">{decisionHistory.length} decisions made</p>
          </div>
        </div>
      )}
    </div>
  );
}
