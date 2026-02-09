import {
  Target, Ban, Activity, Users, Layers, Clock, Gauge, BarChart3,
  ArrowRight, DollarSign, Shield, TrendingUp, TrendingDown, Pause,
  Eye, CheckCircle, AlertTriangle, ShieldAlert, ShieldCheck, ShieldQuestion
} from 'lucide-react';

/* ─── Toast config ──────────────────────────────────────────────── */

export const TOAST_CONFIG = {
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

/* ─── News / Social helpers ─────────────────────────────────────── */

export const NEWS_SOURCES = ['Reuters', 'Bloomberg', 'MarketWatch', 'CNBC', 'WSJ'];
export const UNVERIFIED_SOURCES = ['Anonymous Tip', 'Forum Post', 'Unverified Source', 'Social Media'];
export const USERNAMES = [
  '@trader_mike', '@crypto_whale', '@market_guru', '@wall_st_wolf', '@penny_picker',
  '@bull_runner', '@bear_hunter', '@options_queen', '@value_victor', '@swing_king',
  '@diamond_hands', '@chart_master', '@risk_taker', '@steady_eddie', '@alpha_seeker'
];
export const AVATAR_COLORS = [
  'bg-blue-500', 'bg-purple-500', 'bg-pink-500', 'bg-indigo-500',
  'bg-teal-500', 'bg-orange-500', 'bg-cyan-500', 'bg-rose-500', 'bg-emerald-500', 'bg-violet-500'
];

export function hashStr(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash) + str.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

export function getNewsSource(item) {
  if (item.unverified) {
    return UNVERIFIED_SOURCES[hashStr(item.content) % UNVERIFIED_SOURCES.length];
  }
  return NEWS_SOURCES[hashStr(item.content) % NEWS_SOURCES.length];
}

export function getSocialUser(content) {
  const h = hashStr(content);
  const username = USERNAMES[h % USERNAMES.length];
  const initials = username.slice(1, 3).toUpperCase();
  const color = AVATAR_COLORS[h % AVATAR_COLORS.length];
  return { username, initials, color };
}

export function getEngagement(content, sentiment) {
  const h = hashStr(content);
  const base = sentiment === 'bullish' || sentiment === 'bearish' ? 3 : 1;
  const likes = (h % 40) * base + 2;
  const reposts = Math.floor(likes * 0.3) + (h % 5);
  return { likes, reposts };
}

export function relativeTime(itemTime, currentTime) {
  const diff = currentTime - itemTime;
  if (diff <= 0) return 'Just now';
  if (diff < 60) return `${diff}s ago`;
  return `${Math.floor(diff / 60)}m ago`;
}

export function isBreaking(itemTime, currentTime) {
  return currentTime - itemTime <= 10;
}

/* ─── Feature badges for briefing ───────────────────────────────── */

export const FEATURE_LABELS = {
  order_types_enabled: { label: 'Order Types', icon: Target, desc: 'Limit & stop orders available' },
  halts_enabled: { label: 'Circuit Breakers', icon: Ban, desc: 'Trading halts on big moves' },
  volatility_clustering: { label: 'Volatility Clustering', icon: Activity, desc: 'GARCH-style vol surges' },
  crowd_model_enabled: { label: 'Crowd Behavior', icon: Users, desc: 'Herd sentiment affects prices' },
  margin_enabled: { label: 'Margin Trading', icon: Layers, desc: 'Leverage & margin calls' },
  news_latency_enabled: { label: 'News Delays', icon: Clock, desc: 'Breaking news arrives late' },
  time_pressure_fills: { label: 'Time Pressure', icon: Gauge, desc: 'Prices move while you decide' },
  secondary_asset: { label: 'Correlated Asset', icon: BarChart3, desc: 'A second asset moves in tandem' },
};

export function getActiveFeatures(marketParams) {
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

export const DIFFICULTY_LABELS = ['', 'Beginner', 'Easy', 'Medium', 'Hard', 'Extreme'];
export const DIFFICULTY_COLORS = [
  '', 'text-emerald-600 bg-emerald-100', 'text-blue-600 bg-blue-100',
  'text-amber-600 bg-amber-100', 'text-orange-600 bg-orange-100', 'text-red-600 bg-red-100'
];

/* ─── Volatility label ──────────────────────────────────────────── */

export const volLabel = (vol) => {
  if (vol == null) return null;
  if (vol < 0.01) return { text: 'Low', color: 'text-emerald-700', bg: 'bg-emerald-50' };
  if (vol < 0.03) return { text: 'Med', color: 'text-amber-700', bg: 'bg-amber-50' };
  return { text: 'High', color: 'text-red-700', bg: 'bg-red-50' };
};
