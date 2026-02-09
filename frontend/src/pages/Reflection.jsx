import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { reflectionApi } from '../api/client';
import BiasHeatmap from '../components/BiasHeatmap';
import ProReplayChart from '../components/ProReplayChart';
import {
  TrendingUp, TrendingDown, Target, Brain, GitBranch,
  Lightbulb, AlertTriangle, CheckCircle, ChevronRight,
  Shuffle, Award, ArrowRight, MessageCircle, UserCheck,
  HelpCircle, Eye, FileText, Zap, Search, BarChart2, Clock,
  Share2, Copy, Check, X
} from 'lucide-react';
import clsx from 'clsx';
import logo from '../../assests/logo.png';

const PERSONA_STYLES = {
  encouraging: {
    gradient: 'from-green-50 to-emerald-50',
    border: 'border-green-300',
    icon: '🌱',
    label: 'Encouraging Coach',
  },
  strict: {
    gradient: 'from-red-50 to-orange-50',
    border: 'border-red-300',
    icon: '🎯',
    label: 'Strict Coach',
  },
  analytical: {
    gradient: 'from-blue-50 to-cyan-50',
    border: 'border-blue-300',
    icon: '📊',
    label: 'Analytical Coach',
  },
  supportive: {
    gradient: 'from-brand-lavender/40 to-brand-blue/30',
    border: 'border-brand-navy/30',
    icon: '💬',
    label: 'Supportive Coach',
  },
};

function ProcessGauge({ score }) {
  const getColor = () => {
    if (score >= 70) return 'text-green-700';
    if (score >= 50) return 'text-amber-600';
    return 'text-red-600';
  };

  const getLabel = () => {
    if (score >= 70) return 'Strong';
    if (score >= 50) return 'Average';
    return 'Needs Work';
  };

  return (
    <div className="text-center">
      <div className="relative w-32 h-32 mx-auto">
        <svg className="w-32 h-32 transform -rotate-90">
          <circle
            cx="64"
            cy="64"
            r="56"
            stroke="currentColor"
            strokeWidth="12"
            fill="none"
            className="text-brand-blue/30"
          />
          <circle
            cx="64"
            cy="64"
            r="56"
            stroke="currentColor"
            strokeWidth="12"
            fill="none"
            strokeDasharray={`${score * 3.52} 352`}
            className={getColor()}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={clsx('text-3xl font-bold', getColor())}>{Math.round(score)}</span>
          <span className="text-brand-navy/60 text-sm">{getLabel()}</span>
        </div>
      </div>
    </div>
  );
}

function LuckSkillBar({ luck, skill }) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm">
        <span className="text-brand-navy">Luck</span>
        <span className="text-brand-blue">Skill</span>
      </div>
      <div className="h-4 bg-brand-blue/20 rounded-full overflow-hidden flex">
        <div
          className="bg-brand-navy transition-all duration-500"
          style={{ width: `${luck * 100}%` }}
        />
        <div
          className="bg-blue-500 transition-all duration-500"
          style={{ width: `${skill * 100}%` }}
        />
      </div>
      <div className="flex justify-between text-xs text-brand-navy/60">
        <span>{Math.round(luck * 100)}%</span>
        <span>{Math.round(skill * 100)}%</span>
      </div>
    </div>
  );
}

function PatternCard({ pattern }) {
  return (
    <div className="p-4 bg-brand-lavender/30 rounded-lg">
      <div className="flex items-center justify-between mb-2">
        <h4 className="font-medium text-brand-navy capitalize">
          {pattern.pattern_name.replace(/_/g, ' ')}
        </h4>
        <span className={clsx(
          'text-sm px-2 py-0.5 rounded',
          pattern.confidence >= 0.7 ? 'bg-red-100 text-red-600' :
            pattern.confidence >= 0.5 ? 'bg-yellow-100 text-amber-600' :
              'bg-brand-blue/30 text-brand-navy/70'
        )}>
          {Math.round(pattern.confidence * 100)}% confidence
        </span>
      </div>
      <p className="text-brand-navy/60 text-sm mb-3">{pattern.description}</p>
      <div className="space-y-1">
        {pattern.evidence.map((e, i) => (
          <div key={i} className="flex items-start space-x-2 text-sm text-brand-navy/70">
            <ChevronRight className="h-4 w-4 text-brand-navy flex-shrink-0 mt-0.5" />
            <span>{e}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function CounterfactualCard({ cf, index }) {
  const profit = cf.outcome?.profit_loss || 0;
  const isProfit = profit > 0;

  return (
    <div className="p-4 bg-brand-lavender/30 rounded-lg">
      <div className="flex items-center space-x-2 mb-2">
        <GitBranch className="h-5 w-5 text-brand-navy" />
        <h4 className="font-medium text-brand-navy">Timeline {index + 1}: {cf.timeline_name}</h4>
      </div>
      <p className="text-brand-navy/60 text-sm mb-3">{cf.description}</p>
      <div className="flex items-center justify-between p-3 bg-brand-cream rounded-lg mb-3">
        <span className="text-brand-navy/60">Alternate Outcome:</span>
        <span className={clsx(
          'text-lg font-bold',
          isProfit ? 'text-green-700' : 'text-red-600'
        )}>
          {isProfit ? '+' : ''}{profit.toFixed(2)}
        </span>
      </div>
      <p className="text-sm text-brand-navy/70 italic">"{cf.lesson}"</p>
    </div>
  );
}

function InsightCard({ insight }) {
  return (
    <div className="p-4 bg-gradient-to-r from-brand-lavender/50 to-brand-cream rounded-lg border border-brand-navy/30">
      <div className="flex items-start space-x-3">
        <Lightbulb className="h-5 w-5 text-brand-navy flex-shrink-0 mt-0.5" />
        <div>
          <h4 className="font-medium text-brand-navy mb-1">{insight.title}</h4>
          <p className="text-brand-navy/70 text-sm">{insight.description}</p>
          {insight.recommended_card_id && (
            <Link
              to="/learning"
              className="inline-flex items-center space-x-1 text-brand-navy hover:text-brand-navy-light text-sm mt-2"
            >
              <span>Learn more</span>
              <ArrowRight className="h-4 w-4" />
            </Link>
          )}
        </div>
      </div>
    </div>
  );
}

function RationaleCritiqueCard({ review }) {
  const qualityColors = {
    5: 'text-green-700 bg-green-100',
    4: 'text-green-700 bg-green-100',
    3: 'text-amber-600 bg-yellow-100',
    2: 'text-orange-600 bg-orange-100',
    1: 'text-red-600 bg-red-100',
  };
  const style = qualityColors[review.quality_score] || qualityColors[3];

  return (
    <div className="p-4 bg-brand-lavender/30 rounded-lg border-l-4 border-brand-navy">
      <div className="flex items-center justify-between mb-2">
        <h4 className="font-medium text-brand-navy">Decision #{review.decision_index + 1}</h4>
        <span className={clsx('text-xs px-2 py-1 rounded font-semibold', style)}>
          Quality: {review.quality_score}/5
        </span>
      </div>
      {review.user_rationale && (
        <p className="text-brand-navy/60 text-sm mb-2 italic">
          You said: "{review.user_rationale}"
        </p>
      )}
      <p className="text-brand-navy/70 text-sm mb-3">{review.critique}</p>
      {review.missed_factors?.length > 0 && (
        <div className="mb-2">
          <p className="text-xs text-brand-blue uppercase mb-1">Missed Factors</p>
          <div className="flex flex-wrap gap-1">
            {review.missed_factors.map((f, i) => (
              <span key={i} className="text-xs px-2 py-0.5 bg-orange-100 text-orange-600 rounded">
                {f}
              </span>
            ))}
          </div>
        </div>
      )}
      {review.reasoning_bias && (
        <p className="text-xs text-red-600">
          Reasoning bias detected: {review.reasoning_bias}
        </p>
      )}
    </div>
  );
}

function IsolatedCounterfactualCard({ data }) {
  return (
    <div className="p-4 bg-brand-lavender/30 rounded-lg border-l-4 border-brand-navy">
      <div className="flex items-center justify-between mb-3">
        <div>
          <p className="text-xs text-brand-blue uppercase">Original</p>
          <p className="text-brand-navy font-medium">{data.original_decision}</p>
        </div>
        <ArrowRight className="h-5 w-5 text-brand-navy mx-4" />
        <div className="text-right">
          <p className="text-xs text-brand-navy uppercase">Alternative</p>
          <p className="text-brand-navy/80 font-medium">{data.alternative_decision}</p>
        </div>
      </div>
      {data.ripple_effects?.length > 0 && (
        <div className="mb-3">
          <p className="text-xs text-brand-blue uppercase mb-1">Ripple Effects</p>
          <div className="space-y-1">
            {data.ripple_effects.map((effect, i) => (
              <div key={i} className="flex items-start space-x-2 text-sm text-brand-navy/70">
                <Zap className="h-3 w-3 text-brand-navy flex-shrink-0 mt-1" />
                <span>{effect}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      <div className="p-3 bg-brand-lavender/30 rounded-lg border border-brand-blue/30">
        <p className="text-sm text-brand-navy/70">
          <span className="text-brand-navy font-medium">Causal Impact: </span>
          {data.causal_impact}
        </p>
      </div>
      <p className="text-sm text-brand-navy/70 italic mt-2">"{data.lesson}"</p>
    </div>
  );
}

function CalibrationCard({ data }) {
  const total = data.overconfident_count + data.underconfident_count + data.well_calibrated_count;
  if (total === 0) return null;

  const pct = (count) => Math.round((count / total) * 100);
  const scoreColor = data.calibration_score >= 70 ? 'text-green-700' :
    data.calibration_score >= 40 ? 'text-amber-600' : 'text-red-600';

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-brand-navy mb-4 flex items-center space-x-2">
        <Target className="h-5 w-5 text-brand-navy" />
        <span>Confidence Calibration</span>
      </h3>
      <div className="text-center mb-4">
        <span className={clsx('text-4xl font-bold', scoreColor)}>{Math.round(data.calibration_score)}</span>
        <span className="text-brand-navy/60 text-sm ml-1">/ 100</span>
        <p className="text-brand-navy/60 text-sm mt-1">
          How well your confidence matched actual outcomes
        </p>
      </div>
      <div className="space-y-3">
        {[
          { label: 'Well Calibrated', count: data.well_calibrated_count, color: 'bg-green-500', textColor: 'text-green-700' },
          { label: 'Overconfident', count: data.overconfident_count, color: 'bg-red-500', textColor: 'text-red-600' },
          { label: 'Underconfident', count: data.underconfident_count, color: 'bg-yellow-500', textColor: 'text-amber-600' },
        ].map((item) => (
          <div key={item.label}>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-brand-navy/60">{item.label}</span>
              <span className={item.textColor}>{item.count} ({pct(item.count)}%)</span>
            </div>
            <div className="h-3 bg-brand-blue/20 rounded-full overflow-hidden">
              <div className={clsx('h-full rounded-full transition-all duration-500', item.color)}
                style={{ width: `${pct(item.count)}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function OutcomeDistributionChart({ data }) {
  if (!data?.buckets?.length) return null;

  const maxCount = Math.max(...data.buckets.map((b) => b.count));

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-brand-navy mb-4 flex items-center space-x-2">
        <BarChart2 className="h-5 w-5 text-brand-navy" />
        <span>Outcome Distribution</span>
      </h3>
      <p className="text-brand-navy/60 text-sm mb-4">
        Your decisions replayed across 100 different market scenarios
      </p>

      {/* Histogram */}
      <div className="flex items-end gap-1 h-32 mb-2">
        {data.buckets.map((bucket, i) => {
          const height = maxCount > 0 ? (bucket.count / maxCount) * 100 : 0;
          const isUserBucket = data.actual_outcome >= bucket.min && data.actual_outcome < bucket.max;
          return (
            <div key={i} className="flex-1 flex flex-col items-center justify-end h-full group relative">
              <div
                className={clsx(
                  'w-full rounded-t transition-all duration-300',
                  isUserBucket ? 'bg-brand-navy' : 'bg-brand-blue/30 group-hover:bg-brand-blue/40'
                )}
                style={{ height: `${Math.max(height, 2)}%` }}
              />
              {isUserBucket && (
                <div className="absolute -top-6 text-xs text-brand-navy font-semibold whitespace-nowrap">
                  You
                </div>
              )}
            </div>
          );
        })}
      </div>
      <div className="flex justify-between text-xs text-brand-blue">
        <span>${data.buckets[0]?.min?.toFixed(0)}</span>
        <span>${data.buckets[data.buckets.length - 1]?.max?.toFixed(0)}</span>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t border-brand-blue/20">
        <div className="text-center">
          <p className="text-xs text-brand-blue">Worst</p>
          <p className="text-red-600 font-semibold">${data.worst_outcome?.toFixed(2)}</p>
        </div>
        <div className="text-center">
          <p className="text-xs text-brand-blue">Median</p>
          <p className="text-brand-navy/70 font-semibold">${data.median_outcome?.toFixed(2)}</p>
        </div>
        <div className="text-center">
          <p className="text-xs text-brand-blue">Best</p>
          <p className="text-green-700 font-semibold">${data.best_outcome?.toFixed(2)}</p>
        </div>
      </div>
      <div className="text-center mt-3">
        <p className="text-xs text-brand-blue">Your Percentile</p>
        <p className="text-brand-navy font-bold text-lg">{data.user_percentile}th</p>
      </div>
    </div>
  );
}

export default function Reflection() {
  const { simulationId } = useParams();
  const navigate = useNavigate();

  const [reflection, setReflection] = useState(null);
  const [counterfactuals, setCounterfactuals] = useState([]);
  const [whyData, setWhyData] = useState(null);
  const [proData, setProData] = useState(null);
  const [coaching, setCoaching] = useState(null);
  const [biasHeatmap, setBiasHeatmap] = useState(null);
  const [rationaleReview, setRationaleReview] = useState(null);
  const [calibration, setCalibration] = useState(null);
  const [outcomeDistribution, setOutcomeDistribution] = useState(null);
  const [isolatedCf, setIsolatedCf] = useState({});
  const [loading, setLoading] = useState(true);
  const [secondaryLoading, setSecondaryLoading] = useState(true);
  const [showCounterfactuals, setShowCounterfactuals] = useState(false);
  const [showWhy, setShowWhy] = useState(false);
  const [showPro, setShowPro] = useState(false);
  const [showRationale, setShowRationale] = useState(false);
  const [loadingIsolation, setLoadingIsolation] = useState({});
  const [copied, setCopied] = useState(false);
  const [showShareCard, setShowShareCard] = useState(false);

  const copyShareText = () => {
    if (!reflection) return;
    const pq = reflection.process_quality?.score || 0;
    const biases = reflection.patterns_detected?.map(p => p.pattern_name.replace(/_/g, ' ')).join(', ') || 'none detected';
    const cal = calibration?.calibration_score != null ? `${Math.round(calibration.calibration_score)}%` : 'N/A';
    const text = [
      `IsItLegit — Decision Training Results`,
      ``,
      `Outcome: ${reflection.outcome_summary}`,
      `Process Quality: ${Math.round(pq)}/100`,
      `Calibration: ${cal}`,
      `Biases Detected: ${biases}`,
      `Luck vs Skill: ${Math.round((reflection.luck_factor || 0) * 100)}% luck / ${Math.round((reflection.skill_factor || 0) * 100)}% skill`,
      ``,
      `Key Takeaway: "${reflection.key_takeaway}"`,
      ``,
      `Train your decision-making at IsItLegit`,
    ].join('\n');
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const shareResults = () => {
    setShowShareCard(true);
  };

  useEffect(() => {
    loadReflection();
  }, [simulationId]);

  const loadReflection = async () => {
    try {
      // Phase 1: Load core reflection first (fastest path to visible content)
      let batchData = null;
      try {
        const batchRes = await reflectionApi.getFull(simulationId);
        batchData = batchRes.data;
        if (batchData?.reflection) {
          setReflection(batchData.reflection);
          setCounterfactuals(batchData.counterfactuals || []);
          setCoaching(batchData);
        }
      } catch {
        // batch not available, try individual
        const reflectionRes = await reflectionApi.get(simulationId).catch(() => ({ data: null }));
        setReflection(reflectionRes.data);
      }
    } catch (err) {
      console.error('Failed to load reflection:', err);
    } finally {
      setLoading(false);
    }

    // Phase 2: Load secondary data in the background (non-blocking)
    try {
      const [cfRes, whyRes, proRes, coachRes, heatmapRes, rationaleRes, calRes, outcomeRes] = await Promise.all([
        counterfactuals.length > 0 ? Promise.resolve({ data: counterfactuals }) : reflectionApi.getCounterfactuals(simulationId).catch(() => ({ data: [] })),
        reflectionApi.getWhyDecisions(simulationId).catch(() => ({ data: null })),
        reflectionApi.getProComparison(simulationId).catch(() => ({ data: null })),
        coaching ? Promise.resolve({ data: coaching }) : reflectionApi.getCoaching(simulationId).catch(() => ({ data: null })),
        reflectionApi.getBiasHeatmap(simulationId).catch(() => ({ data: null })),
        reflectionApi.getRationaleReview(simulationId).catch(() => ({ data: null })),
        reflectionApi.getCalibration(simulationId).catch(() => ({ data: null })),
        reflectionApi.getOutcomeDistribution(simulationId).catch(() => ({ data: null })),
      ]);

      if (cfRes.data?.length) setCounterfactuals(cfRes.data);
      setWhyData(whyRes.data);
      setProData(proRes.data);
      if (!coaching) setCoaching(coachRes.data);
      setBiasHeatmap(heatmapRes.data);
      setRationaleReview(rationaleRes.data);
      setCalibration(calRes.data);
      setOutcomeDistribution(outcomeRes.data);
    } catch (err) {
      console.error('Failed to load secondary data:', err);
    } finally {
      setSecondaryLoading(false);
    }
  };

  const loadIsolatedCounterfactual = async (decisionIndex) => {
    if (isolatedCf[decisionIndex] || loadingIsolation[decisionIndex]) return;
    setLoadingIsolation((prev) => ({ ...prev, [decisionIndex]: true }));
    try {
      const res = await reflectionApi.getCounterfactualIsolation(simulationId, decisionIndex);
      setIsolatedCf((prev) => ({ ...prev, [decisionIndex]: res.data }));
    } catch (err) {
      console.error('Failed to load isolated counterfactual:', err);
    } finally {
      setLoadingIsolation((prev) => ({ ...prev, [decisionIndex]: false }));
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="text-center">
          <Brain className="h-16 w-16 text-brand-navy mx-auto mb-4 animate-pulse" />
          <p className="text-brand-navy text-lg">Analyzing your decisions...</p>
          <p className="text-brand-navy/60 text-sm mt-2">Generating personalized insights</p>
        </div>
      </div>
    );
  }

  if (!reflection) {
    return (
      <div className="text-center py-20">
        <AlertTriangle className="h-12 w-12 text-amber-600 mx-auto mb-4" />
        <p className="text-brand-navy text-lg mb-2">Could not load reflection analysis</p>
        <p className="text-brand-navy/60 mb-6 max-w-md mx-auto">
          The AI analysis service may be temporarily unavailable. Try again or return to the dashboard.
        </p>
        <div className="flex justify-center space-x-4">
          <button
            onClick={() => { setLoading(true); loadReflection(); }}
            className="btn btn-primary"
          >
            Try Again
          </button>
          <button onClick={() => navigate('/dashboard')} className="btn btn-secondary">
            Return to Dashboard
          </button>
        </div>
      </div>
    );
  }

  const isProfit = reflection.outcome_type === 'profit';
  const processScore = reflection.process_quality?.score || 0;
  const persona = coaching?.persona || 'supportive';
  const personaStyle = PERSONA_STYLES[persona] || PERSONA_STYLES.supportive;

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Outcome Header */}
      <div className="card text-center">
        <div className={clsx(
          'inline-flex items-center justify-center w-20 h-20 rounded-full mb-4',
          isProfit ? 'bg-green-100' : 'bg-red-100'
        )}>
          {isProfit ? (
            <TrendingUp className="h-10 w-10 text-green-700" />
          ) : (
            <TrendingDown className="h-10 w-10 text-red-600" />
          )}
        </div>
        <h1 className={clsx(
          'text-4xl font-bold mb-2',
          isProfit ? 'text-green-700' : 'text-red-600'
        )}>
          {reflection.outcome_summary}
        </h1>
        <p className="text-brand-navy/60 text-lg">
          Process Quality: <span className={clsx(
            processScore >= 70 ? 'text-green-700' :
              processScore >= 50 ? 'text-amber-600' : 'text-red-600'
          )}>
            {processScore >= 70 ? 'Strong' : processScore >= 50 ? 'Average' : 'Risky'}
          </span>
        </p>
      </div>

      {/* Key Takeaway */}
      <div className={clsx(
        'p-6 rounded-xl border-2',
        isProfit && processScore < 50
          ? 'bg-yellow-500/10 border-yellow-500/50'
          : !isProfit && processScore >= 70
            ? 'bg-blue-500/10 border-blue-500/50'
            : 'bg-brand-cream border-brand-blue/20'
      )}>
        <div className="flex items-start space-x-4">
          {isProfit && processScore < 50 ? (
            <AlertTriangle className="h-8 w-8 text-amber-600 flex-shrink-0" />
          ) : !isProfit && processScore >= 70 ? (
            <CheckCircle className="h-8 w-8 text-brand-blue flex-shrink-0" />
          ) : (
            <Target className="h-8 w-8 text-brand-navy flex-shrink-0" />
          )}
          <div>
            <h3 className="text-lg font-semibold text-brand-navy mb-2">Key Takeaway</h3>
            <p className="text-brand-navy/70">{reflection.key_takeaway}</p>
          </div>
        </div>
      </div>

      {/* Process Quality & Luck/Skill */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold text-brand-navy mb-4 flex items-center space-x-2">
            <Target className="h-5 w-5 text-brand-navy" />
            <span>Process Quality</span>
          </h3>
          <ProcessGauge score={processScore} />
          <p className="text-brand-navy/60 text-sm text-center mt-4">
            {reflection.process_quality?.summary}
          </p>
          <div className="mt-4 space-y-2">
            {Object.entries(reflection.process_quality?.factors || {}).map(([factor, value]) => (
              <div key={factor} className="flex items-center justify-between">
                <span className="text-brand-navy/60 capitalize">{factor.replace(/_/g, ' ')}</span>
                <div className="flex items-center space-x-2">
                  <div className="w-24 h-2 bg-brand-blue/20 rounded-full overflow-hidden">
                    <div
                      className={clsx(
                        'h-full rounded-full',
                        value >= 0.7 ? 'bg-green-500' :
                          value >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'
                      )}
                      style={{ width: `${value * 100}%` }}
                    />
                  </div>
                  <span className="text-brand-navy text-sm w-8">{Math.round(value * 100)}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold text-brand-navy mb-4 flex items-center space-x-2">
            <Shuffle className="h-5 w-5 text-brand-navy" />
            <span>Luck vs Skill</span>
          </h3>
          <div className="py-6">
            <LuckSkillBar luck={reflection.luck_factor} skill={reflection.skill_factor} />
          </div>
          <p className="text-brand-navy/60 text-sm">{reflection.luck_skill_explanation}</p>
        </div>
      </div>

      {/* Calibration & Outcome Distribution */}
      {secondaryLoading && !calibration && !outcomeDistribution ? (
        <div className="grid md:grid-cols-2 gap-6">
          <div className="card animate-pulse"><div className="h-6 w-40 bg-brand-lavender/40 rounded mb-4" /><div className="h-32 bg-brand-lavender/20 rounded" /></div>
          <div className="card animate-pulse"><div className="h-6 w-48 bg-brand-lavender/40 rounded mb-4" /><div className="h-32 bg-brand-lavender/20 rounded" /></div>
        </div>
      ) : (calibration || outcomeDistribution) && (
        <div className="grid md:grid-cols-2 gap-6">
          {calibration && <CalibrationCard data={calibration} />}
          {outcomeDistribution && <OutcomeDistributionChart data={outcomeDistribution} />}
        </div>
      )}

      {/* Bias Heatmap Timeline */}
      {secondaryLoading && !biasHeatmap ? (
        <div className="card animate-pulse"><div className="h-6 w-52 bg-brand-lavender/40 rounded mb-4" /><div className="h-24 bg-brand-lavender/20 rounded" /></div>
      ) : biasHeatmap && biasHeatmap.timeline?.length > 0 && (
        <BiasHeatmap data={biasHeatmap} />
      )}

      {/* Patterns Detected */}
      <div className="card">
        <h3 className="text-lg font-semibold text-brand-navy mb-4 flex items-center space-x-2">
          <Brain className="h-5 w-5 text-brand-navy" />
          <span>Patterns Detected</span>
        </h3>
        <div className="grid md:grid-cols-2 gap-4">
          {reflection.patterns_detected?.map((pattern, i) => (
            <PatternCard key={i} pattern={pattern} />
          ))}
        </div>
      </div>

      {/* Rationale Review */}
      {rationaleReview && rationaleReview.reviews?.length > 0 && (
        <div className="card">
          <button
            onClick={() => setShowRationale(!showRationale)}
            className="w-full flex items-center justify-between"
          >
            <h3 className="text-lg font-semibold text-brand-navy flex items-center space-x-2">
              <FileText className="h-5 w-5 text-brand-navy" />
              <span>Rationale Review</span>
              <span className={clsx(
                'text-sm px-2 py-0.5 rounded ml-2',
                rationaleReview.overall_reasoning_quality >= 4 ? 'bg-green-100 text-green-700' :
                  rationaleReview.overall_reasoning_quality >= 3 ? 'bg-yellow-100 text-amber-600' :
                    'bg-red-100 text-red-600'
              )}>
                Overall: {rationaleReview.overall_reasoning_quality}/5
              </span>
            </h3>
            <ChevronRight className={clsx(
              'h-5 w-5 text-brand-navy/60 transition-transform',
              showRationale && 'rotate-90'
            )} />
          </button>
          <p className="text-brand-navy/60 text-sm mt-2">
            AI critique of your stated reasoning for each decision
          </p>

          {showRationale && (
            <div className="mt-4 space-y-4">
              {rationaleReview.reviews.map((review, i) => (
                <RationaleCritiqueCard key={i} review={review} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Counterfactuals */}
      <div className="card">
        <button
          onClick={() => setShowCounterfactuals(!showCounterfactuals)}
          className="w-full flex items-center justify-between"
        >
          <h3 className="text-lg font-semibold text-brand-navy flex items-center space-x-2">
            <GitBranch className="h-5 w-5 text-brand-navy" />
            <span>Alternate Timelines</span>
          </h3>
          <ChevronRight className={clsx(
            'h-5 w-5 text-brand-navy/60 transition-transform',
            showCounterfactuals && 'rotate-90'
          )} />
        </button>
        <p className="text-brand-navy/60 text-sm mt-2">
          What could have happened with the same decisions in different market conditions?
        </p>

        {showCounterfactuals && (
          <div className="mt-4 space-y-4">
            {counterfactuals.map((cf, i) => (
              <CounterfactualCard key={i} cf={cf} index={i} />
            ))}
          </div>
        )}
      </div>

      {/* Actionable Insights */}
      <div className="card">
        <h3 className="text-lg font-semibold text-brand-navy mb-4 flex items-center space-x-2">
          <Lightbulb className="h-5 w-5 text-amber-600" />
          <span>Actionable Insights</span>
        </h3>
        <div className="space-y-4">
          {reflection.insights?.map((insight, i) => (
            <InsightCard key={i} insight={insight} />
          ))}
        </div>
      </div>

      {/* Personalized Coaching with Adaptive Persona */}
      {(reflection.coaching_message || coaching?.coaching_message) && (
        <div className={clsx(
          'card bg-gradient-to-br border',
          personaStyle.gradient,
          personaStyle.border
        )}>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold text-brand-navy flex items-center space-x-2">
              <MessageCircle className="h-5 w-5 text-brand-navy" />
              <span>Your Personal Coach</span>
            </h3>
            <span className="text-xs px-3 py-1 bg-brand-lavender rounded-full text-brand-navy">
              {personaStyle.icon} {personaStyle.label}
            </span>
          </div>
          <p className="text-brand-navy/80 leading-relaxed italic">
            "{coaching?.coaching_message || reflection.coaching_message}"
          </p>
        </div>
      )}

      {/* Why This Decision? (with per-decision counterfactual isolation) */}
      {whyData && whyData.explanations?.length > 0 && (
        <div className="card">
          <button
            onClick={() => setShowWhy(!showWhy)}
            className="w-full flex items-center justify-between"
          >
            <h3 className="text-lg font-semibold text-brand-navy flex items-center space-x-2">
              <HelpCircle className="h-5 w-5 text-orange-600" />
              <span>Why This Decision?</span>
            </h3>
            <ChevronRight className={clsx(
              'h-5 w-5 text-brand-navy/60 transition-transform',
              showWhy && 'rotate-90'
            )} />
          </button>
          <p className="text-brand-navy/60 text-sm mt-2">
            {whyData.overall_narrative}
          </p>

          {showWhy && (
            <div className="mt-4 space-y-4">
              {whyData.explanations.map((exp, i) => (
                <div key={i}>
                  <div className="p-4 bg-brand-lavender/30 rounded-lg border-l-4 border-orange-500">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium text-brand-navy">
                        Decision #{exp.decision_index + 1}: {exp.decision_type.toUpperCase()} at {exp.timestamp_seconds}s
                      </h4>
                      <span className={clsx(
                        'text-xs px-2 py-1 rounded',
                        exp.severity === 'significant' ? 'bg-red-100 text-red-600' :
                          exp.severity === 'moderate' ? 'bg-yellow-100 text-amber-600' :
                            'bg-brand-blue/30 text-brand-navy/70'
                      )}>
                        {exp.severity}
                      </span>
                    </div>
                    <p className="text-sm text-orange-600 mb-2 capitalize">
                      Detected: {exp.detected_bias.replace(/_/g, ' ')}
                    </p>
                    <p className="text-brand-navy/70 text-sm mb-3">{exp.explanation}</p>
                    <div className="space-y-1 mb-3">
                      {exp.evidence_from_actions.map((e, j) => (
                        <div key={j} className="flex items-start space-x-2 text-sm text-brand-navy/60">
                          <Eye className="h-3 w-3 text-orange-600 flex-shrink-0 mt-1" />
                          <span>{e}</span>
                        </div>
                      ))}
                    </div>

                    {/* Evidence Timestamps */}
                    {exp.evidence_timestamps?.length > 0 && (
                      <div className="mb-3">
                        <p className="text-xs text-brand-blue uppercase mb-1">Key Evidence</p>
                        <div className="flex flex-wrap gap-2">
                          {exp.evidence_timestamps.map((et, k) => (
                            <span key={k} className="inline-flex items-center space-x-1 text-xs px-2 py-1 bg-orange-50 text-orange-600 rounded border border-orange-300">
                              <Clock className="h-3 w-3" />
                              <span>t={et.time}s</span>
                              <span className="text-brand-navy/60">— {et.event}</span>
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Counterfactual Isolation Button */}
                    <button
                      onClick={() => loadIsolatedCounterfactual(exp.decision_index)}
                      disabled={loadingIsolation[exp.decision_index]}
                      className="flex items-center space-x-2 text-sm text-brand-navy hover:text-brand-navy-light transition-colors"
                    >
                      <Search className="h-4 w-4" />
                      <span>
                        {loadingIsolation[exp.decision_index]
                          ? 'Analyzing impact...'
                          : isolatedCf[exp.decision_index]
                            ? 'Impact analyzed'
                            : 'See causal impact of this decision'}
                      </span>
                    </button>
                  </div>

                  {/* Isolated Counterfactual Result */}
                  {isolatedCf[exp.decision_index] && (
                    <div className="mt-2 ml-4">
                      <IsolatedCounterfactualCard data={isolatedCf[exp.decision_index]} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* What Would a Pro Do? */}
      {proData && proData.pro_decisions?.length > 0 && (
        <div className="card">
          <button
            onClick={() => setShowPro(!showPro)}
            className="w-full flex items-center justify-between"
          >
            <h3 className="text-lg font-semibold text-brand-navy flex items-center space-x-2">
              <UserCheck className="h-5 w-5 text-cyan-700" />
              <span>What Would a Pro Do?</span>
            </h3>
            <ChevronRight className={clsx(
              'h-5 w-5 text-brand-navy/60 transition-transform',
              showPro && 'rotate-90'
            )} />
          </button>
          <p className="text-brand-navy/60 text-sm mt-2">
            Side-by-side comparison with an experienced trader's approach
          </p>

          {showPro && (
            <div className="mt-4 space-y-4">
              {/* Outcome comparison */}
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-brand-lavender/30 rounded-lg text-center">
                  <p className="text-brand-navy/60 text-sm mb-1">Your Outcome</p>
                  <p className={clsx(
                    'text-2xl font-bold',
                    (proData.user_final_outcome?.profit_loss || 0) >= 0 ? 'text-green-700' : 'text-red-600'
                  )}>
                    ${(proData.user_final_outcome?.profit_loss || 0).toFixed(2)}
                  </p>
                </div>
                <div className="p-4 bg-cyan-50 rounded-lg text-center border border-brand-blue/30">
                  <p className="text-cyan-700 text-sm mb-1">Pro's Outcome</p>
                  <p className={clsx(
                    'text-2xl font-bold',
                    (proData.pro_final_outcome?.profit_loss || 0) >= 0 ? 'text-green-700' : 'text-red-600'
                  )}>
                    ${(proData.pro_final_outcome?.profit_loss || 0).toFixed(2)}
                  </p>
                </div>
              </div>

              {/* Decision-by-decision comparison */}
              {proData.pro_decisions.map((pd, i) => (
                <div key={i} className="p-4 bg-brand-lavender/30 rounded-lg">
                  <p className="text-brand-navy/60 text-xs mb-2">At t={pd.at_timestamp}s</p>
                  <div className="grid grid-cols-2 gap-4 mb-3">
                    <div>
                      <p className="text-xs text-brand-blue mb-1">You did:</p>
                      <p className="text-sm text-brand-navy">{pd.user_action}</p>
                    </div>
                    <div className="border-l border-brand-blue/30 pl-4">
                      <p className="text-xs text-cyan-700 mb-1">Pro would:</p>
                      <p className="text-sm text-cyan-700">{pd.pro_action}</p>
                    </div>
                  </div>
                  <p className="text-sm text-brand-navy/70 mb-2">
                    <span className="text-brand-blue">Why: </span>{pd.pro_reasoning}
                  </p>
                  <span className="inline-block text-xs px-2 py-1 bg-cyan-100 text-cyan-700 rounded">
                    Skill: {pd.skill_demonstrated}
                  </span>
                </div>
              ))}

              {/* What to practice */}
              {proData.what_to_practice?.length > 0 && (
                <div className="p-4 bg-cyan-50 rounded-lg border border-brand-blue/30">
                  <p className="text-cyan-700 font-medium mb-2">Skills to Practice:</p>
                  <ul className="space-y-1">
                    {proData.what_to_practice.map((skill, i) => (
                      <li key={i} className="flex items-start space-x-2 text-sm text-brand-navy/70">
                        <CheckCircle className="h-4 w-4 text-cyan-700 flex-shrink-0 mt-0.5" />
                        <span>{skill}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Pro Replay Chart */}
              {proData.price_history?.length > 0 && (
                <ProReplayChart
                  priceHistory={proData.price_history}
                  userDecisions={proData.pro_decisions?.map(pd => ({
                    time: pd.at_timestamp,
                    action: pd.user_action,
                  })) || []}
                  proDecisions={proData.pro_decisions?.map(pd => ({
                    time: pd.at_timestamp,
                    action: pd.pro_action,
                  })) || []}
                />
              )}
            </div>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="flex flex-wrap gap-4 justify-center">
        <Link to="/dashboard" className="btn btn-secondary">
          Back to Dashboard
        </Link>
        <Link to="/learning" className="btn btn-primary flex items-center space-x-2">
          <Award className="h-5 w-5" />
          <span>Practice with Learning Cards</span>
        </Link>
        <button
          onClick={shareResults}
          className="btn btn-secondary flex items-center space-x-2"
          aria-label="Share results as a card"
        >
          <Share2 className="h-5 w-5" />
          <span>Share Results</span>
        </button>
      </div>

      {/* Share Card Modal */}
      {showShareCard && reflection && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4" onClick={() => setShowShareCard(false)}>
          <div className="relative max-w-md w-full" onClick={(e) => e.stopPropagation()}>
            {/* Close button */}
            <button onClick={() => setShowShareCard(false)} className="absolute -top-3 -right-3 z-10 bg-brand-navy rounded-full p-1 hover:bg-brand-navy-light">
              <X className="h-5 w-5 text-white" />
            </button>

            {/* The Card */}
            <div className="rounded-2xl overflow-hidden shadow-2xl border border-brand-blue/30">
              {/* Card Header */}
              <div className={`p-6 ${isProfit ? 'bg-gradient-to-br from-green-900 to-emerald-800' : 'bg-gradient-to-br from-red-900 to-rose-800'}`}>
                <div className="flex items-center gap-3 mb-4">
                  <img src={logo} alt="IsItLegit" className="h-10 w-10" />
                  <div>
                    <p className="text-white font-bold text-lg">IsItLegit</p>
                    <p className="text-white/60 text-xs">Decision Training Results</p>
                  </div>
                </div>
                <p className="text-3xl font-bold text-white">{reflection.outcome_summary}</p>
              </div>

              {/* Card Body */}
              <div className="bg-brand-cream p-6 space-y-4">
                {/* Stats Grid */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-brand-lavender/30 rounded-lg p-3 text-center">
                    <p className="text-xs text-brand-navy/60 mb-1">Process Quality</p>
                    <p className="text-2xl font-bold text-brand-navy">{Math.round(processScore)}<span className="text-sm text-brand-navy/60">/100</span></p>
                  </div>
                  <div className="bg-brand-lavender/30 rounded-lg p-3 text-center">
                    <p className="text-xs text-brand-navy/60 mb-1">Calibration</p>
                    <p className="text-2xl font-bold text-brand-navy">
                      {calibration?.calibration_score != null ? `${Math.round(calibration.calibration_score)}%` : 'N/A'}
                    </p>
                  </div>
                  <div className="bg-brand-lavender/30 rounded-lg p-3 text-center">
                    <p className="text-xs text-brand-navy/60 mb-1">Luck</p>
                    <p className="text-xl font-bold text-amber-600">{Math.round((reflection.luck_factor || 0) * 100)}%</p>
                  </div>
                  <div className="bg-brand-lavender/30 rounded-lg p-3 text-center">
                    <p className="text-xs text-brand-navy/60 mb-1">Skill</p>
                    <p className="text-xl font-bold text-cyan-700">{Math.round((reflection.skill_factor || 0) * 100)}%</p>
                  </div>
                </div>

                {/* Biases */}
                {reflection.patterns_detected?.length > 0 && (
                  <div>
                    <p className="text-xs text-brand-navy/60 mb-2">Biases Detected</p>
                    <div className="flex flex-wrap gap-1.5">
                      {reflection.patterns_detected.map((p, i) => (
                        <span key={i} className="text-xs px-2 py-1 rounded-full bg-yellow-100 text-amber-600">
                          {p.pattern_name.replace(/_/g, ' ')}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Key Takeaway */}
                {reflection.key_takeaway && (
                  <div className="bg-brand-lavender/40 border border-brand-navy/20 rounded-lg p-3">
                    <p className="text-xs text-brand-navy mb-1">Key Takeaway</p>
                    <p className="text-sm text-brand-navy/80 italic">"{reflection.key_takeaway}"</p>
                  </div>
                )}

                <p className="text-center text-xs text-brand-blue pt-2">Train your decision-making at IsItLegit</p>
              </div>
            </div>

            {/* Copy Button */}
            <button
              onClick={copyShareText}
              className="btn btn-primary w-full mt-3 flex items-center justify-center gap-2"
            >
              {copied ? <Check className="h-5 w-5" /> : <Copy className="h-5 w-5" />}
              {copied ? 'Copied to Clipboard!' : 'Copy Results as Text'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
