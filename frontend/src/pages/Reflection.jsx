import { useState, useEffect, useRef, useCallback } from 'react';
import html2canvas from 'html2canvas';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { reflectionApi } from '../api/client';
import BiasHeatmap from '../components/BiasHeatmap';
import ProReplayChart from '../components/ProReplayChart';
import SectionErrorBoundary from '../components/SectionErrorBoundary';
import {
  ProcessGauge, LuckSkillBar, PatternCard, CounterfactualCard,
  InsightCard, RationaleCritiqueCard, IsolatedCounterfactualCard,
  CalibrationCard, OutcomeDistributionChart,
} from '../components/reflection';
import {
  TrendingUp, TrendingDown, Target, Brain, GitBranch,
  Lightbulb, AlertTriangle, CheckCircle, ChevronRight,
  Shuffle, Award, ArrowRight, MessageCircle, UserCheck,
  HelpCircle, Eye, FileText, Zap, Search, BarChart2, Clock,
  Share2, Copy, Check, X, Download
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

export default function Reflection() {
  const { simulationId } = useParams();
  const navigate = useNavigate();

  const [reflection, setReflection] = useState(null);
  const [counterfactuals, setCounterfactuals] = useState([]);
  const [whyData, setWhyData] = useState(null);
  const [proData, setProData] = useState(null);
  const [coaching, setCoaching] = useState(null);
  const [aiMetadata, setAiMetadata] = useState(null);
  const [showThinking, setShowThinking] = useState(false);
  const [biasHeatmap, setBiasHeatmap] = useState(null);
  const [rationaleReview, setRationaleReview] = useState(null);
  const [calibration, setCalibration] = useState(null);
  const [outcomeDistribution, setOutcomeDistribution] = useState(null);
  const [isolatedCf, setIsolatedCf] = useState({});
  const [biasClassifier, setBiasClassifier] = useState(null);
  const [confidenceCalibration, setConfidenceCalibration] = useState(null);
  const [loading, setLoading] = useState(true);
  const [secondaryLoading, setSecondaryLoading] = useState(true);
  const [showCounterfactuals, setShowCounterfactuals] = useState(false);
  const [showWhy, setShowWhy] = useState(false);
  const [showPro, setShowPro] = useState(false);
  const [showRationale, setShowRationale] = useState(false);
  const [showAlgoBaseline, setShowAlgoBaseline] = useState(false);
  const [loadingIsolation, setLoadingIsolation] = useState({});
  const [secondaryError, setSecondaryError] = useState(false);
  const [copied, setCopied] = useState(false);
  const [showShareCard, setShowShareCard] = useState(false);
  const [showFullAnalysis, setShowFullAnalysis] = useState(false);
  const [savingImage, setSavingImage] = useState(false);
  const shareCardRef = useRef(null);

  const saveAsImage = useCallback(async () => {
    if (!shareCardRef.current) return;
    setSavingImage(true);
    try {
      const canvas = await html2canvas(shareCardRef.current, {
        backgroundColor: null,
        scale: 2,
        useCORS: true,
      });
      const link = document.createElement('a');
      link.download = 'isitlegit-results.png';
      link.href = canvas.toDataURL('image/png');
      link.click();
    } catch (err) {
      console.error('Failed to save image:', err);
    } finally {
      setSavingImage(false);
    }
  }, []);

  const copyShareText = () => {
    if (!reflection) return;
    const pq = reflection.process_quality?.score || 0;
    const biases = reflection.patterns_detected?.map(p => (p.pattern_name || 'unknown').replace(/_/g, ' ')).join(', ') || 'none detected';
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
      const [cfRes, whyRes, proRes, coachRes, heatmapRes, rationaleRes, calRes, outcomeRes, biasClsRes, confCalRes, metaRes] = await Promise.all([
        counterfactuals.length > 0 ? Promise.resolve({ data: counterfactuals }) : reflectionApi.getCounterfactuals(simulationId).catch(() => ({ data: [] })),
        reflectionApi.getWhyDecisions(simulationId).catch(() => ({ data: null })),
        reflectionApi.getProComparison(simulationId).catch(() => ({ data: null })),
        coaching ? Promise.resolve({ data: coaching }) : reflectionApi.getCoaching(simulationId).catch(() => ({ data: null })),
        reflectionApi.getBiasHeatmap(simulationId).catch(() => ({ data: null })),
        reflectionApi.getRationaleReview(simulationId).catch(() => ({ data: null })),
        reflectionApi.getCalibration(simulationId).catch(() => ({ data: null })),
        reflectionApi.getOutcomeDistribution(simulationId).catch(() => ({ data: null })),
        reflectionApi.getBiasClassifier(simulationId).catch(() => ({ data: null })),
        reflectionApi.getConfidenceCalibration(simulationId).catch(() => ({ data: null })),
        reflectionApi.getAiMetadata(simulationId).catch(() => ({ data: null })),
      ]);

      if (cfRes.data?.length) setCounterfactuals(cfRes.data);
      setWhyData(whyRes.data);
      setProData(proRes.data);
      if (!coaching) setCoaching(coachRes.data);
      setBiasHeatmap(heatmapRes.data);
      setRationaleReview(rationaleRes.data);
      setCalibration(calRes.data);
      setOutcomeDistribution(outcomeRes.data);
      setBiasClassifier(biasClsRes.data);
      setConfidenceCalibration(confCalRes.data);
      setAiMetadata(metaRes.data);
    } catch (err) {
      console.error('Failed to load secondary data:', err);
      setSecondaryError(true);
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

  // Summary mode computed values
  const topBias = reflection.patterns_detected?.length > 0
    ? reflection.patterns_detected.slice().sort((a, b) => b.confidence - a.confidence)[0]
    : null;
  const luckPct = Math.round((reflection.luck_factor || 0) * 100);
  const skillPct = Math.round((reflection.skill_factor || 0) * 100);
  const processLabel = processScore >= 70 ? 'strong' : processScore >= 50 ? 'average' : 'weak';
  const driverLabel = skillPct > luckPct ? 'your decisions drove the outcome' : skillPct === luckPct ? 'luck and skill contributed equally' : 'market conditions drove the outcome more than your choices';

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Outcome Header */}
      <div className={clsx(
        'card text-center overflow-hidden relative',
        isProfit ? 'bg-gradient-to-b from-green-50 to-white' : 'bg-gradient-to-b from-red-50 to-white'
      )}>
        <div className={clsx(
          'inline-flex items-center justify-center w-20 h-20 rounded-2xl mb-4 shadow-lg',
          isProfit ? 'bg-gradient-to-br from-green-500 to-emerald-600' : 'bg-gradient-to-br from-red-500 to-rose-600'
        )}>
          {isProfit ? (
            <TrendingUp className="h-10 w-10 text-white" />
          ) : (
            <TrendingDown className="h-10 w-10 text-white" />
          )}
        </div>
        <h1 className={clsx(
          'text-4xl font-black mb-2',
          isProfit ? 'text-green-700' : 'text-red-600'
        )}>
          {reflection.outcome_summary}
        </h1>
        <div className="flex items-center justify-center gap-4 mt-3">
          <div className={clsx(
            'px-4 py-1.5 rounded-full text-sm font-bold',
            processScore >= 70 ? 'bg-green-100 text-green-700' :
              processScore >= 50 ? 'bg-amber-100 text-amber-700' : 'bg-red-100 text-red-700'
          )}>
            Process: {processScore >= 70 ? 'Strong' : processScore >= 50 ? 'Average' : 'Risky'} ({Math.round(processScore)}/100)
          </div>

          {/* AI Source Badge */}
          {aiMetadata && (
            <div className={clsx(
              'px-3 py-1.5 rounded-full text-xs font-bold inline-flex items-center gap-1.5',
              aiMetadata.source === 'gemini'
                ? 'bg-blue-100 text-blue-700 border border-blue-200'
                : 'bg-gray-100 text-gray-600 border border-gray-200'
            )}>
              {aiMetadata.source === 'gemini' ? (
                <>
                  <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none">
                    <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  Gemini 3 Pro
                </>
              ) : (
                <>
                  <Zap className="w-3 h-3" />
                  Heuristic Analysis
                </>
              )}
            </div>
          )}
        </div>
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

      {/* AI Reasoning Panel — shows Gemini's thinking process */}
      {aiMetadata?.thinking && Object.keys(aiMetadata.thinking).length > 0 && (
        <div className="card p-0 overflow-hidden">
          <button
            onClick={() => setShowThinking(!showThinking)}
            className="w-full flex items-center justify-between p-4 hover:bg-brand-lavender/20 transition-colors"
          >
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-blue-100 flex items-center justify-center">
                <Brain className="h-4 w-4 text-blue-600" />
              </div>
              <div className="text-left">
                <span className="text-sm font-semibold text-brand-navy">AI Reasoning</span>
                <span className="text-xs text-brand-navy/50 ml-2">Gemini's internal thought process</span>
              </div>
            </div>
            <ChevronRight className={clsx(
              'h-5 w-5 text-brand-navy/40 transition-transform',
              showThinking && 'rotate-90'
            )} />
          </button>

          {showThinking && (
            <div className="border-t border-brand-navy/10 p-4 space-y-4 animate-fadeIn">
              {Object.entries(aiMetadata.thinking).map(([key, text]) => (
                <div key={key}>
                  <p className="text-xs font-bold text-blue-600 uppercase tracking-wider mb-1.5">
                    {key.replace(/_/g, ' ')}
                  </p>
                  <div className="bg-gradient-to-r from-blue-50/80 to-transparent rounded-lg p-3 border-l-2 border-blue-300">
                    <p className="text-sm text-brand-navy/70 whitespace-pre-wrap leading-relaxed font-mono">
                      {text.length > 800 ? text.slice(0, 800) + '...' : text}
                    </p>
                  </div>
                </div>
              ))}
              <p className="text-[10px] text-brand-navy/30 italic">
                Extended thinking enabled via Gemini 3 Pro — the model reasons through your decision trace before producing structured analysis.
              </p>
            </div>
          )}
        </div>
      )}

      {/* At a Glance Summary (shown when full analysis is collapsed) */}
      {!showFullAnalysis && (
        <div className="card">
          {/* Narrative sentence */}
          <p className="text-brand-navy/80 leading-relaxed mb-5">
            You made <span className={clsx('font-bold', processScore >= 70 ? 'text-green-700' : processScore >= 50 ? 'text-amber-600' : 'text-red-600')}>{processLabel} decisions</span>
            {topBias ? (
              <>, with <span className="font-bold text-amber-600">{(topBias.pattern_name || 'unknown bias').replace(/_/g, ' ')}</span> as your strongest bias ({Math.round((topBias.confidence || 0) * 100)}% confidence)</>
            ) : (
              <> with no strong biases detected</>
            )}
            . Overall, <span className="font-semibold text-brand-navy">{driverLabel}</span> ({skillPct}% skill / {luckPct}% luck).
          </p>

          {/* Compact stat tiles */}
          <div className="grid sm:grid-cols-3 gap-4 mb-5">
            <div className="flex items-center gap-3 p-3 bg-brand-lavender/30 rounded-xl">
              <div className={clsx(
                'w-12 h-12 rounded-xl flex items-center justify-center text-lg font-black text-white',
                processScore >= 70 ? 'bg-green-500' : processScore >= 50 ? 'bg-amber-500' : 'bg-red-500'
              )}>
                {Math.round(processScore)}
              </div>
              <div>
                <p className="text-xs text-brand-navy/50 uppercase tracking-wide">Process</p>
                <p className="text-sm font-semibold text-brand-navy">
                  {processScore >= 70 ? 'Strong decisions' : processScore >= 50 ? 'Average process' : 'Needs improvement'}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3 p-3 bg-brand-lavender/30 rounded-xl">
              <div className="w-12 h-12 rounded-xl bg-amber-100 flex items-center justify-center">
                <Brain className="h-6 w-6 text-amber-600" />
              </div>
              <div>
                <p className="text-xs text-brand-navy/50 uppercase tracking-wide">Top Bias</p>
                <p className="text-sm font-semibold text-brand-navy capitalize">
                  {topBias ? (topBias.pattern_name || 'unknown bias').replace(/_/g, ' ') : 'None detected'}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3 p-3 bg-brand-lavender/30 rounded-xl">
              <div className="w-12 h-12 rounded-xl bg-cyan-100 flex items-center justify-center">
                <Shuffle className="h-6 w-6 text-cyan-600" />
              </div>
              <div>
                <p className="text-xs text-brand-navy/50 uppercase tracking-wide">Luck / Skill</p>
                <p className="text-sm font-semibold text-brand-navy">
                  {luckPct}% / {skillPct}%
                </p>
              </div>
            </div>
          </div>

          <button
            onClick={() => setShowFullAnalysis(true)}
            className="w-full py-3 rounded-xl bg-brand-navy text-white font-semibold hover:bg-brand-navy-light transition-colors flex items-center justify-center gap-2"
          >
            <span>Show Full Analysis</span>
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* Full Analysis (expandable) */}
      {showFullAnalysis && (<div className="space-y-6 animate-fadeIn">

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
          <div className="card animate-pulse space-y-4">
            <div className="h-5 w-40 bg-gray-200 rounded-lg" />
            <div className="h-24 w-24 bg-gray-200 rounded-full mx-auto" />
            <div className="space-y-2">
              <div className="h-3 bg-gray-100 rounded-lg w-full" />
              <div className="h-3 bg-gray-100 rounded-lg w-3/4" />
              <div className="h-3 bg-gray-100 rounded-lg w-5/6" />
            </div>
          </div>
          <div className="card animate-pulse space-y-4">
            <div className="h-5 w-48 bg-gray-200 rounded-lg" />
            <div className="flex items-end gap-1 h-24">
              {[40, 60, 80, 50, 70, 90, 45, 55].map((h, i) => (
                <div key={i} className="flex-1 bg-gray-200 rounded-t" style={{ height: `${h}%` }} />
              ))}
            </div>
            <div className="h-3 bg-gray-100 rounded-lg w-2/3 mx-auto" />
          </div>
        </div>
      ) : (calibration || outcomeDistribution) && (
        <SectionErrorBoundary section="Calibration & Distribution">
        <div className="grid md:grid-cols-2 gap-6">
          {calibration && <CalibrationCard data={calibration} />}
          {outcomeDistribution && <OutcomeDistributionChart data={outcomeDistribution} />}
        </div>
        </SectionErrorBoundary>
      )}

      {/* Bias Heatmap Timeline */}
      {secondaryLoading && !biasHeatmap ? (
        <div className="card animate-pulse space-y-4">
          <div className="h-5 w-52 bg-gray-200 rounded-lg" />
          <div className="grid grid-cols-8 gap-2">
            {Array.from({ length: 48 }).map((_, i) => (
              <div key={i} className="h-6 bg-gray-100 rounded" />
            ))}
          </div>
        </div>
      ) : biasHeatmap && biasHeatmap.timeline?.length > 0 && (
        <SectionErrorBoundary section="Bias Heatmap">
          <BiasHeatmap data={biasHeatmap} />
        </SectionErrorBoundary>
      )}

      {/* AI Unavailable Banner */}
      {secondaryError && !whyData && !proData && !biasHeatmap && (
        <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 flex items-start gap-3">
          <AlertTriangle className="h-5 w-5 text-amber-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-amber-800">Some AI analysis is unavailable</p>
            <p className="text-xs text-amber-700 mt-1">
              The AI service may be temporarily unavailable. Core analysis is shown below.
              Advanced sections like bias heatmap and pro comparison may not appear.
            </p>
          </div>
        </div>
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
        <SectionErrorBoundary section="Rationale Review">
        <div className="card">
          <button
            onClick={() => setShowRationale(!showRationale)}
            className="w-full flex items-center justify-between group"
          >
            <h3 className="text-lg font-semibold text-brand-navy flex items-center space-x-2">
              <div className="w-8 h-8 rounded-lg bg-brand-lavender flex items-center justify-center">
                <FileText className="h-4 w-4 text-brand-navy" />
              </div>
              <span>Rationale Review</span>
              <span className={clsx(
                'text-xs font-bold px-2.5 py-1 rounded-full',
                rationaleReview.overall_reasoning_quality >= 4 ? 'bg-green-100 text-green-700' :
                  rationaleReview.overall_reasoning_quality >= 3 ? 'bg-yellow-100 text-amber-600' :
                    'bg-red-100 text-red-600'
              )}>
                {rationaleReview.overall_reasoning_quality}/5
              </span>
            </h3>
            <ChevronRight className={clsx(
              'h-5 w-5 text-brand-navy/40 transition-transform duration-200 group-hover:text-brand-navy',
              showRationale && 'rotate-90'
            )} />
          </button>
          <p className="text-brand-navy/60 text-sm mt-2 ml-10">
            AI critique of your stated reasoning for each decision
          </p>

          {showRationale && (
            <div className="mt-4 space-y-4 animate-fadeIn">
              {rationaleReview.reviews.map((review, i) => (
                <RationaleCritiqueCard key={i} review={review} />
              ))}
            </div>
          )}
        </div>
        </SectionErrorBoundary>
      )}

      {/* Counterfactuals */}
      <div className="card">
        <button
          onClick={() => setShowCounterfactuals(!showCounterfactuals)}
          className="w-full flex items-center justify-between group"
        >
          <h3 className="text-lg font-semibold text-brand-navy flex items-center space-x-2">
            <div className="w-8 h-8 rounded-lg bg-purple-100 flex items-center justify-center">
              <GitBranch className="h-4 w-4 text-purple-700" />
            </div>
            <span>Alternate Timelines</span>
            <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-brand-lavender text-brand-navy/60">
              {counterfactuals.length}
            </span>
          </h3>
          <ChevronRight className={clsx(
            'h-5 w-5 text-brand-navy/40 transition-transform duration-200 group-hover:text-brand-navy',
            showCounterfactuals && 'rotate-90'
          )} />
        </button>
        <p className="text-brand-navy/60 text-sm mt-2 ml-10">
          What could have happened with the same decisions in different market conditions?
        </p>

        {showCounterfactuals && (
          <div className="mt-4 space-y-4 animate-fadeIn">
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
        <SectionErrorBoundary section="Why This Decision">
        <div className="card">
          <button
            onClick={() => setShowWhy(!showWhy)}
            className="w-full flex items-center justify-between group"
          >
            <h3 className="text-lg font-semibold text-brand-navy flex items-center space-x-2">
              <div className="w-8 h-8 rounded-lg bg-orange-100 flex items-center justify-center">
                <HelpCircle className="h-4 w-4 text-orange-600" />
              </div>
              <span>Why This Decision?</span>
              <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-orange-100 text-orange-600">
                {whyData.explanations.length} decisions
              </span>
            </h3>
            <ChevronRight className={clsx(
              'h-5 w-5 text-brand-navy/40 transition-transform duration-200 group-hover:text-brand-navy',
              showWhy && 'rotate-90'
            )} />
          </button>
          <p className="text-brand-navy/60 text-sm mt-2 ml-10">
            {whyData.overall_narrative}
          </p>

          {showWhy && (
            <div className="mt-4 space-y-4 animate-fadeIn">
              {whyData.explanations.map((exp, i) => (
                <div key={i}>
                  <div className="p-4 bg-brand-lavender/30 rounded-lg border-l-4 border-orange-500">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium text-brand-navy">
                        Decision #{(exp.decision_index ?? 0) + 1}: {(exp.decision_type || 'action').toUpperCase()} at {exp.timestamp_seconds ?? 0}s
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
                      Detected: {(exp.detected_bias || 'unknown').replace(/_/g, ' ')}
                    </p>
                    <p className="text-brand-navy/70 text-sm mb-3">{exp.explanation}</p>
                    <div className="space-y-1 mb-3">
                      {(exp.evidence_from_actions || []).map((e, j) => (
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
        </SectionErrorBoundary>
      )}

      {/* What Would a Pro Do? */}
      {proData && proData.pro_decisions?.length > 0 && (
        <SectionErrorBoundary section="Pro Comparison">
        <div className="card">
          <button
            onClick={() => setShowPro(!showPro)}
            className="w-full flex items-center justify-between group"
          >
            <h3 className="text-lg font-semibold text-brand-navy flex items-center space-x-2">
              <div className="w-8 h-8 rounded-lg bg-cyan-100 flex items-center justify-center">
                <UserCheck className="h-4 w-4 text-cyan-700" />
              </div>
              <span>What Would a Pro Do?</span>
            </h3>
            <ChevronRight className={clsx(
              'h-5 w-5 text-brand-navy/40 transition-transform duration-200 group-hover:text-brand-navy',
              showPro && 'rotate-90'
            )} />
          </button>
          <p className="text-brand-navy/60 text-sm mt-2 ml-10">
            Side-by-side comparison with an experienced trader's approach
          </p>

          {showPro && (
            <div className="mt-4 space-y-4 animate-fadeIn">
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
              {(proData.pro_decisions || []).map((pd, i) => (
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

              {/* Algorithmic Baseline */}
              {proData.algorithmic_baseline && (
                <div className="mt-4">
                  <button
                    onClick={() => setShowAlgoBaseline(!showAlgoBaseline)}
                    className="flex items-center gap-2 text-sm font-semibold text-brand-navy hover:text-brand-navy-light transition-colors"
                  >
                    <Zap className="h-4 w-4" />
                    Algorithmic Trader Baseline
                    <ChevronRight className={clsx('h-4 w-4 transition-transform', showAlgoBaseline && 'rotate-90')} />
                  </button>

                  {showAlgoBaseline && (
                    <div className="mt-3 space-y-3 animate-fadeIn">
                      <p className="text-xs text-brand-navy/60">{proData.algorithmic_baseline.strategy_description}</p>

                      {/* Algo outcome comparison */}
                      <div className="grid grid-cols-3 gap-3">
                        <div className="p-3 bg-brand-lavender/30 rounded-lg text-center">
                          <p className="text-xs text-brand-navy/60 mb-1">Your P&L</p>
                          <p className={clsx('text-lg font-bold', (proData.user_final_outcome?.profit_loss || 0) >= 0 ? 'text-green-700' : 'text-red-600')}>
                            ${(proData.user_final_outcome?.profit_loss || 0).toFixed(2)}
                          </p>
                        </div>
                        <div className="p-3 bg-amber-50 rounded-lg text-center border border-amber-200">
                          <p className="text-xs text-amber-700 mb-1">Algo P&L</p>
                          <p className={clsx('text-lg font-bold', (proData.algorithmic_baseline.algo_final_outcome?.profit_loss || 0) >= 0 ? 'text-green-700' : 'text-red-600')}>
                            ${(proData.algorithmic_baseline.algo_final_outcome?.profit_loss || 0).toFixed(2)}
                          </p>
                        </div>
                        <div className="p-3 bg-cyan-50 rounded-lg text-center border border-brand-blue/30">
                          <p className="text-xs text-cyan-700 mb-1">AI Pro P&L</p>
                          <p className={clsx('text-lg font-bold', (proData.pro_final_outcome?.profit_loss || 0) >= 0 ? 'text-green-700' : 'text-red-600')}>
                            ${(proData.pro_final_outcome?.profit_loss || 0).toFixed(2)}
                          </p>
                        </div>
                      </div>

                      {/* Algo trades */}
                      <div className="text-xs text-brand-navy/60">
                        <span className="font-semibold">{proData.algorithmic_baseline.algo_final_outcome?.total_trades || 0}</span> trades,{' '}
                        <span className="font-semibold">${proData.algorithmic_baseline.algo_final_outcome?.cumulative_fees?.toFixed(2) || '0.00'}</span> in fees
                      </div>

                      {/* Strategy rules */}
                      <div className="grid grid-cols-3 gap-2">
                        {proData.algorithmic_baseline.rules?.map((r, i) => (
                          <div key={i} className="text-xs p-2 bg-gray-50 rounded border border-gray-200">
                            <span className="font-medium text-brand-navy">{(r.rule || '').replace(/_/g, ' ')}</span>
                            <p className="text-brand-navy/50 mt-0.5">{r.description}</p>
                          </div>
                        ))}
                      </div>

                      {/* Algo decision log */}
                      {proData.algorithmic_baseline.algo_decisions?.length > 0 && (
                        <details className="text-xs">
                          <summary className="cursor-pointer font-semibold text-brand-navy/70 hover:text-brand-navy">
                            View {proData.algorithmic_baseline.algo_decisions.length} algo trades
                          </summary>
                          <div className="mt-2 space-y-1 max-h-40 overflow-y-auto">
                            {proData.algorithmic_baseline.algo_decisions.map((d, i) => (
                              <div key={i} className="flex items-center gap-2 p-1.5 bg-gray-50 rounded">
                                <span className="text-brand-navy/50">t={d.time}s</span>
                                <span className={clsx('font-medium', d.action === 'buy' ? 'text-green-700' : 'text-red-600')}>
                                  {d.action.toUpperCase()}
                                </span>
                                <span className="text-brand-navy/60">{d.amount?.toFixed?.(2)} @ ${d.price}</span>
                                <span className="text-brand-navy/40 ml-auto">{d.reason}</span>
                              </div>
                            ))}
                          </div>
                        </details>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
        </SectionErrorBoundary>
      )}

      {/* Bias Classifier */}
      {biasClassifier && biasClassifier.per_decision?.length > 0 && (
        <SectionErrorBoundary section="Bias Classifier">
        <div className="card p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl bg-purple-100 flex items-center justify-center">
              <BarChart2 className="h-5 w-5 text-purple-700" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-brand-navy">Bias Classifier</h3>
              <p className="text-brand-navy/60 text-xs">Rule-based feature analysis of your decisions</p>
            </div>
          </div>

          {/* Top biases */}
          {biasClassifier.top_biases?.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-4">
              {biasClassifier.top_biases.map((b, i) => (
                <div key={i} className={clsx(
                  'px-3 py-1.5 rounded-full text-xs font-medium',
                  b.score > 0.5 ? 'bg-red-100 text-red-700' :
                  b.score > 0.3 ? 'bg-amber-100 text-amber-700' :
                  'bg-gray-100 text-gray-600'
                )}>
                  #{b.rank} {(b.bias || '').replace(/_/g, ' ')} ({((b.score || 0) * 100).toFixed(0)}%)
                </div>
              ))}
            </div>
          )}

          {/* Aggregate scores bar chart */}
          <div className="space-y-2 mb-4">
            {Object.entries(biasClassifier.aggregate_scores || {})
              .sort(([,a], [,b]) => b - a)
              .map(([bias, score]) => (
                <div key={bias} className="flex items-center gap-2">
                  <span className="text-xs text-brand-navy/70 w-28 text-right">{bias.replace(/_/g, ' ')}</span>
                  <div className="flex-1 h-3 bg-gray-100 rounded-full overflow-hidden">
                    <div
                      className={clsx(
                        'h-full rounded-full transition-all duration-500',
                        score > 0.5 ? 'bg-red-400' : score > 0.3 ? 'bg-amber-400' : 'bg-gray-300'
                      )}
                      style={{ width: `${Math.max(score * 100, 2)}%` }}
                    />
                  </div>
                  <span className="text-xs text-brand-navy/50 w-10">{(score * 100).toFixed(0)}%</span>
                </div>
              ))}
          </div>

          {/* Gemini comparison */}
          {biasClassifier.gemini_comparison && (
            <div className="p-3 bg-purple-50 rounded-lg border border-purple-200">
              <p className="text-xs font-semibold text-purple-700 mb-2">
                Classifier vs AI Agreement: {((biasClassifier.gemini_comparison.agreement_rate || 0) * 100).toFixed(0)}%
              </p>
              <div className="space-y-1">
                {biasClassifier.gemini_comparison.details?.map((d, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs">
                    {d.agreement ? (
                      <CheckCircle className="h-3.5 w-3.5 text-green-600 flex-shrink-0" />
                    ) : (
                      <AlertTriangle className="h-3.5 w-3.5 text-amber-500 flex-shrink-0" />
                    )}
                    <span className="text-brand-navy/70">{(d.bias || '').replace(/_/g, ' ')}</span>
                    <span className="text-brand-navy/40 ml-auto">{d.note}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Feature importance */}
          {biasClassifier.feature_importance && biasClassifier.top_biases?.[0] && (
            <details className="mt-3 text-xs">
              <summary className="cursor-pointer font-semibold text-brand-navy/70 hover:text-brand-navy">
                Feature importance for top bias: {(biasClassifier.top_biases[0]?.bias || '').replace(/_/g, ' ')}
              </summary>
              <div className="mt-2 space-y-1">
                {(biasClassifier.feature_importance[biasClassifier.top_biases[0].bias] || []).map(([feat, imp], i) => (
                  <div key={i} className="flex items-center gap-2">
                    <span className="text-brand-navy/60 w-40">{(feat || '').replace(/_/g, ' ')}</span>
                    <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                      <div className="h-full bg-purple-400 rounded-full" style={{ width: `${imp * 100}%` }} />
                    </div>
                    <span className="text-brand-navy/40 w-10">{(imp * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
        </SectionErrorBoundary>
      )}

      {/* Confidence Calibration */}
      {confidenceCalibration && (confidenceCalibration.calibrated_patterns?.length > 0 || confidenceCalibration.abstained_patterns?.length > 0) && (
        <SectionErrorBoundary section="Confidence Calibration">
        <div className="card p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl bg-teal-100 flex items-center justify-center">
              <Search className="h-5 w-5 text-teal-700" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-brand-navy">Evidence Confidence</h3>
              <p className="text-brand-navy/60 text-xs">{confidenceCalibration.summary}</p>
            </div>
            <span className={clsx(
              'ml-auto text-xs px-2.5 py-1 rounded-full font-medium',
              confidenceCalibration.overall_evidence_quality === 'strong' ? 'bg-green-100 text-green-700' :
              confidenceCalibration.overall_evidence_quality === 'moderate' ? 'bg-amber-100 text-amber-700' :
              'bg-gray-100 text-gray-600'
            )}>
              {confidenceCalibration.overall_evidence_quality} evidence
            </span>
          </div>

          {/* Calibrated patterns */}
          <div className="space-y-3">
            {confidenceCalibration.calibrated_patterns?.map((p, i) => (
              <div key={i} className="p-3 bg-brand-lavender/20 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-brand-navy">{(p.pattern || '').replace(/_/g, ' ')}</span>
                  <span className={clsx(
                    'text-xs px-2 py-0.5 rounded-full font-medium',
                    p.confidence_level === 'high' ? 'bg-green-100 text-green-700' :
                    p.confidence_level === 'medium' ? 'bg-amber-100 text-amber-700' :
                    'bg-gray-100 text-gray-600'
                  )}>
                    {p.confidence_level}
                  </span>
                </div>

                {/* Confidence bars */}
                <div className="grid grid-cols-2 gap-3 mb-2">
                  <div>
                    <p className="text-xs text-brand-navy/50 mb-1">AI confidence</p>
                    <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                      <div className="h-full bg-blue-400 rounded-full" style={{ width: `${(p.gemini_confidence || 0) * 100}%` }} />
                    </div>
                  </div>
                  <div>
                    <p className="text-xs text-brand-navy/50 mb-1">Evidence score</p>
                    <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                      <div className="h-full bg-teal-400 rounded-full" style={{ width: `${(p.evidence_score || 0) * 100}%` }} />
                    </div>
                  </div>
                </div>

                <p className="text-xs text-brand-navy/60">{p.reasoning}</p>

                {/* Evidence details */}
                <details className="mt-2">
                  <summary className="text-xs cursor-pointer text-brand-navy/50 hover:text-brand-navy">
                    {p.evidence_details?.filter(e => e.matched).length}/{p.evidence_details?.length} signals matched
                  </summary>
                  <div className="mt-1 space-y-0.5">
                    {p.evidence_details?.map((e, j) => (
                      <div key={j} className="flex items-start gap-1.5 text-xs">
                        {e.matched ? (
                          <CheckCircle className="h-3 w-3 text-green-600 flex-shrink-0 mt-0.5" />
                        ) : (
                          <X className="h-3 w-3 text-gray-300 flex-shrink-0 mt-0.5" />
                        )}
                        <span className={e.matched ? 'text-brand-navy/70' : 'text-brand-navy/40'}>{e.signal}</span>
                        {e.detail && <span className="text-brand-navy/40 ml-auto">{e.detail}</span>}
                      </div>
                    ))}
                  </div>
                </details>
              </div>
            ))}

            {/* Abstained patterns */}
            {confidenceCalibration.abstained_patterns?.length > 0 && (
              <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                <p className="text-xs font-semibold text-gray-600 mb-2">
                  Insufficient Evidence ({confidenceCalibration.abstained_patterns.length} pattern{confidenceCalibration.abstained_patterns.length > 1 ? 's' : ''})
                </p>
                {confidenceCalibration.abstained_patterns.map((p, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs text-gray-500 mb-1">
                    <HelpCircle className="h-3 w-3 flex-shrink-0" />
                    <span>{(p.pattern || '').replace(/_/g, ' ')}</span>
                    <span className="ml-auto">AI said {((p.gemini_confidence || 0) * 100).toFixed(0)}% but no evidence found</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
        </SectionErrorBoundary>
      )}

      {/* Collapse button */}
      <div className="text-center">
        <button
          onClick={() => setShowFullAnalysis(false)}
          className="text-sm text-brand-navy/50 hover:text-brand-navy transition-colors"
        >
          Collapse to summary
        </button>
      </div>

      </div>)}

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
            <div ref={shareCardRef} className="rounded-2xl overflow-hidden shadow-2xl border border-brand-blue/30">
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
                          {(p.pattern_name || 'unknown').replace(/_/g, ' ')}
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

            {/* Action Buttons */}
            <div className="flex gap-2 mt-3">
              <button
                onClick={saveAsImage}
                disabled={savingImage}
                className="btn btn-primary flex-1 flex items-center justify-center gap-2"
              >
                <Download className="h-5 w-5" />
                {savingImage ? 'Saving...' : 'Save as Image'}
              </button>
              <button
                onClick={copyShareText}
                className="btn btn-secondary flex-1 flex items-center justify-center gap-2"
              >
                {copied ? <Check className="h-5 w-5" /> : <Copy className="h-5 w-5" />}
                {copied ? 'Copied!' : 'Copy Text'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
