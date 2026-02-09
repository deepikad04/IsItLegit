import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { profileApi } from '../api/client';
import {
  RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer,
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid
} from 'recharts';
import {
  User, TrendingUp, TrendingDown, Target, Brain, Award,
  Calendar, Activity, ChevronRight, AlertTriangle, CheckCircle,
  BookOpen, ThumbsUp, ThumbsDown, Shield, Flame, ToggleLeft, ToggleRight
} from 'lucide-react';
import clsx from 'clsx';

function StatCard({ icon: Icon, label, value, trend, trendValue }) {
  return (
    <div className="card">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-lg bg-brand-lavender flex items-center justify-center">
            <Icon className="h-5 w-5 text-brand-navy" />
          </div>
          <div>
            <p className="text-brand-navy/60 text-sm">{label}</p>
            <p className="text-2xl font-bold text-brand-navy">{value}</p>
          </div>
        </div>
        {trend && (
          <div className={clsx(
            'flex items-center space-x-1 text-sm',
            trend === 'up' ? 'text-green-700' : trend === 'down' ? 'text-red-600' : 'text-brand-navy/60'
          )}>
            {trend === 'up' ? <TrendingUp className="h-4 w-4" /> :
             trend === 'down' ? <TrendingDown className="h-4 w-4" /> : null}
            <span>{trendValue}</span>
          </div>
        )}
      </div>
    </div>
  );
}

function BiasPatternRow({ pattern }) {
  const getColor = () => {
    if (pattern.score >= 0.7) return 'text-red-600 bg-red-500';
    if (pattern.score >= 0.4) return 'text-amber-600 bg-amber-500';
    return 'text-green-700 bg-green-500';
  };

  const getTrendIcon = () => {
    if (pattern.trend === 'improving') return <TrendingDown className="h-4 w-4 text-green-700" />;
    if (pattern.trend === 'worsening') return <TrendingUp className="h-4 w-4 text-red-600" />;
    return null;
  };

  return (
    <div className="flex items-center justify-between py-3 border-b border-brand-blue/20 last:border-0">
      <div className="flex-1">
        <p className="text-brand-navy font-medium capitalize">{(pattern.name || 'unknown').replace(/_/g, ' ')}</p>
        <p className="text-brand-navy/60 text-sm">{pattern.description}</p>
      </div>
      <div className="flex items-center space-x-4">
        {getTrendIcon()}
        <div className="w-32">
          <div className="flex justify-between text-xs mb-1">
            <span className={getColor().split(' ')[0]}>{Math.round(pattern.score * 100)}%</span>
          </div>
          <div className="h-2 bg-brand-blue/20 rounded-full overflow-hidden">
            <div
              className={clsx('h-full rounded-full', getColor().split(' ')[1])}
              style={{ width: `${pattern.score * 100}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function HistoryItem({ item }) {
  const isProfit = item.profit_loss > 0;

  return (
    <div className="flex items-center justify-between py-3 border-b border-brand-blue/20 last:border-0">
      <div className="flex items-center space-x-3">
        <div className={clsx(
          'w-10 h-10 rounded-lg flex items-center justify-center',
          isProfit ? 'bg-green-100' : 'bg-red-100'
        )}>
          {isProfit ? (
            <TrendingUp className="h-5 w-5 text-green-700" />
          ) : (
            <TrendingDown className="h-5 w-5 text-red-600" />
          )}
        </div>
        <div>
          <p className="text-brand-navy font-medium">{item.scenario_name}</p>
          <p className="text-brand-navy/60 text-sm">
            {new Date(item.completed_at).toLocaleDateString()}
          </p>
        </div>
      </div>
      <div className="text-right">
        <p className={clsx(
          'font-bold',
          isProfit ? 'text-green-700' : 'text-red-600'
        )}>
          {isProfit ? '+' : ''}{item.profit_loss.toFixed(2)}
        </p>
        <p className="text-brand-navy/60 text-sm">
          Process: {item.process_quality_score}%
        </p>
      </div>
      <Link
        to={`/reflection/${item.id}`}
        className="ml-4 text-brand-navy/60 hover:text-brand-navy"
      >
        <ChevronRight className="h-5 w-5" />
      </Link>
    </div>
  );
}

const BIAS_BADGES = [
  { id: 'anchoring_breaker', label: 'Anchoring Breaker', icon: '⚓', desc: 'Made a decision without anchoring to initial price', bias: 'anchoring', threshold: 0.3 },
  { id: 'fomo_fighter', label: 'FOMO Fighter', icon: '🛡️', desc: 'Resisted fear of missing out in a rally', bias: 'fomo', threshold: 0.3 },
  { id: 'loss_master', label: 'Loss Master', icon: '💎', desc: 'Cut losses without emotional delay', bias: 'loss_aversion', threshold: 0.3 },
  { id: 'cool_headed', label: 'Cool Headed', icon: '🧊', desc: 'Maintained low impulsivity under pressure', bias: 'impulsivity', threshold: 0.3 },
  { id: 'independent_mind', label: 'Independent Mind', icon: '🧠', desc: 'Decided against the social consensus', bias: 'social_proof_reliance', threshold: 0.3 },
  { id: 'calibrated', label: 'Well Calibrated', icon: '🎯', desc: 'Confidence matched outcomes accurately', bias: 'overconfidence', threshold: 0.3 },
  { id: 'veteran', label: 'Veteran', icon: '🏅', desc: 'Completed 10+ simulations', simCount: 10 },
  { id: 'streak_3', label: 'On a Roll', icon: '🔥', desc: '3+ simulations in a row', streak: 3 },
];

function BadgeGrid({ profile }) {
  const biasPatterns = profile?.bias_patterns || [];
  const earned = BIAS_BADGES.filter(badge => {
    if (badge.simCount) return (profile?.total_simulations_analyzed || 0) >= badge.simCount;
    if (badge.streak) return (profile?.total_simulations_analyzed || 0) >= badge.streak;
    if (badge.bias) {
      const pattern = biasPatterns.find(p => p.name === badge.bias);
      return pattern && pattern.score <= badge.threshold;
    }
    return false;
  });
  const locked = BIAS_BADGES.filter(b => !earned.includes(b));

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-brand-navy mb-4 flex items-center space-x-2">
        <Award className="h-5 w-5 text-brand-navy" />
        <span>Bias Badges</span>
        <span className="text-sm text-brand-navy/60 font-normal ml-2">
          {earned.length}/{BIAS_BADGES.length}
        </span>
      </h3>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {earned.map(badge => (
          <div key={badge.id} className="p-3 bg-brand-lavender/40 border border-brand-navy/20 rounded-lg text-center" role="img" aria-label={`Earned badge: ${badge.label}`}>
            <span className="text-2xl">{badge.icon}</span>
            <p className="text-brand-navy text-sm font-medium mt-1">{badge.label}</p>
            <p className="text-brand-navy/60 text-xs mt-0.5">{badge.desc}</p>
          </div>
        ))}
        {locked.map(badge => (
          <div key={badge.id} className="p-3 bg-brand-cream/50 border border-brand-blue/20 rounded-lg text-center opacity-40" role="img" aria-label={`Locked badge: ${badge.label}`}>
            <span className="text-2xl grayscale">🔒</span>
            <p className="text-brand-navy/60 text-sm font-medium mt-1">{badge.label}</p>
            <p className="text-brand-blue text-xs mt-0.5">{badge.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function PlaybookSection({ playbook }) {
  if (!playbook) return null;

  return (
    <div className="card bg-gradient-to-br from-brand-lavender/40 to-brand-blue/20 border border-brand-navy/20">
      <div className="flex items-center space-x-3 mb-6">
        <div className="w-10 h-10 rounded-lg bg-brand-lavender flex items-center justify-center">
          <BookOpen className="h-5 w-5 text-brand-navy" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-brand-navy">Your Decision Playbook</h3>
          <p className="text-brand-navy/60 text-sm">Personal rules based on your patterns</p>
        </div>
      </div>

      {/* Key Rules */}
      {playbook.key_rules?.length > 0 && (
        <div className="mb-6">
          <p className="text-sm text-brand-navy/60 uppercase mb-3 flex items-center space-x-2">
            <Shield className="h-4 w-4" />
            <span>Key Rules</span>
          </p>
          <div className="space-y-2">
            {playbook.key_rules.map((rule, i) => (
              <div key={i} className="p-3 bg-white/60 rounded-lg border border-brand-blue/20">
                <p className="text-brand-navy text-sm font-medium">{rule}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Do / Don't columns */}
      <div className="grid md:grid-cols-2 gap-4">
        {/* Do's */}
        <div>
          <p className="text-sm text-green-700 uppercase mb-3 flex items-center space-x-2">
            <ThumbsUp className="h-4 w-4" />
            <span>Do</span>
          </p>
          <div className="space-y-2">
            {playbook.dos?.map((item, i) => (
              <div key={i} className="flex items-start space-x-2 p-2 bg-green-50 rounded-lg">
                <CheckCircle className="h-4 w-4 text-green-700 flex-shrink-0 mt-0.5" />
                <p className="text-brand-navy text-sm">{item}</p>
              </div>
            ))}
            {(!playbook.dos || playbook.dos.length === 0) && (
              <p className="text-brand-navy/60 text-sm">Complete more simulations to generate</p>
            )}
          </div>
        </div>

        {/* Don'ts */}
        <div>
          <p className="text-sm text-red-600 uppercase mb-3 flex items-center space-x-2">
            <ThumbsDown className="h-4 w-4" />
            <span>Don't</span>
          </p>
          <div className="space-y-2">
            {playbook.donts?.map((item, i) => (
              <div key={i} className="flex items-start space-x-2 p-2 bg-red-50 rounded-lg">
                <AlertTriangle className="h-4 w-4 text-red-600 flex-shrink-0 mt-0.5" />
                <p className="text-brand-navy text-sm">{item}</p>
              </div>
            ))}
            {(!playbook.donts || playbook.donts.length === 0) && (
              <p className="text-brand-navy/60 text-sm">Complete more simulations to generate</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Profile() {
  const [profile, setProfile] = useState(null);
  const [summary, setSummary] = useState(null);
  const [history, setHistory] = useState([]);
  const [playbook, setPlaybook] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showThenVsNow, setShowThenVsNow] = useState(false);

  useEffect(() => {
    loadProfile();
  }, []);

  const loadProfile = async () => {
    try {
      // Load fast endpoints first, don't block on playbook (may call Gemini)
      const [profileRes, summaryRes, historyRes] = await Promise.all([
        profileApi.get().catch(() => ({ data: null })),
        profileApi.getSummary().catch(() => ({ data: null })),
        profileApi.getHistory().catch(() => ({ data: [] })),
      ]);
      setProfile(profileRes.data);
      setSummary(summaryRes.data);
      setHistory(historyRes.data);
    } catch (err) {
      console.error('Failed to load profile:', err);
    } finally {
      setLoading(false);
    }

    // Load playbook in background (may take a while if Gemini generates it)
    try {
      const playbookRes = await profileApi.getPlaybook();
      setPlaybook(playbookRes.data);
    } catch (err) {
      console.error('Failed to load playbook:', err);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-brand-navy"></div>
      </div>
    );
  }

  // Generate radar chart data from bias patterns
  const radarData = profile?.bias_patterns?.map(p => ({
    subject: (p.name || 'unknown').replace(/_/g, ' ').split(' ').map(w => w[0].toUpperCase() + w.slice(1)).join(' '),
    value: Math.round((1 - p.score) * 100), // Invert so higher = better
    fullMark: 100
  })) || [];

  // Generate improvement chart data
  const improvementData = profile?.improvement_trajectory?.map(p => ({
    date: new Date(p.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    score: p.overall_score
  })) || [];

  const hasProfile = profile && profile.total_simulations_analyzed > 0;

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="card">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-16 h-16 rounded-full bg-brand-navy flex items-center justify-center">
              <User className="h-8 w-8 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-brand-navy">Your Behavior Profile</h1>
              <p className="text-brand-navy/60">
                {hasProfile
                  ? `Based on ${profile.total_simulations_analyzed} simulations`
                  : 'Complete simulations to build your profile'}
              </p>
            </div>
          </div>
          {summary && (
            <div className="text-right">
              <p className="text-brand-navy/60 text-sm">Overall Score</p>
              <p className={clsx(
                'text-4xl font-bold',
                summary.overall_score >= 70 ? 'text-green-700' :
                  summary.overall_score >= 50 ? 'text-amber-600' : 'text-red-600'
              )}>
                {Math.round(summary.overall_score)}
              </p>
            </div>
          )}
        </div>
      </div>

      {!hasProfile ? (
        <div className="card text-center py-12">
          <Brain className="h-16 w-16 text-brand-blue mx-auto mb-4" />
          <h2 className="text-xl font-bold text-brand-navy mb-2">No Profile Data Yet</h2>
          <p className="text-brand-navy/60 mb-6 max-w-md mx-auto">
            Your behavior profile is built from your simulation decisions. Complete a few simulations to start seeing your patterns and insights.
          </p>
          <Link to="/dashboard" className="btn btn-primary">
            Start Your First Simulation
          </Link>
        </div>
      ) : (
        <>
          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <StatCard
              icon={Target}
              label="Overall Score"
              value={Math.round(summary?.overall_score || 0)}
              trend={summary?.recent_trend === 'improving' ? 'up' : summary?.recent_trend === 'declining' ? 'down' : undefined}
              trendValue={summary?.recent_trend}
            />
            <StatCard
              icon={Activity}
              label="Simulations"
              value={profile.total_simulations_analyzed}
            />
            <StatCard
              icon={Calendar}
              label="This Week"
              value={summary?.simulations_this_week || 0}
            />
            <StatCard
              icon={Award}
              label="Top Strength"
              value={summary?.top_strength?.replace(/_/g, ' ') || 'N/A'}
            />
          </div>

          {/* Streak Counter */}
          {(summary?.current_streak || 0) > 0 && (
            <div className="card bg-gradient-to-r from-orange-50 to-amber-50 border-orange-200">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-orange-400 to-red-500 flex items-center justify-center shadow-lg">
                    <Flame className="h-7 w-7 text-white" />
                  </div>
                  <div>
                    <p className="text-sm text-orange-600 font-semibold">Current Streak</p>
                    <p className="text-3xl font-black text-orange-700">
                      {summary.current_streak} day{summary.current_streak !== 1 ? 's' : ''}
                    </p>
                  </div>
                </div>
                <p className="text-sm text-orange-500 max-w-xs text-right">
                  {summary.current_streak >= 7 ? 'Incredible consistency!' :
                   summary.current_streak >= 3 ? 'Keep it going!' :
                   'Build your streak!'}
                </p>
              </div>
            </div>
          )}

          {/* Bias Badges */}
          <BadgeGrid profile={profile} />

          {/* Decision Playbook */}
          <PlaybookSection playbook={playbook} />

          {/* Strengths & Weaknesses + Radar Chart */}
          <div className="grid md:grid-cols-2 gap-6">
            <div className="card">
              <h3 className="text-lg font-semibold text-brand-navy mb-4">Behavioral Radar</h3>
              {radarData.length > 0 ? (
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={radarData}>
                      <PolarGrid stroke="#D1D3EB" />
                      <PolarAngleAxis dataKey="subject" tick={{ fill: '#3A3E61', fontSize: 12 }} />
                      <Radar
                        name="Score"
                        dataKey="value"
                        stroke="#3A3E61"
                        fill="#A4B9D8"
                        fillOpacity={0.4}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <p className="text-brand-navy/60 text-center py-8">Not enough data for radar chart</p>
              )}
              <p className="text-brand-navy/60 text-sm text-center mt-2">
                Higher values = better control over that bias
              </p>
            </div>

            <div className="card">
              <h3 className="text-lg font-semibold text-brand-navy mb-4">Strengths & Weaknesses</h3>
              <div className="space-y-4">
                <div>
                  <p className="text-sm text-brand-navy/60 mb-2">Strengths</p>
                  <div className="flex flex-wrap gap-2">
                    {profile.strengths?.length > 0 ? profile.strengths.map((s, i) => (
                      <span key={i} className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm flex items-center space-x-1">
                        <CheckCircle className="h-4 w-4" />
                        <span className="capitalize">{s.replace(/_/g, ' ')}</span>
                      </span>
                    )) : (
                      <span className="text-brand-navy/60">Keep practicing to identify strengths</span>
                    )}
                  </div>
                </div>
                <div>
                  <p className="text-sm text-brand-navy/60 mb-2">Areas to Improve</p>
                  <div className="flex flex-wrap gap-2">
                    {profile.weaknesses?.length > 0 ? profile.weaknesses.map((w, i) => (
                      <span key={i} className="px-3 py-1 bg-red-100 text-red-600 rounded-full text-sm flex items-center space-x-1">
                        <AlertTriangle className="h-4 w-4" />
                        <span className="capitalize">{w.replace(/_/g, ' ')}</span>
                      </span>
                    )) : (
                      <span className="text-brand-navy/60">No major weaknesses detected yet</span>
                    )}
                  </div>
                </div>
                <div className="pt-4 border-t border-brand-blue/20">
                  <p className="text-sm text-brand-navy/60 mb-1">Decision Style</p>
                  <p className="text-brand-navy capitalize">{profile.decision_style || 'Unknown'}</p>
                </div>
                <div>
                  <p className="text-sm text-brand-navy/60 mb-1">Stress Response</p>
                  <p className="text-brand-navy capitalize">{profile.stress_response || 'Unknown'}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Then vs Now Comparison */}
          {improvementData.length > 1 && (
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-brand-navy">Improvement Over Time</h3>
                <button
                  onClick={() => setShowThenVsNow(!showThenVsNow)}
                  className={clsx(
                    'flex items-center gap-2 text-sm font-semibold px-3 py-1.5 rounded-lg transition-colors',
                    showThenVsNow ? 'bg-brand-navy text-white' : 'bg-brand-lavender text-brand-navy hover:bg-brand-navy/10'
                  )}
                >
                  {showThenVsNow ? <ToggleRight className="h-4 w-4" /> : <ToggleLeft className="h-4 w-4" />}
                  Then vs Now
                </button>
              </div>

              {showThenVsNow && improvementData.length >= 2 ? (
                <div className="grid grid-cols-2 gap-4 mb-6">
                  <div className="p-4 bg-gray-50 rounded-xl border border-gray-200 text-center">
                    <p className="text-xs text-gray-400 uppercase tracking-wider font-bold mb-2">Then (First Session)</p>
                    <p className={clsx(
                      'text-4xl font-black',
                      improvementData[0].score >= 50 ? 'text-amber-600' : 'text-red-600'
                    )}>
                      {Math.round(improvementData[0].score)}
                    </p>
                    <p className="text-xs text-gray-400 mt-1">{improvementData[0].date}</p>
                  </div>
                  <div className="p-4 bg-brand-lavender/30 rounded-xl border border-brand-navy/20 text-center">
                    <p className="text-xs text-brand-navy/60 uppercase tracking-wider font-bold mb-2">Now (Latest)</p>
                    <p className={clsx(
                      'text-4xl font-black',
                      improvementData[improvementData.length - 1].score >= 70 ? 'text-green-700' :
                      improvementData[improvementData.length - 1].score >= 50 ? 'text-amber-600' : 'text-red-600'
                    )}>
                      {Math.round(improvementData[improvementData.length - 1].score)}
                    </p>
                    <p className="text-xs text-brand-navy/60 mt-1">{improvementData[improvementData.length - 1].date}</p>
                  </div>
                  {(() => {
                    const diff = improvementData[improvementData.length - 1].score - improvementData[0].score;
                    return (
                      <div className="col-span-2 text-center p-3 bg-white rounded-xl border border-gray-200">
                        <span className={clsx(
                          'text-lg font-bold',
                          diff > 0 ? 'text-green-700' : diff < 0 ? 'text-red-600' : 'text-gray-600'
                        )}>
                          {diff > 0 ? '+' : ''}{Math.round(diff)} points
                        </span>
                        <span className="text-sm text-gray-400 ml-2">
                          {diff > 10 ? 'Great improvement!' : diff > 0 ? 'Steady progress' : diff === 0 ? 'Same score' : 'Room to grow'}
                        </span>
                      </div>
                    );
                  })()}
                </div>
              ) : null}

              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={improvementData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#D1D3EB" />
                    <XAxis dataKey="date" stroke="#3A3E61" tick={{ fill: '#3A3E61', fontSize: 11 }} />
                    <YAxis stroke="#3A3E61" tick={{ fill: '#3A3E61', fontSize: 11 }} domain={[0, 100]} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#F1EDE2', border: '1px solid #D1D3EB', borderRadius: '8px' }}
                      labelStyle={{ color: '#3A3E61' }}
                    />
                    <Line
                      type="monotone"
                      dataKey="score"
                      stroke="#3A3E61"
                      strokeWidth={2}
                      dot={{ fill: '#3A3E61' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Bias Patterns */}
          <div className="card">
            <h3 className="text-lg font-semibold text-brand-navy mb-4">Bias Patterns</h3>
            <div className="divide-y divide-brand-blue/20">
              {profile.bias_patterns?.map((pattern, i) => (
                <BiasPatternRow key={i} pattern={pattern} />
              ))}
            </div>
          </div>

          {/* Recent History */}
          <div className="card">
            <h3 className="text-lg font-semibold text-brand-navy mb-4">Recent Simulations</h3>
            {history.length > 0 ? (
              <div className="divide-y divide-brand-blue/20">
                {history.slice(0, 10).map((item, i) => (
                  <HistoryItem key={i} item={item} />
                ))}
              </div>
            ) : (
              <p className="text-brand-navy/60 text-center py-4">No simulation history yet</p>
            )}
          </div>
        </>
      )}
    </div>
  );
}
