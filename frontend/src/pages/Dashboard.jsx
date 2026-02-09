import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { scenariosApi, simulationsApi, profileApi } from '../api/client';
import { useAuth } from '../context/AuthContext';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid
} from 'recharts';
import {
  Play,
  TrendingUp,
  Target,
  Clock,
  ChevronRight,
  Zap,
  Award,
  AlertTriangle,
  BarChart2,
  Lock,
  BookOpen,
  ArrowUpRight,
  Sparkles,
  Filter,
  X,
  Brain,
  Flame,
  Compass,
  Users,
  Hash,
} from 'lucide-react';
import clsx from 'clsx';

export default function Dashboard() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [scenarios, setScenarios] = useState([]);
  const [recentSimulations, setRecentSimulations] = useState([]);
  const [profileSummary, setProfileSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [generatingChallenge, setGeneratingChallenge] = useState(false);
  const [filterCategory, setFilterCategory] = useState('all');
  const [filterDifficulty, setFilterDifficulty] = useState('all');
  const [progressData, setProgressData] = useState([]);
  const [communityStats, setCommunityStats] = useState(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [scenariosRes, simulationsRes, profileRes] = await Promise.all([
        scenariosApi.listUnlocked().catch(() => scenariosApi.list()),
        simulationsApi.list({ limit: 5, status: 'completed' }),
        profileApi.getSummary()
      ]);
      setScenarios(scenariosRes.data);
      setRecentSimulations(simulationsRes.data);
      setProfileSummary(profileRes.data);

      // Load community stats (non-blocking)
      profileApi.getCommunityStats().then(res => {
        setCommunityStats(res.data);
      }).catch(() => {});

      // Load full profile for progress chart (non-blocking)
      profileApi.get().then(res => {
        if (res.data?.improvement_trajectory) {
          setProgressData(res.data.improvement_trajectory.map(p => ({
            date: new Date(p.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
            score: p.overall_score || p.score || 0,
          })));
        }
      }).catch(() => {});
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  const startSimulation = (scenario) => {
    if (scenario.is_locked) return;
    navigate(`/simulation/${scenario.id}`);
  };

  const generateAIChallenge = async () => {
    setGeneratingChallenge(true);
    try {
      const res = await scenariosApi.generateAdaptive();
      navigate(`/simulation/${res.data.id}`);
    } catch (err) {
      console.error('Failed to generate adaptive scenario:', err);
      setGeneratingChallenge(false);
    }
  };

  const getDifficultyColor = (difficulty) => {
    const colors = {
      1: 'text-green-600',
      2: 'text-yellow-600',
      3: 'text-orange-500',
      4: 'text-red-500',
      5: 'text-purple-600'
    };
    return colors[difficulty] || 'text-brand-blue';
  };

  const getDifficultyDot = (difficulty) => {
    const colors = {
      1: 'bg-green-500',
      2: 'bg-yellow-500',
      3: 'bg-orange-500',
      4: 'bg-red-500',
      5: 'bg-purple-500'
    };
    return colors[difficulty] || 'bg-brand-blue';
  };

  const getCategoryIcon = (category) => {
    const icons = {
      fomo_trap: AlertTriangle,
      patience_test: Clock,
      loss_aversion: TrendingUp,
      social_proof: Zap,
      risk_management: Target,
      contrarian: BarChart2
    };
    return icons[category] || Target;
  };

  const CATEGORY_LABELS = {
    fomo_trap: 'FOMO Trap',
    patience_test: 'Patience Test',
    loss_aversion: 'Loss Aversion',
    social_proof: 'Social Proof',
    risk_management: 'Risk Management',
    contrarian: 'Contrarian',
  };

  const formatUnlockRequirements = (reqs) => {
    if (!reqs) return '';
    const parts = [];
    if (reqs.min_simulations) parts.push(`${reqs.min_simulations} simulations completed`);
    if (reqs.min_process_score) parts.push(`${reqs.min_process_score}% avg process score`);
    return parts.join(' + ');
  };

  // Derive unique categories from loaded scenarios
  const categories = [...new Set(scenarios.map((s) => s.category))].sort();

  // Filter and sort: unlocked first, then by difficulty ascending
  const filteredScenarios = scenarios
    .filter((s) => filterCategory === 'all' || s.category === filterCategory)
    .filter((s) => filterDifficulty === 'all' || s.difficulty === Number(filterDifficulty))
    .sort((a, b) => {
      if (a.is_locked !== b.is_locked) return a.is_locked ? 1 : -1;
      return a.difficulty - b.difficulty;
    });

  const hasActiveFilters = filterCategory !== 'all' || filterDifficulty !== 'all';
  const clearFilters = () => { setFilterCategory('all'); setFilterDifficulty('all'); };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-brand-navy mx-auto mb-3"></div>
          <p className="text-brand-navy/60 text-sm">Loading your dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Hero Section — Welcome + Stats */}
      <div className="card bg-gradient-to-br from-brand-cream to-brand-lavender/40 border-brand-blue/20 animate-fade-in-up">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
          <div>
            <p className="text-brand-navy/50 text-sm font-medium uppercase tracking-wider mb-1">Welcome back</p>
            <h1 className="text-3xl font-bold text-brand-navy">{user?.username}</h1>
            <p className="text-brand-navy/60 mt-2">Ready to sharpen your decision-making skills?</p>
          </div>
          <div className="flex items-center gap-6 flex-wrap">
            {/* Simulations stat */}
            <div className="text-center">
              <div className="w-14 h-14 rounded-2xl bg-brand-navy/10 flex items-center justify-center mx-auto mb-1.5">
                <Award className="h-7 w-7 text-brand-navy" />
              </div>
              <p className="text-2xl font-bold text-brand-navy">{user?.total_simulations || 0}</p>
              <p className="text-xs text-brand-navy/50 uppercase tracking-wider">Simulations</p>
            </div>
            {/* Score stat */}
            {profileSummary && profileSummary.overall_score > 0 && (
              <div className="text-center">
                <div className="w-14 h-14 rounded-2xl bg-brand-navy/10 flex items-center justify-center mx-auto mb-1.5">
                  <Target className="h-7 w-7 text-brand-navy" />
                </div>
                <p className="text-2xl font-bold text-brand-navy">{Math.round(profileSummary.overall_score)}</p>
                <p className="text-xs text-brand-navy/50 uppercase tracking-wider">Score</p>
              </div>
            )}
            {/* Strength */}
            {profileSummary?.top_strength && (
              <div className="text-center">
                <div className="w-14 h-14 rounded-2xl bg-green-100 flex items-center justify-center mx-auto mb-1.5">
                  <TrendingUp className="h-7 w-7 text-green-700" />
                </div>
                <p className="text-sm font-semibold text-brand-navy capitalize">{profileSummary.top_strength.replace(/_/g, ' ')}</p>
                <p className="text-xs text-brand-navy/50 uppercase tracking-wider">Strength</p>
              </div>
            )}
            {/* Streak */}
            {profileSummary?.current_streak > 0 && (
              <div className="text-center">
                <div className="w-14 h-14 rounded-2xl bg-orange-100 flex items-center justify-center mx-auto mb-1.5">
                  <Flame className="h-7 w-7 text-orange-500" />
                </div>
                <p className="text-2xl font-bold text-orange-600">{profileSummary.current_streak}</p>
                <p className="text-xs text-brand-navy/50 uppercase tracking-wider">Streak</p>
              </div>
            )}
            {/* Focus area */}
            {profileSummary?.top_weakness && (
              <div className="text-center">
                <div className="w-14 h-14 rounded-2xl bg-amber-100 flex items-center justify-center mx-auto mb-1.5">
                  <AlertTriangle className="h-7 w-7 text-amber-600" />
                </div>
                <p className="text-sm font-semibold text-brand-navy capitalize">{profileSummary.top_weakness.replace(/_/g, ' ')}</p>
                <p className="text-xs text-brand-navy/50 uppercase tracking-wider">Focus</p>
              </div>
            )}
          </div>
        </div>
        {/* Profile link */}
        {profileSummary && profileSummary.overall_score > 0 && (
          <div className="mt-4 pt-4 border-t border-brand-blue/20 flex justify-end">
            <Link to="/profile" className="text-sm text-brand-navy hover:text-brand-navy-light flex items-center gap-1 transition-colors">
              View Full Profile <ArrowUpRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        )}
      </div>

      {/* New User Empty State */}
      {(user?.total_simulations || 0) === 0 && (
        <div className="card bg-gradient-to-br from-brand-cream to-brand-lavender/30 border-brand-blue/20 text-center py-8 animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
          <Compass className="h-12 w-12 text-brand-navy mx-auto mb-3" />
          <h2 className="text-xl font-bold text-brand-navy mb-2">Welcome to IsItLegit!</h2>
          <p className="text-brand-navy/60 max-w-md mx-auto mb-6">
            Start your first simulation to begin building your decision profile. Pick any scenario below — no experience needed.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-4">
            <Link to="/learning" className="btn btn-secondary flex items-center gap-2">
              <BookOpen className="h-4 w-4" /> Learn the Basics
            </Link>
          </div>
        </div>
      )}

      {/* Progress Chart + Killer Stat (for returning users) */}
      {progressData.length > 1 && (() => {
        const firstScore = progressData[0]?.score || 0;
        const latestScore = progressData[progressData.length - 1]?.score || 0;
        const improvement = latestScore - firstScore;
        const simCount = progressData.length;
        return (
          <div className="card animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-brand-navy flex items-center gap-2">
                <TrendingUp className="h-5 w-5" /> Your Progress
              </h2>
              <Link to="/profile" className="text-sm text-brand-navy/60 hover:text-brand-navy flex items-center gap-1 transition-colors">
                Full Profile <ArrowUpRight className="h-3.5 w-3.5" />
              </Link>
            </div>
            {/* Killer stat banner */}
            {improvement > 0 && simCount >= 2 && (
              <div className="mb-4 p-4 bg-gradient-to-r from-emerald-50 to-cyan-50 border border-emerald-200 rounded-xl flex items-center gap-4">
                <div className="w-14 h-14 rounded-2xl bg-emerald-100 flex items-center justify-center flex-shrink-0">
                  <TrendingUp className="h-7 w-7 text-emerald-600" />
                </div>
                <div>
                  <p className="text-2xl font-black text-emerald-700">
                    +{Math.round(improvement)} points
                  </p>
                  <p className="text-sm text-emerald-600/80">
                    Process quality improved across {simCount} simulations ({firstScore} → {latestScore})
                  </p>
                </div>
              </div>
            )}
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={progressData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="date" stroke="#9ca3af" tick={{ fontSize: 11, fill: '#6b7280' }} />
                  <YAxis stroke="#9ca3af" tick={{ fontSize: 11, fill: '#6b7280' }} domain={[0, 100]} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#F1EDE2', border: '1px solid #D1D3EB', borderRadius: '8px', fontSize: '13px' }}
                    labelStyle={{ color: '#3A3E61' }}
                  />
                  <Line type="monotone" dataKey="score" stroke="#3A3E61" strokeWidth={2} dot={{ fill: '#3A3E61', r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        );
      })()}

      {/* Community Stats */}
      {communityStats && communityStats.total_simulations > 0 && (
        <div className="card animate-fade-in-up" style={{ animationDelay: '0.15s' }}>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-brand-navy flex items-center gap-2">
              <Users className="h-5 w-5" /> Community Insights
            </h2>
            {communityStats.your_percentile != null && (
              <span className="text-sm font-bold text-brand-navy bg-brand-lavender px-3 py-1 rounded-full">
                Top {100 - communityStats.your_percentile}%
              </span>
            )}
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-50 rounded-xl p-4 text-center">
              <p className="text-2xl font-black text-brand-navy">{communityStats.total_traders}</p>
              <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mt-1">Traders</p>
            </div>
            <div className="bg-gray-50 rounded-xl p-4 text-center">
              <p className="text-2xl font-black text-brand-navy">{communityStats.total_simulations}</p>
              <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mt-1">Simulations</p>
            </div>
            <div className="bg-gray-50 rounded-xl p-4 text-center">
              <p className="text-2xl font-black text-brand-navy">{communityStats.avg_process_score}%</p>
              <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mt-1">Avg Score</p>
            </div>
            <div className="bg-gray-50 rounded-xl p-4 text-center">
              <p className="text-2xl font-black text-brand-navy">{communityStats.total_decisions}</p>
              <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mt-1">Decisions</p>
            </div>
          </div>
          {(communityStats.most_common_bias || communityStats.most_popular_scenario) && (
            <div className="flex flex-wrap gap-3 mt-4">
              {communityStats.most_common_bias && (
                <div className="flex items-center gap-2 bg-amber-50 border border-amber-200 rounded-xl px-4 py-2.5">
                  <AlertTriangle className="h-4 w-4 text-amber-600" />
                  <span className="text-sm font-bold text-amber-800">
                    Most common bias: {communityStats.most_common_bias} ({communityStats.most_common_bias_pct}%)
                  </span>
                </div>
              )}
              {communityStats.most_popular_scenario && (
                <div className="flex items-center gap-2 bg-blue-50 border border-blue-200 rounded-xl px-4 py-2.5">
                  <Target className="h-4 w-4 text-blue-600" />
                  <span className="text-sm font-bold text-blue-800">
                    Most played: {communityStats.most_popular_scenario}
                  </span>
                </div>
              )}
            </div>
          )}
          {communityStats.score_distribution && (
            <div className="mt-4 pt-4 border-t border-gray-200">
              <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3">Score Distribution</p>
              <div className="flex items-end gap-2 h-16">
                {Object.entries(communityStats.score_distribution).map(([level, count]) => {
                  const total = Object.values(communityStats.score_distribution).reduce((a, b) => a + b, 0);
                  const pct = total > 0 ? (count / total) * 100 : 0;
                  const colors = { beginner: 'bg-red-400', developing: 'bg-amber-400', proficient: 'bg-blue-400', expert: 'bg-emerald-400' };
                  return (
                    <div key={level} className="flex-1 flex flex-col items-center gap-1">
                      <div className={clsx('w-full rounded-t-lg transition-all', colors[level])}
                        style={{ height: `${Math.max(pct, 4)}%` }} />
                      <span className="text-xs font-bold text-gray-500 capitalize">{level}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Recommended Scenario */}
      {profileSummary?.top_weakness && filteredScenarios.length > 0 && (() => {
        const weaknessToCategory = {
          fomo: 'fomo_trap', loss_aversion: 'loss_aversion', impulsivity: 'fomo_trap',
          social_proof_reliance: 'social_proof', overconfidence: 'risk_management',
          anchoring: 'patience_test', herd_following: 'social_proof',
        };
        const targetCategory = weaknessToCategory[profileSummary.top_weakness];
        const recommended = filteredScenarios.find(s => s.category === targetCategory && !s.is_locked);
        if (!recommended) return null;
        const RecIcon = getCategoryIcon(recommended.category);
        return (
          <div className="card bg-gradient-to-r from-brand-lavender/40 to-brand-cream border-brand-navy/20 animate-fade-in-up" style={{ animationDelay: '0.15s' }}>
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div className="flex items-start space-x-4">
                <div className="w-12 h-12 rounded-xl bg-brand-navy/10 flex items-center justify-center flex-shrink-0">
                  <RecIcon className="h-6 w-6 text-brand-navy" />
                </div>
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <h2 className="text-lg font-semibold text-brand-navy">Recommended for You</h2>
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-brand-lavender text-brand-navy text-xs font-bold">
                      <Target className="h-3 w-3" /> Targets {profileSummary.top_weakness.replace(/_/g, ' ')}
                    </span>
                  </div>
                  <p className="text-brand-navy/70 text-sm">
                    <span className="font-semibold">{recommended.name}</span> — {recommended.description}
                  </p>
                </div>
              </div>
              <button
                onClick={() => startSimulation(recommended)}
                className="btn btn-primary flex items-center space-x-2 whitespace-nowrap"
              >
                <Play className="h-4 w-4" />
                <span>Start</span>
              </button>
            </div>
          </div>
        );
      })()}

      {/* AI Challenge Card */}
      {profileSummary && (user?.total_simulations || 0) >= 1 && (
        <div className="card bg-gradient-to-r from-brand-navy/10 to-brand-lavender/40 border border-brand-navy/20 ring-1 ring-brand-navy/10 shadow-md animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
          <div className="flex flex-col md:flex-row md:items-center md:justify-between">
            <div className="flex items-start space-x-4">
              <div className="w-12 h-12 rounded-xl bg-brand-lavender flex items-center justify-center flex-shrink-0">
                <Zap className="h-6 w-6 text-brand-navy animate-pulse" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-brand-navy mb-1">AI Challenge</h2>
                <p className="text-brand-navy/70 text-sm">
                  Gemini has analyzed your behavior profile and designed a scenario targeting your
                  {profileSummary.top_weakness
                    ? ` ${profileSummary.top_weakness.replace(/_/g, ' ')} tendencies`
                    : ' weakest patterns'}
                  . Think you can beat it?
                </p>
              </div>
            </div>
            <button
              onClick={generateAIChallenge}
              disabled={generatingChallenge}
              className="mt-4 md:mt-0 btn btn-primary flex items-center space-x-2 whitespace-nowrap"
            >
              {generatingChallenge ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-t-2 border-b-2 border-white" />
                  <span>Generating...</span>
                </>
              ) : (
                <>
                  <Target className="h-4 w-4" />
                  <span>Accept Challenge</span>
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Scenarios Section */}
      <div>
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
          <h2 className="text-xl font-semibold text-brand-navy">Available Scenarios</h2>
          <div className="flex items-center gap-2 flex-wrap">
            {/* Category filter */}
            <select
              value={filterCategory}
              onChange={(e) => setFilterCategory(e.target.value)}
              className="text-sm border border-brand-blue/20 rounded-lg px-3 py-1.5 bg-white text-brand-navy focus:outline-none focus:ring-2 focus:ring-brand-navy/20"
            >
              <option value="all">All Categories</option>
              {categories.map((cat) => (
                <option key={cat} value={cat}>{CATEGORY_LABELS[cat] || cat}</option>
              ))}
            </select>
            {/* Difficulty filter */}
            <select
              value={filterDifficulty}
              onChange={(e) => setFilterDifficulty(e.target.value)}
              className="text-sm border border-brand-blue/20 rounded-lg px-3 py-1.5 bg-white text-brand-navy focus:outline-none focus:ring-2 focus:ring-brand-navy/20"
            >
              <option value="all">All Levels</option>
              {[1, 2, 3, 4, 5].map((d) => (
                <option key={d} value={d}>Level {d}</option>
              ))}
            </select>
            {hasActiveFilters && (
              <button onClick={clearFilters} className="text-sm text-brand-navy/50 hover:text-red-500 flex items-center gap-1 transition-colors">
                <X className="h-3.5 w-3.5" /> Clear
              </button>
            )}
            <span className="text-sm text-brand-navy/50 ml-1">{filteredScenarios.length} of {scenarios.length}</span>
          </div>
        </div>
        {filteredScenarios.length === 0 ? (
          <div className="card text-center py-10">
            <Target className="h-10 w-10 text-brand-blue mx-auto mb-3" />
            <p className="text-brand-navy font-semibold mb-1">No scenarios match your filters</p>
            <p className="text-brand-navy/60 text-sm mb-4">Try adjusting the category or difficulty level.</p>
            <button onClick={clearFilters} className="btn btn-secondary text-sm">Clear Filters</button>
          </div>
        ) : null}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredScenarios.map((scenario, i) => {
            const CategoryIcon = getCategoryIcon(scenario.category);
            const isLocked = scenario.is_locked;
            return (
              <div
                key={scenario.id}
                className={clsx(
                  'card relative animate-fade-in-up transition-all duration-300',
                  isLocked ? 'opacity-75' : 'hover:shadow-md hover:-translate-y-1 cursor-pointer'
                )}
                style={{ animationDelay: `${(i + 2) * 0.1}s` }}
                onClick={() => !isLocked && startSimulation(scenario)}
              >
                {/* Lock Overlay */}
                {isLocked && (
                  <div className="absolute inset-0 bg-white/70 rounded-2xl flex flex-col items-center justify-center z-10 backdrop-blur-[1px]">
                    <Lock className="h-8 w-8 text-brand-navy/60 mb-2" />
                    <p className="text-brand-navy/70 text-sm font-medium mb-1">Locked</p>
                    <p className="text-brand-blue text-xs text-center px-4">
                      {formatUnlockRequirements(scenario.unlock_requirements)}
                    </p>
                  </div>
                )}

                <div className="flex items-start justify-between mb-3">
                  <div className="w-10 h-10 rounded-xl bg-brand-navy/10 flex items-center justify-center">
                    <CategoryIcon className="h-5 w-5 text-brand-navy" />
                  </div>
                  <div className="flex items-center gap-2">
                    {scenario.is_ai_generated && (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-brand-lavender text-brand-navy text-xs font-medium">
                        <Sparkles className="h-3 w-3" />
                        AI
                      </span>
                    )}
                    <span className={`text-sm font-medium flex items-center gap-1.5 ${getDifficultyColor(scenario.difficulty)}`}>
                      <span className={`w-2 h-2 rounded-full ${getDifficultyDot(scenario.difficulty)}`} />
                      Level {scenario.difficulty}
                    </span>
                  </div>
                </div>
                <h3 className="text-lg font-semibold text-brand-navy mb-2">{scenario.name}</h3>
                <p className="text-brand-navy/60 text-sm mb-4 line-clamp-2">{scenario.description}</p>
                <div className="flex items-center justify-between">
                  <div className="flex items-center text-sm text-brand-blue">
                    <Clock className="h-4 w-4 mr-1" />
                    {scenario.time_pressure_seconds}s
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); startSimulation(scenario); }}
                    disabled={isLocked}
                    className={clsx(
                      'btn flex items-center space-x-2',
                      isLocked ? 'btn-secondary opacity-50 cursor-not-allowed' : 'btn-primary'
                    )}
                  >
                    <Play className="h-4 w-4" />
                    <span>Start</span>
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Recent Simulations */}
      {recentSimulations.length > 0 && (
        <div className="animate-fade-in-up" style={{ animationDelay: '0.4s' }}>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-brand-navy">Recent Simulations</h2>
            <Link to="/profile" className="text-sm text-brand-navy hover:text-brand-navy-light flex items-center transition-colors">
              View all <ChevronRight className="h-4 w-4 ml-1" />
            </Link>
          </div>
          <div className="card p-0 overflow-hidden">
            <div className="divide-y divide-brand-blue/10">
              {recentSimulations.map((sim) => {
                const isProfit = (sim.final_outcome?.profit_loss || 0) >= 0;
                return (
                  <Link
                    key={sim.id}
                    to={`/reflection/${sim.id}`}
                    className="flex items-center justify-between py-4 px-6 hover:bg-brand-lavender/20 transition-colors group"
                  >
                    <div className="flex items-center gap-3">
                      <div className={clsx(
                        'w-1 h-10 rounded-full flex-shrink-0',
                        isProfit ? 'bg-green-500' : 'bg-red-500'
                      )} />
                      <div>
                        <p className="text-brand-navy font-medium group-hover:text-brand-navy-light transition-colors">
                          {sim.scenario?.name || 'Simulation'}
                        </p>
                        <p className="text-sm text-brand-navy/50">
                          {new Date(sim.completed_at).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-4">
                      <div className="text-right">
                        <p className={clsx('font-semibold', isProfit ? 'text-green-600' : 'text-red-500')}>
                          {isProfit ? '+' : ''}${Math.abs(sim.final_outcome?.profit_loss || 0).toFixed(2)}
                        </p>
                        <p className="text-sm text-brand-navy/50">
                          Process: {Math.round(sim.process_quality_score || 0)}%
                        </p>
                      </div>
                      <ChevronRight className="h-5 w-5 text-brand-blue/50 group-hover:text-brand-navy transition-colors" />
                    </div>
                  </Link>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Learning CTA */}
      <div className="card bg-gradient-to-r from-brand-lavender/50 to-brand-cream border-brand-navy/20 animate-fade-in-up" style={{ animationDelay: '0.5s' }}>
        <div className="flex flex-col md:flex-row md:items-center md:justify-between">
          <div className="flex items-start space-x-4">
            <div className="w-12 h-12 rounded-xl bg-brand-navy/10 flex items-center justify-center flex-shrink-0">
              <BookOpen className="h-6 w-6 text-brand-navy" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-brand-navy mb-1">Personalized Learning</h2>
              <p className="text-brand-navy/70 text-sm">Get bite-sized lessons tailored to your decision patterns.</p>
            </div>
          </div>
          <Link to="/learning" className="mt-4 md:mt-0 btn btn-primary flex items-center space-x-2">
            <BookOpen className="h-4 w-4" />
            <span>Start Learning</span>
          </Link>
        </div>
      </div>
    </div>
  );
}
