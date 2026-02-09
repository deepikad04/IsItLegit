import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { scenariosApi, simulationsApi, profileApi } from '../api/client';
import { useAuth } from '../context/AuthContext';
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

  const formatUnlockRequirements = (reqs) => {
    if (!reqs) return '';
    const parts = [];
    if (reqs.min_simulations) parts.push(`${reqs.min_simulations} simulations completed`);
    if (reqs.min_process_score) parts.push(`${reqs.min_process_score}% avg process score`);
    return parts.join(' + ');
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-brand-navy"></div>
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
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-brand-navy">Available Scenarios</h2>
          <span className="text-sm text-brand-navy/50">{scenarios.length} scenarios</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {scenarios.map((scenario, i) => {
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
