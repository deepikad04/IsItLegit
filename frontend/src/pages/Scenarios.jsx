import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { scenariosApi, profileApi } from '../api/client';
import { useAuth } from '../context/AuthContext';
import {
  Play,
  TrendingUp,
  Target,
  Clock,
  Zap,
  AlertTriangle,
  BarChart2,
  Lock,
  Sparkles,
  X,
} from 'lucide-react';
import clsx from 'clsx';

export default function Scenarios() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [scenarios, setScenarios] = useState([]);
  const [profileSummary, setProfileSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [generatingChallenge, setGeneratingChallenge] = useState(false);
  const [filterCategory, setFilterCategory] = useState('all');
  const [filterDifficulty, setFilterDifficulty] = useState('all');

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [scenariosRes, profileRes] = await Promise.all([
        scenariosApi.listUnlocked().catch(() => scenariosApi.list()),
        profileApi.getSummary()
      ]);
      setScenarios(scenariosRes.data);
      setProfileSummary(profileRes.data);
    } catch (err) {
      console.error('Failed to load scenarios:', err);
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
    const colors = { 1: 'text-green-600', 2: 'text-yellow-600', 3: 'text-orange-500', 4: 'text-red-500', 5: 'text-purple-600' };
    return colors[difficulty] || 'text-brand-blue';
  };

  const getDifficultyDot = (difficulty) => {
    const colors = { 1: 'bg-green-500', 2: 'bg-yellow-500', 3: 'bg-orange-500', 4: 'bg-red-500', 5: 'bg-purple-500' };
    return colors[difficulty] || 'bg-brand-blue';
  };

  const getCategoryIcon = (category) => {
    const icons = { fomo_trap: AlertTriangle, patience_test: Clock, loss_aversion: TrendingUp, social_proof: Zap, risk_management: Target, contrarian: BarChart2 };
    return icons[category] || Target;
  };

  const CATEGORY_LABELS = {
    fomo_trap: 'FOMO Trap', patience_test: 'Patience Test', loss_aversion: 'Loss Aversion',
    social_proof: 'Social Proof', risk_management: 'Risk Management', contrarian: 'Contrarian',
  };

  const formatUnlockRequirements = (reqs) => {
    if (!reqs) return '';
    const parts = [];
    if (reqs.min_simulations) parts.push(`${reqs.min_simulations} simulations completed`);
    if (reqs.min_process_score) parts.push(`${reqs.min_process_score}% avg process score`);
    return parts.join(' + ');
  };

  const categories = [...new Set(scenarios.map((s) => s.category))].sort();

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
          <p className="text-brand-navy/60 text-sm">Loading scenarios...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
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
          <div className="card bg-gradient-to-r from-brand-lavender/40 to-brand-cream border-brand-navy/20 animate-fade-in-up">
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
              <button onClick={() => startSimulation(recommended)} className="btn btn-primary flex items-center space-x-2 whitespace-nowrap">
                <Play className="h-4 w-4" /><span>Start</span>
              </button>
            </div>
          </div>
        );
      })()}

      {/* AI Challenge Card */}
      {profileSummary && (user?.total_simulations || 0) >= 1 && (
        <div className="card bg-gradient-to-r from-brand-navy/10 to-brand-lavender/40 border border-brand-navy/20 ring-1 ring-brand-navy/10 shadow-md animate-fade-in-up">
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
                <><div className="animate-spin rounded-full h-4 w-4 border-t-2 border-b-2 border-white" /><span>Generating...</span></>
              ) : (
                <><Target className="h-4 w-4" /><span>Accept Challenge</span></>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Scenarios Grid */}
      <div>
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
          <h2 className="text-xl font-semibold text-brand-navy">Available Scenarios</h2>
          <div className="flex items-center gap-2 flex-wrap">
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
                style={{ animationDelay: `${(i + 1) * 0.05}s` }}
                onClick={() => !isLocked && startSimulation(scenario)}
              >
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
                        <Sparkles className="h-3 w-3" /> AI
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
                    <Play className="h-4 w-4" /><span>Start</span>
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
