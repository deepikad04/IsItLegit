import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { simulationsApi, profileApi } from '../api/client';
import { useAuth } from '../context/AuthContext';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid
} from 'recharts';
import {
  Play,
  TrendingUp,
  Target,
  ChevronRight,
  Award,
  AlertTriangle,
  BookOpen,
  ArrowUpRight,
  Flame,
  Compass,
  User,
  FileText,
  Activity,
} from 'lucide-react';
import clsx from 'clsx';

export default function Dashboard() {
  const { user } = useAuth();
  const [recentSimulations, setRecentSimulations] = useState([]);
  const [profileSummary, setProfileSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [progressData, setProgressData] = useState([]);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [simulationsRes, profileRes] = await Promise.all([
        simulationsApi.list({ limit: 5, status: 'completed' }),
        profileApi.getSummary()
      ]);
      setRecentSimulations(simulationsRes.data);
      setProfileSummary(profileRes.data);

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

      {/* Quick Links */}
      {(user?.total_simulations || 0) >= 1 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 animate-fade-in-up" style={{ animationDelay: '0.05s' }}>
          <Link to="/scenarios" className="card py-3 px-4 flex items-center gap-3 hover:shadow-md hover:-translate-y-0.5 transition-all group">
            <Play className="h-5 w-5 text-brand-navy group-hover:text-brand-navy-light" />
            <span className="text-sm font-semibold text-brand-navy">Scenarios</span>
          </Link>
          <Link to="/profile" className="card py-3 px-4 flex items-center gap-3 hover:shadow-md hover:-translate-y-0.5 transition-all group">
            <User className="h-5 w-5 text-brand-navy group-hover:text-brand-navy-light" />
            <span className="text-sm font-semibold text-brand-navy">Profile</span>
          </Link>
          <Link to="/learning" className="card py-3 px-4 flex items-center gap-3 hover:shadow-md hover:-translate-y-0.5 transition-all group">
            <BookOpen className="h-5 w-5 text-brand-navy group-hover:text-brand-navy-light" />
            <span className="text-sm font-semibold text-brand-navy">Learning</span>
          </Link>
          {recentSimulations.length > 0 && (
            <Link to={`/reflection/${recentSimulations[0].id}`} className="card py-3 px-4 flex items-center gap-3 hover:shadow-md hover:-translate-y-0.5 transition-all group">
              <FileText className="h-5 w-5 text-brand-navy group-hover:text-brand-navy-light" />
              <span className="text-sm font-semibold text-brand-navy">Last Reflection</span>
            </Link>
          )}
        </div>
      )}

      {/* New User Empty State */}
      {(user?.total_simulations || 0) === 0 && (
        <div className="card bg-gradient-to-br from-brand-cream to-brand-lavender/30 border-brand-blue/20 text-center py-8 animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
          <Compass className="h-12 w-12 text-brand-navy mx-auto mb-3" />
          <h2 className="text-xl font-bold text-brand-navy mb-2">Welcome to IsItLegit!</h2>
          <p className="text-brand-navy/60 max-w-md mx-auto mb-6">
            Start your first simulation to begin building your decision profile. Pick a scenario to get started — no experience needed.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-4">
            <Link to="/scenarios" className="btn btn-primary flex items-center gap-2">
              <Play className="h-4 w-4" /> Browse Scenarios
            </Link>
            <Link to="/learning" className="btn btn-secondary flex items-center gap-2">
              <BookOpen className="h-4 w-4" /> Learn the Basics
            </Link>
          </div>
        </div>
      )}

      {/* Progress Chart + Killer Stat (for returning users — min 2 simulations) */}
      {(user?.total_simulations || 0) >= 2 && progressData.length > 1 && (() => {
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

      {/* Impact Metrics — show process quality trend across recent simulations */}
      {recentSimulations.length >= 2 && (() => {
        const scores = recentSimulations
          .filter(s => s.process_quality_score != null)
          .slice()
          .reverse(); // oldest first
        if (scores.length < 2) return null;
        const first = Math.round(scores[0].process_quality_score);
        const last = Math.round(scores[scores.length - 1].process_quality_score);
        const delta = last - first;
        return (
          <div className="card animate-fade-in-up" style={{ animationDelay: '0.15s' }}>
            <h2 className="text-lg font-semibold text-brand-navy flex items-center gap-2 mb-4">
              <Activity className="h-5 w-5" /> Impact Metrics
            </h2>
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-gray-50 rounded-xl p-4 text-center">
                <p className="text-2xl font-black text-brand-navy">{first}%</p>
                <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mt-1">First Sim</p>
              </div>
              <div className="bg-gray-50 rounded-xl p-4 text-center">
                <p className="text-2xl font-black text-brand-navy">{last}%</p>
                <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mt-1">Latest Sim</p>
              </div>
              <div className={clsx(
                'rounded-xl p-4 text-center',
                delta > 0 ? 'bg-emerald-50' : delta < 0 ? 'bg-red-50' : 'bg-gray-50'
              )}>
                <p className={clsx(
                  'text-2xl font-black',
                  delta > 0 ? 'text-emerald-600' : delta < 0 ? 'text-red-500' : 'text-brand-navy'
                )}>
                  {delta > 0 ? '+' : ''}{delta}%
                </p>
                <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mt-1">Change</p>
              </div>
            </div>
            {/* Mini sparkline of process quality across recent sims */}
            <div className="mt-4 flex items-end gap-1.5 h-12">
              {scores.map((sim, i) => {
                const pq = Math.round(sim.process_quality_score);
                return (
                  <div key={i} className="flex-1 flex flex-col items-center gap-0.5" title={`${sim.scenario?.name}: ${pq}%`}>
                    <div
                      className={clsx('w-full rounded-t-md transition-all', pq >= 60 ? 'bg-emerald-400' : pq >= 40 ? 'bg-amber-400' : 'bg-red-400')}
                      style={{ height: `${Math.max(pq, 5)}%` }}
                    />
                  </div>
                );
              })}
            </div>
            <p className="text-xs text-brand-navy/40 mt-2 text-center">Process quality across your last {scores.length} simulations</p>
          </div>
        );
      })()}

      {/* Recent Simulations */}
      {recentSimulations.length > 0 && (
        <div className="animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
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
      <div className="card bg-gradient-to-r from-brand-lavender/50 to-brand-cream border-brand-navy/20 animate-fade-in-up" style={{ animationDelay: '0.3s' }}>
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
