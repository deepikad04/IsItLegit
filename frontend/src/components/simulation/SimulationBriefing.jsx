import { Play } from 'lucide-react';
import clsx from 'clsx';
import { getActiveFeatures, DIFFICULTY_LABELS, DIFFICULTY_COLORS } from './helpers';

export default function SimulationBriefing({ scenario, loading, onStart, onBack }) {
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
          <button onClick={onBack} className="text-sm font-semibold text-gray-400 hover:text-gray-700 transition-colors">
            Back to Dashboard
          </button>
          <button onClick={onStart} disabled={loading}
            className="flex items-center gap-3 bg-brand-navy text-white px-8 py-4 rounded-xl font-bold text-lg hover:bg-brand-navy-light transition-all shadow-lg hover:shadow-xl active:scale-[0.98]">
            <Play className="h-6 w-6" />
            {loading ? 'Starting...' : 'Begin Simulation'}
          </button>
        </div>
      </div>
    </div>
  );
}
