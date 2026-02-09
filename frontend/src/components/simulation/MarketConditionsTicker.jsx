import { Activity, Users, DollarSign, Ban } from 'lucide-react';
import clsx from 'clsx';
import { volLabel } from './helpers';

export default function MarketConditionsTicker({ mc, fees, drawdownPct, marginStatus, crowdSentiment, isHalted }) {
  if (!mc) return null;

  return (
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
  );
}
