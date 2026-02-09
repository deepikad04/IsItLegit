import { X, Minus } from 'lucide-react';
import clsx from 'clsx';

export default function DecisionHistoryDrawer({ show, onClose, decisions }) {
  if (!show) return null;

  return (
    <div className="fixed right-0 top-0 bottom-0 w-full sm:w-80 bg-white border-l border-gray-200 shadow-2xl z-40 flex flex-col">
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <h3 className="text-sm font-bold text-gray-900 uppercase tracking-wider">Your Moves</h3>
        <button onClick={onClose} className="text-gray-400 hover:text-gray-700">
          <X className="h-5 w-5" />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {decisions.length === 0 ? (
          <div className="text-center py-8">
            <Minus className="h-8 w-8 text-gray-300 mx-auto mb-2" />
            <p className="text-sm text-gray-400">No decisions yet</p>
            <p className="text-xs text-gray-300 mt-1">Your moves will appear here</p>
          </div>
        ) : (
          decisions.map((d, i) => (
            <div key={i} className={clsx(
              'p-3 rounded-xl border text-sm',
              d.type === 'buy' ? 'bg-emerald-50 border-emerald-200' :
              d.type === 'sell' ? 'bg-red-50 border-red-200' :
              d.type === 'hold' ? 'bg-blue-50 border-blue-200' :
              'bg-amber-50 border-amber-200'
            )}>
              <div className="flex items-center justify-between mb-1">
                <span className={clsx(
                  'font-bold uppercase text-xs tracking-wide',
                  d.type === 'buy' ? 'text-emerald-700' :
                  d.type === 'sell' ? 'text-red-700' :
                  d.type === 'hold' ? 'text-blue-700' :
                  'text-amber-700'
                )}>
                  #{i + 1} {d.type}
                </span>
                <span className="text-xs text-gray-400">{Math.floor(d.time / 60)}:{(d.time % 60).toString().padStart(2, '0')}</span>
              </div>
              {(d.type === 'buy' || d.type === 'sell') && (
                <p className="text-xs text-gray-600">
                  ${d.amount} @ ${d.price?.toFixed(2)}
                  {d.orderType !== 'market' && ` (${d.orderType})`}
                </p>
              )}
              {d.rationale && (
                <p className="text-xs text-gray-500 mt-1 italic truncate">"{d.rationale}"</p>
              )}
            </div>
          ))
        )}
      </div>
      <div className="p-4 border-t border-gray-200 bg-gray-50">
        <p className="text-xs text-gray-400 text-center">{decisions.length} decisions made</p>
      </div>
    </div>
  );
}
