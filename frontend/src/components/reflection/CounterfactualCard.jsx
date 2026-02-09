import clsx from 'clsx';
import { GitBranch } from 'lucide-react';

export default function CounterfactualCard({ cf, index }) {
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
