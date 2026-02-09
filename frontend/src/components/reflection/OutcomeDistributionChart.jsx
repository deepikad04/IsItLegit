import clsx from 'clsx';
import { BarChart2 } from 'lucide-react';

export default function OutcomeDistributionChart({ data }) {
  if (!data?.buckets?.length) return null;

  const maxCount = Math.max(...data.buckets.map((b) => b.count));

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-brand-navy mb-4 flex items-center space-x-2">
        <BarChart2 className="h-5 w-5 text-brand-navy" />
        <span>Outcome Distribution</span>
      </h3>
      <p className="text-brand-navy/60 text-sm mb-4">
        Your decisions replayed across 100 different market scenarios
      </p>

      {/* Histogram */}
      <div className="flex items-end gap-1 h-32 mb-2">
        {data.buckets.map((bucket, i) => {
          const height = maxCount > 0 ? (bucket.count / maxCount) * 100 : 0;
          const isUserBucket = data.actual_outcome >= bucket.min && data.actual_outcome < bucket.max;
          return (
            <div key={i} className="flex-1 flex flex-col items-center justify-end h-full group relative">
              <div
                className={clsx(
                  'w-full rounded-t transition-all duration-300',
                  isUserBucket ? 'bg-brand-navy' : 'bg-brand-blue/30 group-hover:bg-brand-blue/40'
                )}
                style={{ height: `${Math.max(height, 2)}%` }}
              />
              {isUserBucket && (
                <div className="absolute -top-6 text-xs text-brand-navy font-semibold whitespace-nowrap">
                  You
                </div>
              )}
            </div>
          );
        })}
      </div>
      <div className="flex justify-between text-xs text-brand-blue">
        <span>${data.buckets[0]?.min?.toFixed(0)}</span>
        <span>${data.buckets[data.buckets.length - 1]?.max?.toFixed(0)}</span>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t border-brand-blue/20">
        <div className="text-center">
          <p className="text-xs text-brand-blue">Worst</p>
          <p className="text-red-600 font-semibold">${data.worst_outcome?.toFixed(2)}</p>
        </div>
        <div className="text-center">
          <p className="text-xs text-brand-blue">Median</p>
          <p className="text-brand-navy/70 font-semibold">${data.median_outcome?.toFixed(2)}</p>
        </div>
        <div className="text-center">
          <p className="text-xs text-brand-blue">Best</p>
          <p className="text-green-700 font-semibold">${data.best_outcome?.toFixed(2)}</p>
        </div>
      </div>
      <div className="text-center mt-3">
        <p className="text-xs text-brand-blue">Your Percentile</p>
        <p className="text-brand-navy font-bold text-lg">{data.user_percentile}th</p>
      </div>
    </div>
  );
}
