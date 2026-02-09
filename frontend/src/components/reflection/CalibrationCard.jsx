import clsx from 'clsx';
import { Target } from 'lucide-react';

export default function CalibrationCard({ data }) {
  const total = data.overconfident_count + data.underconfident_count + data.well_calibrated_count;
  if (total === 0) return null;

  const pct = (count) => Math.round((count / total) * 100);
  const scoreColor = data.calibration_score >= 70 ? 'text-green-700' :
    data.calibration_score >= 40 ? 'text-amber-600' : 'text-red-600';

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-brand-navy mb-4 flex items-center space-x-2">
        <Target className="h-5 w-5 text-brand-navy" />
        <span>Confidence Calibration</span>
      </h3>
      <div className="text-center mb-4">
        <span className={clsx('text-4xl font-bold', scoreColor)}>{Math.round(data.calibration_score)}</span>
        <span className="text-brand-navy/60 text-sm ml-1">/ 100</span>
        <p className="text-brand-navy/60 text-sm mt-1">
          How well your confidence matched actual outcomes
        </p>
      </div>
      <div className="space-y-3">
        {[
          { label: 'Well Calibrated', count: data.well_calibrated_count, color: 'bg-green-500', textColor: 'text-green-700' },
          { label: 'Overconfident', count: data.overconfident_count, color: 'bg-red-500', textColor: 'text-red-600' },
          { label: 'Underconfident', count: data.underconfident_count, color: 'bg-yellow-500', textColor: 'text-amber-600' },
        ].map((item) => (
          <div key={item.label}>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-brand-navy/60">{item.label}</span>
              <span className={item.textColor}>{item.count} ({pct(item.count)}%)</span>
            </div>
            <div className="h-3 bg-brand-blue/20 rounded-full overflow-hidden">
              <div className={clsx('h-full rounded-full transition-all duration-500', item.color)}
                style={{ width: `${pct(item.count)}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
