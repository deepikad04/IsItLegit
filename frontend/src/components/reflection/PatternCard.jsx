import clsx from 'clsx';
import { CheckCircle } from 'lucide-react';

export default function PatternCard({ pattern }) {
  const confidencePct = Math.round(pattern.confidence * 100);

  return (
    <div className={clsx(
      'p-4 rounded-xl border-l-4 bg-white shadow-sm',
      pattern.confidence >= 0.7 ? 'border-red-500' :
        pattern.confidence >= 0.5 ? 'border-amber-500' :
          'border-brand-blue'
    )}>
      <div className="flex items-center justify-between mb-2">
        <h4 className="font-bold text-brand-navy capitalize text-base">
          {pattern.pattern_name.replace(/_/g, ' ')}
        </h4>
        <div className="flex items-center gap-2">
          <div className="w-16 h-2 bg-gray-100 rounded-full overflow-hidden">
            <div
              className={clsx(
                'h-full rounded-full',
                pattern.confidence >= 0.7 ? 'bg-red-500' :
                  pattern.confidence >= 0.5 ? 'bg-amber-500' : 'bg-brand-blue'
              )}
              style={{ width: `${confidencePct}%` }}
            />
          </div>
          <span className={clsx(
            'text-xs font-bold px-2 py-0.5 rounded-full',
            pattern.confidence >= 0.7 ? 'bg-red-100 text-red-700' :
              pattern.confidence >= 0.5 ? 'bg-yellow-100 text-amber-700' :
                'bg-brand-blue/20 text-brand-navy/70'
          )}>
            {confidencePct}%
          </span>
        </div>
      </div>
      <p className="text-brand-navy/60 text-sm mb-3">{pattern.description}</p>
      <div className="space-y-1.5">
        {pattern.evidence.map((e, i) => (
          <div key={i} className="flex items-start space-x-2 text-sm text-brand-navy/70">
            <CheckCircle className="h-3.5 w-3.5 text-brand-navy/40 flex-shrink-0 mt-0.5" />
            <span>{e}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
