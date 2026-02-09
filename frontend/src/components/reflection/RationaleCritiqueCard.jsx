import clsx from 'clsx';

export default function RationaleCritiqueCard({ review }) {
  const qualityColors = {
    5: 'text-green-700 bg-green-100',
    4: 'text-green-700 bg-green-100',
    3: 'text-amber-600 bg-yellow-100',
    2: 'text-orange-600 bg-orange-100',
    1: 'text-red-600 bg-red-100',
  };
  const style = qualityColors[review.quality_score] || qualityColors[3];

  return (
    <div className="p-4 bg-brand-lavender/30 rounded-lg border-l-4 border-brand-navy">
      <div className="flex items-center justify-between mb-2">
        <h4 className="font-medium text-brand-navy">Decision #{review.decision_index + 1}</h4>
        <span className={clsx('text-xs px-2 py-1 rounded font-semibold', style)}>
          Quality: {review.quality_score}/5
        </span>
      </div>
      {review.user_rationale && (
        <p className="text-brand-navy/60 text-sm mb-2 italic">
          You said: "{review.user_rationale}"
        </p>
      )}
      <p className="text-brand-navy/70 text-sm mb-3">{review.critique}</p>
      {review.missed_factors?.length > 0 && (
        <div className="mb-2">
          <p className="text-xs text-brand-blue uppercase mb-1">Missed Factors</p>
          <div className="flex flex-wrap gap-1">
            {review.missed_factors.map((f, i) => (
              <span key={i} className="text-xs px-2 py-0.5 bg-orange-100 text-orange-600 rounded">
                {f}
              </span>
            ))}
          </div>
        </div>
      )}
      {review.reasoning_bias && (
        <p className="text-xs text-red-600">
          Reasoning bias detected: {review.reasoning_bias}
        </p>
      )}
    </div>
  );
}
