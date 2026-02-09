import clsx from 'clsx';

export default function ProcessGauge({ score }) {
  const getColor = () => {
    if (score >= 70) return 'text-green-600';
    if (score >= 50) return 'text-amber-500';
    return 'text-red-500';
  };

  const getStrokeColor = () => {
    if (score >= 70) return '#16a34a';
    if (score >= 50) return '#f59e0b';
    return '#ef4444';
  };

  const getLabel = () => {
    if (score >= 70) return 'Strong';
    if (score >= 50) return 'Average';
    return 'Needs Work';
  };

  const circumference = 2 * Math.PI * 56;
  const dashOffset = circumference - (score / 100) * circumference;

  return (
    <div className="text-center">
      <div className="relative w-36 h-36 mx-auto">
        <svg className="w-36 h-36 transform -rotate-90" viewBox="0 0 128 128">
          <circle
            cx="64" cy="64" r="56"
            stroke="#e5e7eb"
            strokeWidth="10"
            fill="none"
          />
          <circle
            cx="64" cy="64" r="56"
            stroke={getStrokeColor()}
            strokeWidth="10"
            fill="none"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            style={{
              transition: 'stroke-dashoffset 1.2s ease-out',
            }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={clsx('text-4xl font-black', getColor())}>{Math.round(score)}</span>
          <span className="text-brand-navy/50 text-xs font-semibold uppercase tracking-wider mt-0.5">{getLabel()}</span>
        </div>
      </div>
    </div>
  );
}
