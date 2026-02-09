import { useState, useEffect } from 'react';
import { MessageSquare, X } from 'lucide-react';
import clsx from 'clsx';

const BIAS_COLORS = {
  fomo: 'border-orange-500 bg-orange-500/10',
  impulsivity: 'border-red-500 bg-red-500/10',
  loss_aversion: 'border-yellow-500 bg-yellow-500/10',
  overconfidence: 'border-purple-500 bg-purple-500/10',
  anchoring: 'border-blue-500 bg-blue-500/10',
  social_proof_reliance: 'border-cyan-500 bg-cyan-500/10',
};

const BIAS_LABELS = {
  fomo: 'FOMO',
  impulsivity: 'Impulsivity',
  loss_aversion: 'Loss Aversion',
  overconfidence: 'Overconfidence',
  anchoring: 'Anchoring',
  social_proof_reliance: 'Social Proof',
};

export default function CoachNudge({ nudge, onDismiss }) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (nudge) {
      setVisible(true);
      const timer = setTimeout(() => {
        setVisible(false);
        setTimeout(onDismiss, 300); // wait for exit animation
      }, 8000);
      return () => clearTimeout(timer);
    }
  }, [nudge, onDismiss]);

  if (!nudge) return null;

  const colorClass = BIAS_COLORS[nudge.bias] || 'border-primary-500 bg-primary-500/10';

  return (
    <div className={clsx(
      'fixed top-4 right-4 z-50 max-w-sm transition-all duration-300 ease-out',
      visible ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0'
    )}>
      <div className={clsx('rounded-xl border-l-4 p-4 shadow-2xl backdrop-blur-sm', colorClass)}>
        <div className="flex items-start gap-3">
          <MessageSquare className="h-5 w-5 text-white flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-semibold text-brand-lavender uppercase tracking-wider">
                Coach: {BIAS_LABELS[nudge.bias] || nudge.bias}
              </span>
              <button onClick={() => { setVisible(false); onDismiss(); }} className="text-brand-blue hover:text-white">
                <X className="h-4 w-4" />
              </button>
            </div>
            <p className="text-sm text-white font-medium">{nudge.message}</p>
          </div>
        </div>
        {/* Progress bar for auto-dismiss */}
        <div className="mt-2 h-0.5 bg-brand-navy-light rounded-full overflow-hidden">
          <div className="h-full bg-white/40 rounded-full animate-shrink" style={{ animationDuration: '8s' }} />
        </div>
      </div>
    </div>
  );
}
