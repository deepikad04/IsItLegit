import { ArrowRight, Zap } from 'lucide-react';

export default function IsolatedCounterfactualCard({ data }) {
  return (
    <div className="p-4 bg-brand-lavender/30 rounded-lg border-l-4 border-brand-navy">
      <div className="flex items-center justify-between mb-3">
        <div>
          <p className="text-xs text-brand-blue uppercase">Original</p>
          <p className="text-brand-navy font-medium">{data.original_decision}</p>
        </div>
        <ArrowRight className="h-5 w-5 text-brand-navy mx-4" />
        <div className="text-right">
          <p className="text-xs text-brand-navy uppercase">Alternative</p>
          <p className="text-brand-navy/80 font-medium">{data.alternative_decision}</p>
        </div>
      </div>
      {data.ripple_effects?.length > 0 && (
        <div className="mb-3">
          <p className="text-xs text-brand-blue uppercase mb-1">Ripple Effects</p>
          <div className="space-y-1">
            {data.ripple_effects.map((effect, i) => (
              <div key={i} className="flex items-start space-x-2 text-sm text-brand-navy/70">
                <Zap className="h-3 w-3 text-brand-navy flex-shrink-0 mt-1" />
                <span>{effect}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      <div className="p-3 bg-brand-lavender/30 rounded-lg border border-brand-blue/30">
        <p className="text-sm text-brand-navy/70">
          <span className="text-brand-navy font-medium">Causal Impact: </span>
          {data.causal_impact}
        </p>
      </div>
      <p className="text-sm text-brand-navy/70 italic mt-2">"{data.lesson}"</p>
    </div>
  );
}
