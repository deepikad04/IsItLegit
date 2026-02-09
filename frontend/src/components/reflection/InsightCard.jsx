import { Lightbulb, ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function InsightCard({ insight }) {
  return (
    <div className="p-4 bg-gradient-to-r from-brand-lavender/50 to-brand-cream rounded-lg border border-brand-navy/30">
      <div className="flex items-start space-x-3">
        <Lightbulb className="h-5 w-5 text-brand-navy flex-shrink-0 mt-0.5" />
        <div>
          <h4 className="font-medium text-brand-navy mb-1">{insight.title}</h4>
          <p className="text-brand-navy/70 text-sm">{insight.description}</p>
          {insight.recommended_card_id && (
            <Link
              to="/learning"
              className="inline-flex items-center space-x-1 text-brand-navy hover:text-brand-navy-light text-sm mt-2"
            >
              <span>Learn more</span>
              <ArrowRight className="h-4 w-4" />
            </Link>
          )}
        </div>
      </div>
    </div>
  );
}
