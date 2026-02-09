import { Component } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

export default class SectionErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error(`SectionErrorBoundary [${this.props.section || 'unknown'}]:`, error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="card p-5">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 rounded-xl bg-amber-100 flex items-center justify-center flex-shrink-0">
              <AlertTriangle className="h-5 w-5 text-amber-600" />
            </div>
            <div className="flex-1">
              <p className="text-sm font-semibold text-brand-navy">
                {this.props.section || 'This section'} couldn't load
              </p>
              <p className="text-xs text-brand-navy/50 mt-1">
                The AI analysis may be temporarily unavailable.
              </p>
              <button
                onClick={this.handleReset}
                className="mt-3 text-xs font-semibold text-brand-navy hover:text-brand-navy-light flex items-center gap-1.5 transition-colors"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                Try Again
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
