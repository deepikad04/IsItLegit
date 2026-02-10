import { useState, useRef } from 'react';
import { Upload, Image, AlertTriangle, TrendingUp, TrendingDown, Minus, Eye, Loader2, X } from 'lucide-react';
import { reflectionApi } from '../api/client';

const BIAS_COLORS = {
  high: 'bg-red-100 text-red-800 border-red-200',
  medium: 'bg-amber-100 text-amber-800 border-amber-200',
  low: 'bg-green-100 text-green-800 border-green-200',
};

const ACTION_ICONS = {
  buy: TrendingUp,
  sell: TrendingDown,
  hold: Minus,
  wait: Eye,
};

export default function ChartAnalysis() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  const handleFile = (f) => {
    if (!f) return;
    const allowed = ['image/png', 'image/jpeg', 'image/webp', 'image/gif'];
    if (!allowed.includes(f.type)) {
      setError('Please upload a PNG, JPEG, WebP, or GIF image.');
      return;
    }
    if (f.size > 10 * 1024 * 1024) {
      setError('Image too large. Maximum size is 10 MB.');
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setError(null);
    setResult(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await reflectionApi.analyzeChart(file);
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Analysis failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  const ActionIcon = result ? (ACTION_ICONS[result.recommended_action] || Eye) : Eye;

  return (
    <div className="card animate-fade-in-up">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-8 h-8 rounded-lg bg-brand-blue/10 flex items-center justify-center">
          <Image className="w-4 h-4 text-brand-blue" />
        </div>
        <div>
          <h3 className="font-semibold text-brand-navy">Chart Analysis</h3>
          <p className="text-xs text-brand-navy/50">Gemini Vision — upload any trading chart</p>
        </div>
      </div>

      {!preview && (
        <div
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
            dragOver ? 'border-brand-blue bg-brand-blue/5' : 'border-brand-navy/20 hover:border-brand-blue/50'
          }`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          <Upload className="w-8 h-8 mx-auto mb-2 text-brand-navy/30" />
          <p className="text-sm text-brand-navy/60 mb-1">Drop a chart screenshot here or click to browse</p>
          <p className="text-xs text-brand-navy/40">PNG, JPEG, WebP, or GIF — max 10 MB</p>
          <input
            ref={inputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp,image/gif"
            className="hidden"
            onChange={(e) => handleFile(e.target.files[0])}
          />
        </div>
      )}

      {preview && !result && (
        <div className="space-y-3">
          <div className="relative">
            <img src={preview} alt="Chart preview" className="w-full rounded-lg border border-brand-navy/10 max-h-64 object-contain bg-white" />
            <button onClick={reset} className="absolute top-2 right-2 p-1 bg-white/90 rounded-full shadow hover:bg-white">
              <X className="w-4 h-4 text-brand-navy/60" />
            </button>
          </div>
          <button
            onClick={analyze}
            disabled={loading}
            className="w-full btn-primary flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Analyzing with Gemini Vision...
              </>
            ) : (
              <>
                <Eye className="w-4 h-4" />
                Analyze Chart
              </>
            )}
          </button>
        </div>
      )}

      {error && (
        <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-4 mt-1">
          <div className="relative">
            <img src={preview} alt="Analyzed chart" className="w-full rounded-lg border border-brand-navy/10 max-h-48 object-contain bg-white" />
            <button onClick={reset} className="absolute top-2 right-2 p-1 bg-white/90 rounded-full shadow hover:bg-white">
              <X className="w-4 h-4 text-brand-navy/60" />
            </button>
          </div>

          {/* Trend + Action */}
          <div className="flex items-start gap-3">
            <div className="flex-1 p-3 bg-brand-cream/50 rounded-lg">
              <p className="text-xs font-medium text-brand-navy/50 mb-1">Trend</p>
              <p className="text-sm text-brand-navy">{result.trend_summary}</p>
            </div>
            <div className="p-3 bg-brand-cream/50 rounded-lg text-center min-w-[90px]">
              <p className="text-xs font-medium text-brand-navy/50 mb-1">Action</p>
              <div className="flex items-center justify-center gap-1">
                <ActionIcon className="w-4 h-4 text-brand-blue" />
                <span className="text-sm font-semibold text-brand-navy capitalize">{result.recommended_action}</span>
              </div>
              <p className="text-[10px] text-brand-navy/40 mt-0.5">{Math.round(result.confidence * 100)}% conf.</p>
            </div>
          </div>

          {/* Patterns */}
          {result.key_patterns?.length > 0 && (
            <div>
              <p className="text-xs font-medium text-brand-navy/50 mb-1.5">Patterns Detected</p>
              <div className="flex flex-wrap gap-1.5">
                {result.key_patterns.map((p, i) => (
                  <span key={i} className="px-2 py-0.5 bg-brand-blue/10 text-brand-blue text-xs rounded-full">{p}</span>
                ))}
              </div>
            </div>
          )}

          {/* Bias Warnings */}
          {result.bias_warnings?.length > 0 && (
            <div>
              <div className="flex items-center gap-1.5 mb-2">
                <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
                <p className="text-xs font-medium text-brand-navy/50">Bias Warnings</p>
              </div>
              <div className="space-y-2">
                {result.bias_warnings.map((w, i) => (
                  <div key={i} className={`p-2.5 rounded-lg border ${BIAS_COLORS[w.risk_level] || BIAS_COLORS.medium}`}>
                    <div className="flex items-center justify-between mb-0.5">
                      <span className="text-xs font-semibold capitalize">{w.bias.replace(/_/g, ' ')}</span>
                      <span className="text-[10px] uppercase font-medium">{w.risk_level} risk</span>
                    </div>
                    <p className="text-xs opacity-80">{w.explanation}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Reasoning */}
          <div className="p-3 bg-brand-navy/5 rounded-lg">
            <p className="text-xs font-medium text-brand-navy/50 mb-1">AI Reasoning</p>
            <p className="text-sm text-brand-navy/80">{result.reasoning}</p>
          </div>

          {result._source === 'heuristic' && (
            <p className="text-[10px] text-brand-navy/40 text-center">Heuristic fallback — Gemini API unavailable</p>
          )}
        </div>
      )}
    </div>
  );
}
