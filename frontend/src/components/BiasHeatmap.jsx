import { useState } from 'react';
import clsx from 'clsx';

const BIAS_LABELS = {
  fomo: 'FOMO',
  impulsivity: 'Impulsivity',
  loss_aversion: 'Loss Aversion',
  overconfidence: 'Overconfidence',
  anchoring: 'Anchoring',
  social_proof_reliance: 'Social Proof',
};

const intensityColor = (value) => {
  if (value >= 0.7) return 'bg-red-500';
  if (value >= 0.4) return 'bg-yellow-500';
  if (value > 0) return 'bg-green-500';
  return 'bg-brand-blue/20';
};

const intensityOpacity = (value) => {
  if (value === 0) return 'opacity-20';
  if (value < 0.3) return 'opacity-40';
  if (value < 0.6) return 'opacity-70';
  return 'opacity-100';
};

export default function BiasHeatmap({ data }) {
  const [selectedCell, setSelectedCell] = useState(null);

  if (!data || !data.timeline || data.timeline.length === 0) {
    return null;
  }

  const { timeline, peak_bias_moment, dominant_bias } = data;
  const biasTypes = Object.keys(BIAS_LABELS);

  return (
    <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-md">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-brand-navy flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-red-100 flex items-center justify-center">
            <span className="text-sm">🔥</span>
          </div>
          Bias Heatmap Timeline
        </h3>
        <div className="flex items-center gap-3">
          <span className="text-xs font-bold px-2.5 py-1 rounded-full bg-red-100 text-red-700">
            Peak at t={peak_bias_moment}s
          </span>
          <span className="text-xs font-bold px-2.5 py-1 rounded-full bg-amber-100 text-amber-700 capitalize">
            {BIAS_LABELS[dominant_bias] || dominant_bias}
          </span>
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-5 mb-4 text-xs font-semibold text-brand-navy/50">
        <div className="flex items-center gap-1.5">
          <div className="w-4 h-4 rounded-md bg-gray-100 border border-gray-200" />
          <span>None</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-4 h-4 rounded-md bg-green-400" />
          <span>Low</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-4 h-4 rounded-md bg-yellow-400" />
          <span>Medium</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-4 h-4 rounded-md bg-red-500" />
          <span>High</span>
        </div>
      </div>

      {/* Heatmap grid */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr>
              <th className="text-left text-xs text-brand-blue pb-2 pr-4 w-32">Bias</th>
              {timeline.map((entry, i) => (
                <th key={i} className="text-center text-xs text-brand-blue pb-2 px-1">
                  D{entry.decision_index + 1}
                  <br />
                  <span className="text-brand-navy/60">{entry.timestamp_seconds}s</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {biasTypes.map((bias) => (
              <tr key={bias}>
                <td className="text-sm text-brand-navy/70 pr-4 py-1">{BIAS_LABELS[bias]}</td>
                {timeline.map((entry, i) => {
                  const value = entry.biases[bias] || 0;
                  const isSelected = selectedCell?.bias === bias && selectedCell?.index === i;
                  return (
                    <td key={i} className="px-1 py-1">
                      <button
                        onClick={() => setSelectedCell(isSelected ? null : { bias, index: i, entry })}
                        className={clsx(
                          'w-full h-8 rounded-md cursor-pointer transition-all duration-200 hover:scale-110 hover:shadow-md',
                          intensityColor(value),
                          intensityOpacity(value),
                          isSelected && 'ring-2 ring-brand-navy ring-offset-1'
                        )}
                        title={`${BIAS_LABELS[bias]}: ${(value * 100).toFixed(0)}%`}
                      />
                    </td>
                  );
                })}
              </tr>
            ))}
            {/* Intensity row */}
            <tr>
              <td className="text-sm text-brand-blue pr-4 py-1 pt-2 border-t border-brand-blue/20">Overall</td>
              {timeline.map((entry, i) => (
                <td key={i} className="px-1 py-1 pt-2 border-t border-brand-blue/20">
                  <div className={clsx(
                    'text-center text-xs font-bold rounded py-1',
                    entry.intensity === 'high' ? 'text-red-600 bg-red-100' :
                    entry.intensity === 'medium' ? 'text-amber-600 bg-yellow-100' :
                    'text-green-700 bg-green-100'
                  )}>
                    {entry.intensity}
                  </div>
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>

      {/* Selected cell detail */}
      {selectedCell && (
        <div className="mt-4 p-4 bg-gradient-to-r from-brand-lavender/30 to-brand-cream rounded-xl border border-brand-blue/20 text-sm animate-fadeIn">
          <div className="flex items-center gap-3 mb-2">
            <span className="text-brand-navy font-bold">Decision #{selectedCell.entry.decision_index + 1}</span>
            <span className="text-xs text-brand-navy/50 font-mono">t={selectedCell.entry.timestamp_seconds}s</span>
            <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-amber-100 text-amber-700">
              {BIAS_LABELS[selectedCell.bias]}: {((selectedCell.entry.biases[selectedCell.bias] || 0) * 100).toFixed(0)}%
            </span>
          </div>
          <p className="text-brand-navy/60">{selectedCell.entry.evidence}</p>
        </div>
      )}
    </div>
  );
}
