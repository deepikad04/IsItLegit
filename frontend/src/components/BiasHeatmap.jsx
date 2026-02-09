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
    <div className="bg-white/90 backdrop-blur-sm rounded-2xl p-6 border border-brand-blue/30 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-brand-navy">Bias Heatmap Timeline</h3>
        <div className="flex items-center gap-4 text-sm">
          <span className="text-brand-navy/60">
            Peak bias at <span className="text-red-500 font-semibold">t={peak_bias_moment}s</span>
          </span>
          <span className="text-brand-navy/60">
            Dominant: <span className="text-yellow-600 font-semibold">{BIAS_LABELS[dominant_bias] || dominant_bias}</span>
          </span>
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 mb-4 text-xs text-brand-navy/60">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-brand-blue/20 opacity-20" />
          <span>None</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-green-500 opacity-60" />
          <span>Low</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-yellow-500 opacity-80" />
          <span>Medium</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-red-500" />
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
                          'w-full h-8 rounded cursor-pointer transition-all',
                          intensityColor(value),
                          intensityOpacity(value),
                          isSelected && 'ring-2 ring-brand-navy'
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
        <div className="mt-4 p-3 bg-brand-lavender/40 rounded-lg text-sm">
          <div className="text-brand-navy/70">
            <span className="text-brand-navy font-semibold">Decision #{selectedCell.entry.decision_index + 1}</span>
            {' '}at t={selectedCell.entry.timestamp_seconds}s —{' '}
            <span className="text-amber-600">{BIAS_LABELS[selectedCell.bias]}</span>
            {': '}
            {((selectedCell.entry.biases[selectedCell.bias] || 0) * 100).toFixed(0)}%
          </div>
          <p className="text-brand-navy/60 mt-1">{selectedCell.entry.evidence}</p>
        </div>
      )}
    </div>
  );
}
