import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import BiasHeatmap from '../BiasHeatmap';

const mockData = {
  timeline: [
    {
      decision_index: 0,
      time: 10,
      biases: {
        fomo: 0.8,
        impulsivity: 0.5,
        loss_aversion: 0.1,
        overconfidence: 0.3,
        anchoring: 0.0,
        social_proof_reliance: 0.6,
      },
    },
    {
      decision_index: 1,
      time: 50,
      biases: {
        fomo: 0.2,
        impulsivity: 0.1,
        loss_aversion: 0.7,
        overconfidence: 0.0,
        anchoring: 0.4,
        social_proof_reliance: 0.1,
      },
    },
  ],
  peak_bias_moment: 10,
  dominant_bias: 'fomo',
};

describe('BiasHeatmap', () => {
  it('renders nothing when data is null', () => {
    const { container } = render(<BiasHeatmap data={null} />);
    expect(container.innerHTML).toBe('');
  });

  it('renders nothing when timeline is empty', () => {
    const { container } = render(<BiasHeatmap data={{ timeline: [] }} />);
    expect(container.innerHTML).toBe('');
  });

  it('renders the heatmap title', () => {
    render(<BiasHeatmap data={mockData} />);
    expect(screen.getByText('Bias Heatmap Timeline')).toBeInTheDocument();
  });

  it('shows the peak bias moment', () => {
    render(<BiasHeatmap data={mockData} />);
    expect(screen.getByText('Peak at t=10s')).toBeInTheDocument();
  });

  it('displays all bias type labels', () => {
    render(<BiasHeatmap data={mockData} />);
    // FOMO appears twice: as row label and as dominant bias badge
    expect(screen.getAllByText('FOMO').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('Loss Aversion')).toBeInTheDocument();
    expect(screen.getByText('Overconfidence')).toBeInTheDocument();
  });

  it('shows the dominant bias badge', () => {
    render(<BiasHeatmap data={mockData} />);
    // dominant_bias 'fomo' maps to BIAS_LABELS -> 'FOMO'
    const fomoElements = screen.getAllByText('FOMO');
    // At least 2: one row label, one dominant bias badge
    expect(fomoElements.length).toBeGreaterThanOrEqual(2);
  });
});
