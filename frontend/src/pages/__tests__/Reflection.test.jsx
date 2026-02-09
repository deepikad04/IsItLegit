import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import Reflection from '../Reflection';

// Mock the API client
vi.mock('../../api/client', () => ({
  reflectionApi: {
    getFull: vi.fn(),
    get: vi.fn(),
    getCounterfactuals: vi.fn().mockResolvedValue({ data: [] }),
    getWhyDecisions: vi.fn().mockResolvedValue({ data: null }),
    getProComparison: vi.fn().mockResolvedValue({ data: null }),
    getCoaching: vi.fn().mockResolvedValue({ data: null }),
    getBiasHeatmap: vi.fn().mockResolvedValue({ data: null }),
    getRationaleReview: vi.fn().mockResolvedValue({ data: null }),
    getCalibration: vi.fn().mockResolvedValue({ data: null }),
    getOutcomeDistribution: vi.fn().mockResolvedValue({ data: null }),
    getCounterfactualIsolation: vi.fn().mockResolvedValue({ data: null }),
  },
}));

// Mock static asset imports
vi.mock('../../../assests/logo.png', () => ({ default: 'logo.png' }));

// Mock recharts to avoid canvas issues in jsdom
vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }) => children,
  LineChart: ({ children }) => <div data-testid="line-chart">{children}</div>,
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
  ReferenceLine: () => null,
}));

const { reflectionApi } = await import('../../api/client');

const mockReflection = {
  simulation_id: 'test-123',
  outcome_summary: '+$500.00',
  outcome_type: 'profit',
  process_quality: {
    score: 75,
    factors: { timing: 0.8, information_usage: 0.7, risk_sizing: 0.9, emotional_indicators: 0.6 },
    summary: 'Strong analytical approach.',
  },
  patterns_detected: [
    {
      pattern_name: 'fomo',
      confidence: 0.65,
      description: 'Fear of missing out detected',
      evidence: ['Bought quickly after price spike'],
    },
  ],
  luck_factor: 0.3,
  skill_factor: 0.7,
  luck_skill_explanation: 'Your decisions drove the outcome.',
  insights: [{ title: 'Good timing', description: 'You waited before acting.' }],
  key_takeaway: 'Solid process led to a good outcome.',
  coaching_message: 'Keep up the good work!',
};

function renderReflection() {
  return render(
    <MemoryRouter initialEntries={['/reflection/test-123']}>
      <Routes>
        <Route path="/reflection/:simulationId" element={<Reflection />} />
      </Routes>
    </MemoryRouter>
  );
}

describe('Reflection Page', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows loading state initially', () => {
    reflectionApi.getFull.mockReturnValue(new Promise(() => {})); // never resolves
    renderReflection();
    expect(screen.getByText('Analyzing your decisions...')).toBeInTheDocument();
  });

  it('shows error state when reflection fails to load', async () => {
    reflectionApi.getFull.mockRejectedValue(new Error('fail'));
    reflectionApi.get.mockRejectedValue(new Error('fail'));
    renderReflection();
    await waitFor(() => {
      expect(screen.getByText('Could not load reflection analysis')).toBeInTheDocument();
    });
  });

  it('renders the outcome summary after loading', async () => {
    reflectionApi.getFull.mockResolvedValue({
      data: { reflection: mockReflection, counterfactuals: [] },
    });
    renderReflection();
    await waitFor(() => {
      expect(screen.getByText('+$500.00')).toBeInTheDocument();
    });
  });

  it('renders the key takeaway', async () => {
    reflectionApi.getFull.mockResolvedValue({
      data: { reflection: mockReflection, counterfactuals: [] },
    });
    renderReflection();
    await waitFor(() => {
      expect(screen.getByText('Solid process led to a good outcome.')).toBeInTheDocument();
    });
  });

  it('shows the summary mode by default with narrative sentence', async () => {
    reflectionApi.getFull.mockResolvedValue({
      data: { reflection: mockReflection, counterfactuals: [] },
    });
    renderReflection();
    await waitFor(() => {
      expect(screen.getByText(/strong decisions/)).toBeInTheDocument();
      expect(screen.getByText('Show Full Analysis')).toBeInTheDocument();
    });
  });

  it('expands full analysis when button is clicked', async () => {
    reflectionApi.getFull.mockResolvedValue({
      data: { reflection: mockReflection, counterfactuals: [] },
    });
    renderReflection();
    await waitFor(() => {
      expect(screen.getByText('Show Full Analysis')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText('Show Full Analysis'));

    expect(screen.getByText('Process Quality')).toBeInTheDocument();
    expect(screen.getByText('Luck vs Skill')).toBeInTheDocument();
    expect(screen.getByText('Collapse to summary')).toBeInTheDocument();
  });

  it('displays process score badge in summary', async () => {
    reflectionApi.getFull.mockResolvedValue({
      data: { reflection: mockReflection, counterfactuals: [] },
    });
    renderReflection();
    await waitFor(() => {
      expect(screen.getByText('75')).toBeInTheDocument();
      expect(screen.getByText('Strong decisions')).toBeInTheDocument();
    });
  });
});
