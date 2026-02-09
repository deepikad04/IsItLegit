import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import Simulation from '../Simulation';

// Mock the API client
vi.mock('../../api/client', () => ({
  scenariosApi: {
    get: vi.fn(),
  },
  simulationsApi: {
    start: vi.fn(),
    getState: vi.fn(),
    makeDecision: vi.fn(),
    complete: vi.fn(),
    abandon: vi.fn(),
    challenge: vi.fn(),
    skipTime: vi.fn(),
    getStreamToken: vi.fn(),
  },
  streamApi: {
    simulationStream: vi.fn(),
  },
}));

// Mock static asset imports
vi.mock('../../../assests/logo.png', () => ({ default: 'logo.png' }));

// Mock recharts
vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }) => children,
  ComposedChart: ({ children }) => <div data-testid="price-chart">{children}</div>,
  Line: () => null,
  Area: () => null,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
  ReferenceLine: () => null,
  CartesianGrid: () => null,
}));

// Mock CoachNudge
vi.mock('../../components/CoachNudge', () => ({
  default: () => null,
}));

const { scenariosApi, simulationsApi } = await import('../../api/client');

const mockScenario = {
  id: 'scenario-1',
  name: 'The Next Big Thing',
  description: 'A hot tech IPO is surging. Do you buy the hype?',
  difficulty: 2,
  category: 'fomo_trap',
  time_pressure_seconds: 180,
  initial_data: {
    asset: 'HYPE',
    price: 50.0,
    your_balance: 10000,
    market_sentiment: 'bullish',
    market_params: {
      base_spread_pct: 0.002,
      fixed_fee: 0,
      pct_fee: 0,
    },
    news_headlines: [{ content: 'HYPE stock surging on IPO day' }],
    holdings: {},
  },
};

const mockSimState = {
  id: 'sim-123',
  scenario_id: 'scenario-1',
  scenario_name: 'The Next Big Thing',
  status: 'in_progress',
  time_elapsed: 0,
  time_remaining: 180,
  current_price: 52.0,
  price_history: [50.0, 51.0, 52.0],
  portfolio: { cash: 10000, holdings: {}, total_value: 10000, cumulative_fees: 0 },
  available_info: { news: [], social: [] },
  recent_events: [],
  market_conditions: { bid: 51.9, ask: 52.1, spread_pct: 0.004 },
};

function renderSimulation() {
  return render(
    <MemoryRouter initialEntries={['/simulation/scenario-1']}>
      <Routes>
        <Route path="/simulation/:scenarioId" element={<Simulation />} />
      </Routes>
    </MemoryRouter>
  );
}

describe('Simulation Page', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows loading state initially', () => {
    scenariosApi.get.mockReturnValue(new Promise(() => {})); // never resolves
    renderSimulation();
    expect(screen.getByText('Loading scenario...')).toBeInTheDocument();
  });

  it('shows error state when scenario fails to load', async () => {
    scenariosApi.get.mockRejectedValue({
      response: { data: { detail: 'Scenario not found' } },
    });
    renderSimulation();
    await waitFor(() => {
      expect(screen.getByText('Unable to Load')).toBeInTheDocument();
      expect(screen.getByText('Scenario not found')).toBeInTheDocument();
    });
  });

  it('renders briefing screen with scenario name and details', async () => {
    scenariosApi.get.mockResolvedValue({ data: mockScenario });
    renderSimulation();
    await waitFor(() => {
      expect(screen.getByText('The Next Big Thing')).toBeInTheDocument();
      expect(screen.getByText('Easy')).toBeInTheDocument(); // difficulty 2
      expect(screen.getByText('HYPE')).toBeInTheDocument();
      expect(screen.getByText('$50.00')).toBeInTheDocument();
      expect(screen.getByText('$10,000')).toBeInTheDocument();
      expect(screen.getByText('Begin Simulation')).toBeInTheDocument();
    });
  });

  it('shows market mood on briefing screen', async () => {
    scenariosApi.get.mockResolvedValue({ data: mockScenario });
    renderSimulation();
    await waitFor(() => {
      expect(screen.getByText(/bullish/i)).toBeInTheDocument();
    });
  });

  it('shows error buttons (Try Again / Return to Dashboard) on failure', async () => {
    scenariosApi.get.mockRejectedValue({
      response: { data: { detail: 'Network error' } },
    });
    renderSimulation();
    await waitFor(() => {
      expect(screen.getByText('Try Again')).toBeInTheDocument();
      expect(screen.getByText('Return to Dashboard')).toBeInTheDocument();
    });
  });

  it('renders active market features on briefing for advanced scenario', async () => {
    const advancedScenario = {
      ...mockScenario,
      difficulty: 5,
      initial_data: {
        ...mockScenario.initial_data,
        market_params: {
          ...mockScenario.initial_data.market_params,
          halts_enabled: true,
          order_types_enabled: true,
          volatility_clustering: true,
          crowd_model_enabled: true,
          margin_enabled: true,
          news_latency_enabled: true,
        },
      },
    };
    scenariosApi.get.mockResolvedValue({ data: advancedScenario });
    renderSimulation();
    await waitFor(() => {
      expect(screen.getByText('Active Market Features')).toBeInTheDocument();
      expect(screen.getByText('Circuit Breakers')).toBeInTheDocument();
      expect(screen.getByText('Order Types')).toBeInTheDocument();
      expect(screen.getByText('Volatility Clustering')).toBeInTheDocument();
      expect(screen.getByText('Crowd Behavior')).toBeInTheDocument();
      expect(screen.getByText('Margin Trading')).toBeInTheDocument();
      expect(screen.getByText('News Delays')).toBeInTheDocument();
    });
  });

  it('shows existing holdings on briefing screen', async () => {
    const scenarioWithHoldings = {
      ...mockScenario,
      initial_data: {
        ...mockScenario.initial_data,
        holdings: { HYPE: 100 },
      },
    };
    scenariosApi.get.mockResolvedValue({ data: scenarioWithHoldings });
    renderSimulation();
    await waitFor(() => {
      expect(screen.getByText(/100 shares of HYPE/)).toBeInTheDocument();
    });
  });
});
