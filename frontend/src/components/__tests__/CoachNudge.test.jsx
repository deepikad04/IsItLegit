import { render, screen, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import CoachNudge from '../CoachNudge';

describe('CoachNudge', () => {
  it('renders nothing when nudge is null', () => {
    const { container } = render(<CoachNudge nudge={null} onDismiss={() => {}} />);
    expect(container.innerHTML).toBe('');
  });

  it('renders the nudge message', () => {
    const nudge = { bias: 'fomo', message: 'Are you sure? This looks like FOMO.' };
    render(<CoachNudge nudge={nudge} onDismiss={() => {}} />);
    expect(screen.getByText('Are you sure? This looks like FOMO.')).toBeInTheDocument();
  });

  it('displays the correct bias label', () => {
    const nudge = { bias: 'loss_aversion', message: 'Consider the fundamentals.' };
    render(<CoachNudge nudge={nudge} onDismiss={() => {}} />);
    expect(screen.getByText(/Loss Aversion/)).toBeInTheDocument();
  });

  it('calls onDismiss when close button is clicked', async () => {
    const onDismiss = vi.fn();
    const nudge = { bias: 'fomo', message: 'Watch out for FOMO.' };
    render(<CoachNudge nudge={nudge} onDismiss={onDismiss} />);

    const closeButton = screen.getByRole('button');
    await userEvent.click(closeButton);
    expect(onDismiss).toHaveBeenCalled();
  });

  it('falls back to default styling for unknown bias types', () => {
    const nudge = { bias: 'unknown_bias', message: 'Some coaching message.' };
    render(<CoachNudge nudge={nudge} onDismiss={() => {}} />);
    expect(screen.getByText('Some coaching message.')).toBeInTheDocument();
    expect(screen.getByText(/unknown_bias/)).toBeInTheDocument();
  });
});
