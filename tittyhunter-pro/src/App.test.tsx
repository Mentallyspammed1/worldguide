import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import App from './App';

describe('App shell', () => {
  it('renders the main navigation and switches between local tabs', () => {
    render(<App />);

    expect(screen.getByRole('heading', { name: 'pyrmsearch' })).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: /HUNT/ }).length).toBeGreaterThan(0);
    expect(screen.getByRole('button', { name: /INSTALL APP/ })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /FAVORITES/ }));
    expect(screen.getByRole('heading', { name: /SAVED FAVORITES/ })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /SCREENSHOTS/ }));
    expect(screen.getByRole('heading', { name: /SAVED SCREENSHOTS/ })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /RECENT/ }));
    expect(screen.getByRole('heading', { name: /RECENTLY WATCHED/ })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /COLLECTIONS/ }));
    expect(screen.getByRole('heading', { name: /COLLECTIONS/ })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /BACKUP/ }));
    expect(screen.getByRole('heading', { name: /DATA BACKUP/ })).toBeInTheDocument();
  });

  it('lets the user collapse and expand the system console', () => {
    render(<App />);

    fireEvent.click(screen.getByLabelText('Collapse system console'));
    expect(screen.getByLabelText('Expand system console')).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText('Expand system console'));
    expect(screen.getByLabelText('Collapse system console')).toBeInTheDocument();
  });
});
