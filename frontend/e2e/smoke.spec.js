import { test, expect } from '@playwright/test';

test.describe('IsItLegit — Smoke Tests', () => {
  test('home page loads with branding and CTA', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/IsItLegit/i);
    await expect(page.locator('text=Get Started')).toBeVisible();
  });

  test('login page renders form fields', async ({ page }) => {
    await page.goto('/login');
    await expect(page.locator('input[type="email"], input[name="email"]')).toBeVisible();
    await expect(page.locator('input[type="password"]')).toBeVisible();
  });

  test('register page renders form fields', async ({ page }) => {
    await page.goto('/register');
    await expect(page.locator('input[type="email"], input[name="email"]')).toBeVisible();
    await expect(page.locator('input[type="password"]')).toBeVisible();
  });

  test('unauthenticated user is redirected from dashboard', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForURL(/\/(login)?$/);
    const url = page.url();
    expect(url.includes('/login') || url.endsWith('/')).toBeTruthy();
  });

  test('navigation between public pages works', async ({ page }) => {
    await page.goto('/');
    await page.click('text=Get Started');
    await page.waitForURL(/\/(register|login)/);
    expect(page.url()).toMatch(/\/(register|login)/);
  });

  test('app loads without console errors', async ({ page }) => {
    const errors = [];
    page.on('pageerror', (err) => errors.push(err.message));
    await page.goto('/');
    await page.waitForTimeout(1000);
    // Filter out expected third-party errors
    const critical = errors.filter(e => !e.includes('ResizeObserver'));
    expect(critical).toHaveLength(0);
  });
});
