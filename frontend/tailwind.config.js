/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          navy: '#3A3E61',
          'navy-light': '#4E5280',
          'navy-dark': '#2C3050',
          lavender: '#D1D3EB',
          'lavender-light': '#E4E5F3',
          blue: '#A4B9D8',
          'blue-light': '#C0D0E6',
          cream: '#F1EDE2',
          'cream-dark': '#E5DFD2',
        },
        primary: {
          50: '#E4E5F3',
          100: '#D1D3EB',
          200: '#B8BBD9',
          300: '#A4B9D8',
          400: '#7A80AE',
          500: '#4E5280',
          600: '#3A3E61',
          700: '#2C3050',
          800: '#1E2240',
          900: '#141730',
        },
        danger: {
          400: '#f87171',
          500: '#ef4444',
          600: '#dc2626',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      keyframes: {
        shrink: {
          '0%': { width: '100%' },
          '100%': { width: '0%' },
        },
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        scaleIn: {
          '0%': { opacity: '0', transform: 'scale(0.95)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
      },
      animation: {
        shrink: 'shrink linear forwards',
        'fade-in-up': 'fadeInUp 0.5s ease-out both',
        'scale-in': 'scaleIn 0.4s ease-out both',
      },
    },
  },
  plugins: [],
}
