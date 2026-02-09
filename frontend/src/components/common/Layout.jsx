import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import {
  LayoutDashboard, BookOpen, User, LogOut, Menu, X
} from 'lucide-react';
import { useState } from 'react';
import clsx from 'clsx';
import logo from '../../../assests/logo.png';
import bgImage from '../../../assests/bg.png';

const navItems = [
  { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/learning', label: 'Learning', icon: BookOpen },
  { path: '/profile', label: 'Profile', icon: User },
];

export default function Layout({ children }) {
  const { user, logout } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  return (
    <div className="min-h-screen" style={{ backgroundImage: `url(${bgImage})`, backgroundSize: 'cover', backgroundPosition: 'center', backgroundAttachment: 'fixed' }}>
      {/* Desktop Sidebar */}
      <aside className="hidden lg:flex lg:flex-col lg:w-64 lg:fixed lg:inset-y-0 bg-brand-navy shadow-xl">
        <div className="flex items-center space-x-3 p-6">
          <img src={logo} alt="IsItLegit" className="h-9 w-9" />
          <span className="text-xl font-bold text-white">IsItLegit</span>
        </div>

        <nav className="flex-1 px-4 space-y-1">
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={clsx(
                'flex items-center space-x-3 px-4 py-3 rounded-xl transition-all',
                location.pathname === item.path
                  ? 'bg-white/20 text-white font-semibold'
                  : 'text-brand-lavender hover:bg-white/10 hover:text-white'
              )}
            >
              <item.icon className="h-5 w-5" />
              <span>{item.label}</span>
            </Link>
          ))}
        </nav>

        <div className="p-4 border-t border-white/10">
          <div className="flex items-center space-x-3 px-4 py-2 mb-2">
            <div className="w-9 h-9 rounded-xl bg-brand-lavender flex items-center justify-center">
              <span className="text-brand-navy font-bold">
                {user?.username?.[0]?.toUpperCase() || 'U'}
              </span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-white font-medium truncate">{user?.username}</p>
              <p className="text-brand-blue text-sm truncate">{user?.email}</p>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="flex items-center space-x-3 px-4 py-3 w-full rounded-xl text-brand-lavender hover:bg-white/10 hover:text-white transition-all"
          >
            <LogOut className="h-5 w-5" />
            <span>Sign Out</span>
          </button>
        </div>
      </aside>

      {/* Mobile Header */}
      <header className="lg:hidden fixed top-0 inset-x-0 bg-brand-navy shadow-lg z-50">
        <div className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center space-x-3">
            <img src={logo} alt="IsItLegit" className="h-8 w-8" />
            <span className="text-xl font-bold text-white">IsItLegit</span>
          </div>
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="text-brand-lavender hover:text-white"
          >
            {mobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
          </button>
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <nav className="px-4 py-2 border-t border-white/10">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                onClick={() => setMobileMenuOpen(false)}
                className={clsx(
                  'flex items-center space-x-3 px-4 py-3 rounded-xl transition-all',
                  location.pathname === item.path
                    ? 'bg-white/20 text-white font-semibold'
                    : 'text-brand-lavender hover:bg-white/10 hover:text-white'
                )}
              >
                <item.icon className="h-5 w-5" />
                <span>{item.label}</span>
              </Link>
            ))}
            <button
              onClick={handleLogout}
              className="flex items-center space-x-3 px-4 py-3 w-full rounded-xl text-brand-lavender hover:bg-white/10 hover:text-white transition-all"
            >
              <LogOut className="h-5 w-5" />
              <span>Sign Out</span>
            </button>
          </nav>
        )}
      </header>

      {/* Main Content */}
      <main className="lg:pl-64 pt-16 lg:pt-0">
        <div className="p-6">
          {children}
        </div>

        {/* Disclaimer Footer */}
        <footer className="border-t border-brand-blue/20 px-6 py-4 mt-8">
          <p className="text-xs text-brand-navy/40 text-center max-w-2xl mx-auto leading-relaxed">
            IsItLegit is an educational simulation tool designed to teach behavioral finance concepts.
            It does not provide real financial advice. All market data, prices, and scenarios are
            simulated and do not reflect actual market conditions. Never make real investment decisions
            based on this application. Always consult a licensed financial advisor.
          </p>
        </footer>
      </main>
    </div>
  );
}
