import { Link } from 'react-router-dom';
import { Target, TrendingUp, Zap, Shield, BarChart3, Brain } from 'lucide-react';
import landingBg from '../../assests/LandingPage.png';

export default function Home() {
  const features = [
    {
      icon: Target,
      title: 'Practice Real Scenarios',
      description: 'Face high-pressure financial decisions in realistic simulations with time pressure and emotional triggers.'
    },
    {
      icon: Brain,
      title: 'AI-Powered Analysis',
      description: 'Get deep insights into your decision-making patterns, biases, and cognitive habits.'
    },
    {
      icon: BarChart3,
      title: 'Process Over Outcome',
      description: 'Learn that good decisions can have bad outcomes and vice versa. Focus on improving your process.'
    },
    {
      icon: Zap,
      title: 'Counterfactual Learning',
      description: 'See alternate timelines where the same decisions lead to different outcomes.'
    },
    {
      icon: Shield,
      title: 'Risk-Free Environment',
      description: 'Make mistakes and learn from them without losing real money.'
    },
    {
      icon: TrendingUp,
      title: 'Track Your Progress',
      description: 'Watch your decision-making skills improve over time with personalized insights.'
    }
  ];

  return (
    <div className="min-h-screen bg-brand-cream" role="main">
      {/* Skip to content link for keyboard users */}
      <a href="#features" className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-brand-navy focus:text-white focus:rounded-lg">
        Skip to content
      </a>

      {/* Hero Section — full LandingPage.png background */}
      <header className="relative overflow-hidden min-h-[85vh] flex items-end" aria-label="Hero">
        <img
          src={landingBg}
          alt=""
          aria-hidden="true"
          className="absolute inset-0 w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-brand-cream via-transparent to-transparent" aria-hidden="true" />
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 pb-16 pt-32 relative w-full">
          <div className="text-center">
            <p className="text-lg md:text-xl text-brand-navy/80 max-w-2xl mx-auto mb-8 font-medium">
              A simulation-based decision training platform that teaches you how to make better financial decisions through experience, not advice.
            </p>
            <nav className="flex flex-col sm:flex-row items-center justify-center gap-4" aria-label="Get started">
              <Link to="/register" className="btn btn-primary text-lg px-10 py-3.5">
                Start Training Free
              </Link>
              <Link to="/login" className="btn btn-secondary text-lg px-10 py-3.5">
                Sign In
              </Link>
            </nav>
          </div>
        </div>
      </header>

      {/* What Makes This Different */}
      <section id="features" className="py-20 bg-white/50" aria-labelledby="features-heading">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-14">
            <h2 id="features-heading" className="text-3xl font-bold text-brand-navy mb-4">
              This Is Not Another Finance App
            </h2>
            <p className="text-brand-navy/60 max-w-2xl mx-auto text-lg">
              We don't teach you what to invest in. We teach you how to think when making decisions under uncertainty.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6" role="list">
            {features.map((feature, index) => (
              <article key={index} className="card hover:shadow-md hover:border-brand-blue/50 transition-all" role="listitem">
                <div className="w-12 h-12 rounded-xl bg-brand-lavender flex items-center justify-center mb-4">
                  <feature.icon className="h-6 w-6 text-brand-navy" aria-hidden="true" />
                </div>
                <h3 className="text-lg font-semibold text-brand-navy mb-2">{feature.title}</h3>
                <p className="text-brand-navy/60 text-sm leading-relaxed">{feature.description}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-20" aria-labelledby="how-it-works-heading">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-14">
            <h2 id="how-it-works-heading" className="text-3xl font-bold text-brand-navy mb-4">How It Works</h2>
          </div>

          <ol className="grid grid-cols-1 md:grid-cols-3 gap-10" role="list">
            <li className="text-center">
              <div className="w-14 h-14 rounded-2xl bg-brand-navy text-white text-xl font-bold flex items-center justify-center mx-auto mb-5" aria-hidden="true">
                1
              </div>
              <h3 className="text-xl font-semibold text-brand-navy mb-2">Enter a Simulation</h3>
              <p className="text-brand-navy/60">
                Face realistic market scenarios with time pressure, news events, and social signals.
              </p>
            </li>

            <li className="text-center">
              <div className="w-14 h-14 rounded-2xl bg-brand-navy text-white text-xl font-bold flex items-center justify-center mx-auto mb-5" aria-hidden="true">
                2
              </div>
              <h3 className="text-xl font-semibold text-brand-navy mb-2">Make Decisions</h3>
              <p className="text-brand-navy/60">
                Buy, sell, or hold. Every choice is logged — what you saw, what you ignored, how fast you acted.
              </p>
            </li>

            <li className="text-center">
              <div className="w-14 h-14 rounded-2xl bg-brand-navy text-white text-xl font-bold flex items-center justify-center mx-auto mb-5" aria-hidden="true">
                3
              </div>
              <h3 className="text-xl font-semibold text-brand-navy mb-2">Get AI Analysis</h3>
              <p className="text-brand-navy/60">
                Discover your patterns, see alternate timelines, and learn what to improve.
              </p>
            </li>
          </ol>
        </div>
      </section>

      {/* CTA */}
      <section className="bg-brand-navy py-20 rounded-t-3xl" aria-labelledby="cta-heading">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <h2 id="cta-heading" className="text-3xl font-bold text-white mb-4">
            Ready to Improve Your Decision Making?
          </h2>
          <p className="text-brand-lavender mb-10 text-lg">
            Start with free simulations. No credit card required.
          </p>
          <Link to="/register" className="btn bg-white text-brand-navy hover:bg-brand-cream text-lg px-10 py-3.5 shadow-lg">
            Get Started Now
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-brand-navy-dark py-8" role="contentinfo">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <p className="text-brand-lavender font-medium">IsItLegit - Decision Training Platform</p>
          <p className="text-brand-blue text-sm mt-2">Learn from experience, not advice.</p>
        </div>
      </footer>
    </div>
  );
}
