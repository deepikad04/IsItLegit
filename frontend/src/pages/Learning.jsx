import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { learningApi, profileApi } from '../api/client';
import {
  BookOpen, ChevronLeft, ChevronRight, CheckCircle, XCircle,
  Brain, Target, Calculator, Newspaper, Award, Lock, ArrowRight, Sparkles
} from 'lucide-react';
import clsx from 'clsx';

// Map user bias patterns to relevant learning module categories
const BIAS_TO_MODULE = {
  fomo: 'emotional',
  loss_aversion: 'emotional',
  impulsivity: 'emotional',
  overconfidence: 'confidence',
  anchoring: 'anchoring',
  social_proof_reliance: 'social',
  herd_following: 'social',
  recency_bias: 'technical',
  patience: 'patience',
};

const MODULE_ICONS = {
  brain: Brain,
  calculator: Calculator,
  newspaper: Newspaper,
  target: Target,
};

const MODULE_COLOR = {
  bg: 'from-brand-navy to-brand-navy-light',
  card: 'border-brand-blue/30 bg-brand-cream-dark',
  badge: 'bg-brand-lavender text-brand-navy',
};

// ── Module List View ─────────────────────────────────────────────────

function ModuleCard({ module, onSelect, isRecommended }) {
  const Icon = MODULE_ICONS[module.icon] || BookOpen;
  const colors = MODULE_COLOR;

  return (
    <button
      onClick={() => onSelect(module)}
      className={clsx(
        'card text-left w-full border transition-all hover:scale-[1.02] hover:shadow-lg',
        isRecommended && !module.completed ? 'border-brand-navy/30 ring-1 ring-brand-navy/10 shadow-md' : colors.card
      )}
    >
      <div className="flex items-start space-x-4">
        <div className={clsx(
          'w-14 h-14 rounded-xl bg-gradient-to-br flex items-center justify-center flex-shrink-0',
          colors.bg
        )}>
          <Icon className="h-7 w-7 text-white" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between mb-1">
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-semibold text-brand-navy">{module.title}</h3>
              {isRecommended && !module.completed && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-brand-lavender text-brand-navy text-xs font-bold">
                  <Sparkles className="h-3 w-3" /> For You
                </span>
              )}
            </div>
            {module.completed ? (
              <span className="flex items-center space-x-1 text-green-700 text-sm">
                <CheckCircle className="h-4 w-4" />
                <span>Done</span>
              </span>
            ) : (
              <ArrowRight className="h-5 w-5 text-brand-blue" />
            )}
          </div>
          <p className="text-brand-navy/60 text-sm mb-3">{module.description}</p>
          <div className="flex items-center space-x-4 text-xs text-brand-blue">
            <span>{module.lesson_count} lessons</span>
            <span>{module.quiz_count} quiz questions</span>
          </div>
        </div>
      </div>
    </button>
  );
}

// ── Lesson View ──────────────────────────────────────────────────────

function LessonView({ module, lessonIndex, onNext, onBack }) {
  const lesson = module.lessons[lessonIndex];
  const colors = MODULE_COLOR;
  const isLast = lessonIndex === module.lessons.length - 1;

  return (
    <div className="max-w-3xl mx-auto">
      {/* Progress */}
      <div className="mb-6">
        <div className="flex items-center justify-between text-sm text-brand-navy/60 mb-2">
          <button onClick={onBack} className="flex items-center space-x-1 hover:text-brand-navy transition-colors">
            <ChevronLeft className="h-4 w-4" />
            <span>{lessonIndex === 0 ? 'Back to Modules' : 'Previous'}</span>
          </button>
          <span>Lesson {lessonIndex + 1} of {module.lessons.length}</span>
        </div>
        <div className="h-1.5 bg-brand-blue/20 rounded-full overflow-hidden">
          <div
            className={clsx('h-full bg-gradient-to-r rounded-full transition-all duration-500', colors.bg)}
            style={{ width: `${((lessonIndex + 1) / (module.lessons.length + 1)) * 100}%` }}
          />
        </div>
      </div>

      {/* Lesson Content */}
      <div className="card">
        <div className={clsx(
          'px-6 py-4 rounded-t-xl bg-gradient-to-r text-white -mx-6 -mt-6 mb-6',
          colors.bg
        )}>
          <p className="text-white/60 text-sm">{module.title}</p>
          <h2 className="text-2xl font-bold text-white">{lesson.title}</h2>
        </div>

        <div className="prose prose-invert max-w-none">
          {lesson.content.split('\n\n').map((para, i) => (
            <p key={i} className="text-brand-navy/70 leading-relaxed mb-4 last:mb-0">
              {para}
            </p>
          ))}
        </div>

        {lesson.key_insight && (
          <div className={clsx('mt-6 p-4 rounded-lg border', colors.card)}>
            <p className="text-sm text-brand-navy/60 mb-1">Key Insight</p>
            <p className="text-brand-navy font-medium">{lesson.key_insight}</p>
          </div>
        )}

        <div className="mt-8 flex justify-end">
          <button
            onClick={onNext}
            className="btn btn-primary flex items-center space-x-2"
          >
            <span>{isLast ? 'Start Quiz' : 'Next Lesson'}</span>
            <ChevronRight className="h-5 w-5" />
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Quiz View ────────────────────────────────────────────────────────

function QuizView({ module, onComplete, onBack }) {
  const [currentQ, setCurrentQ] = useState(0);
  const [selected, setSelected] = useState(null);
  const [answered, setAnswered] = useState(false);
  const [score, setScore] = useState(0);
  const [finished, setFinished] = useState(false);

  const colors = MODULE_COLOR;
  const quiz = module.quiz;
  const question = quiz[currentQ];

  const handleSelect = (optionIndex) => {
    if (answered) return;
    setSelected(optionIndex);
    setAnswered(true);
    if (optionIndex === question.correct) {
      setScore(score + 1);
    }
  };

  const handleNext = () => {
    if (currentQ < quiz.length - 1) {
      setCurrentQ(currentQ + 1);
      setSelected(null);
      setAnswered(false);
    } else {
      setFinished(true);
      onComplete(score + (selected === question.correct ? 0 : 0)); // score already updated
    }
  };

  if (finished) {
    const pct = Math.round((score / quiz.length) * 100);
    const passed = pct >= 67; // 2/3 correct

    return (
      <div className="max-w-xl mx-auto text-center">
        <div className="card py-12">
          <div className={clsx(
            'w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-4',
            passed ? 'bg-green-100' : 'bg-yellow-100'
          )}>
            {passed ? (
              <Award className="h-10 w-10 text-green-700" />
            ) : (
              <Target className="h-10 w-10 text-amber-600" />
            )}
          </div>
          <h2 className="text-2xl font-bold text-brand-navy mb-2">
            {passed ? 'Module Complete!' : 'Almost There!'}
          </h2>
          <p className="text-brand-navy/60 mb-2">
            You scored {score}/{quiz.length} ({pct}%)
          </p>
          <p className="text-brand-blue text-sm mb-6">
            {passed
              ? 'Great understanding of the material. These concepts will help you in your next simulation.'
              : 'Review the lessons and try again to solidify your understanding.'}
          </p>
          <div className="flex justify-center space-x-4">
            {!passed && (
              <button
                onClick={() => {
                  setCurrentQ(0);
                  setSelected(null);
                  setAnswered(false);
                  setScore(0);
                  setFinished(false);
                }}
                className="btn btn-secondary"
              >
                Retry Quiz
              </button>
            )}
            <button
              onClick={onBack}
              className="btn btn-primary"
            >
              Back to Modules
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto">
      {/* Progress */}
      <div className="mb-6">
        <div className="flex items-center justify-between text-sm text-brand-navy/60 mb-2">
          <span className="flex items-center space-x-1">
            <Award className="h-4 w-4" />
            <span>Quiz: {module.title}</span>
          </span>
          <span>Question {currentQ + 1} of {quiz.length}</span>
        </div>
        <div className="h-1.5 bg-brand-blue/20 rounded-full overflow-hidden">
          <div
            className={clsx('h-full bg-gradient-to-r rounded-full transition-all duration-500', colors.bg)}
            style={{ width: `${((module.lessons.length + currentQ + 1) / (module.lessons.length + quiz.length)) * 100}%` }}
          />
        </div>
      </div>

      <div className="card">
        <h3 className="text-xl font-semibold text-brand-navy mb-6">
          {question.question}
        </h3>

        <div className="space-y-3 mb-6">
          {question.options.map((option, i) => {
            let optionStyle = 'border-brand-blue/20 hover:border-brand-blue/40 hover:bg-brand-lavender/30';
            if (answered) {
              if (i === question.correct) {
                optionStyle = 'border-green-500 bg-green-500/10';
              } else if (i === selected && i !== question.correct) {
                optionStyle = 'border-red-500 bg-red-500/10';
              } else {
                optionStyle = 'border-brand-blue/20 opacity-50';
              }
            } else if (i === selected) {
              optionStyle = 'border-brand-navy bg-brand-navy/10';
            }

            return (
              <button
                key={i}
                onClick={() => handleSelect(i)}
                disabled={answered}
                className={clsx(
                  'w-full text-left p-4 rounded-lg border transition-all flex items-start space-x-3',
                  optionStyle
                )}
              >
                <span className={clsx(
                  'w-7 h-7 rounded-full border-2 flex items-center justify-center flex-shrink-0 text-sm font-medium',
                  answered && i === question.correct ? 'border-green-500 text-green-700' :
                  answered && i === selected ? 'border-red-500 text-red-600' :
                  'border-brand-blue/40 text-brand-navy/60'
                )}>
                  {answered && i === question.correct ? (
                    <CheckCircle className="h-5 w-5" />
                  ) : answered && i === selected && i !== question.correct ? (
                    <XCircle className="h-5 w-5" />
                  ) : (
                    String.fromCharCode(65 + i)
                  )}
                </span>
                <span className="text-brand-navy/80">{option}</span>
              </button>
            );
          })}
        </div>

        {/* Explanation */}
        {answered && (
          <div className={clsx(
            'p-4 rounded-lg border mb-6',
            selected === question.correct
              ? 'border-green-500/30 bg-green-500/5'
              : 'border-red-500/30 bg-red-500/5'
          )}>
            <p className={clsx(
              'text-sm font-medium mb-1',
              selected === question.correct ? 'text-green-700' : 'text-red-600'
            )}>
              {selected === question.correct ? 'Correct!' : 'Not quite.'}
            </p>
            <p className="text-brand-navy/70 text-sm">{question.explanation}</p>
          </div>
        )}

        {answered && (
          <div className="flex justify-end">
            <button onClick={handleNext} className="btn btn-primary flex items-center space-x-2">
              <span>{currentQ < quiz.length - 1 ? 'Next Question' : 'See Results'}</span>
              <ChevronRight className="h-5 w-5" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main Learning Page ───────────────────────────────────────────────

export default function Learning() {
  const navigate = useNavigate();
  const [modules, setModules] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeModule, setActiveModule] = useState(null);
  const [view, setView] = useState('list'); // 'list', 'lesson', 'quiz'
  const [lessonIndex, setLessonIndex] = useState(0);
  const [recommendedCategories, setRecommendedCategories] = useState([]);
  const [topWeakness, setTopWeakness] = useState(null);

  useEffect(() => {
    loadModules();
  }, []);

  const loadModules = async () => {
    try {
      const [modulesRes, profileRes] = await Promise.all([
        learningApi.getModules(),
        profileApi.get().catch(() => ({ data: null })),
      ]);
      setModules(modulesRes.data);

      // Derive recommended module categories from user's weakest biases
      if (profileRes.data?.bias_patterns) {
        const weakBiases = profileRes.data.bias_patterns
          .filter(p => p.score >= 0.4)
          .sort((a, b) => b.score - a.score);

        if (weakBiases.length > 0) {
          setTopWeakness(weakBiases[0].name.replace(/_/g, ' '));
          const cats = [...new Set(weakBiases.map(b => BIAS_TO_MODULE[b.name]).filter(Boolean))];
          setRecommendedCategories(cats);
        }
      }
    } catch (err) {
      console.error('Failed to load modules:', err);
    } finally {
      setLoading(false);
    }
  };

  const openModule = (mod) => {
    setActiveModule(mod);
    setLessonIndex(0);
    setView('lesson');
  };

  const handleLessonNext = () => {
    if (lessonIndex < activeModule.lessons.length - 1) {
      setLessonIndex(lessonIndex + 1);
    } else {
      setView('quiz');
    }
  };

  const handleLessonBack = () => {
    if (lessonIndex > 0) {
      setLessonIndex(lessonIndex - 1);
    } else {
      setView('list');
      setActiveModule(null);
    }
  };

  const handleQuizComplete = async (score) => {
    const pct = Math.round((score / activeModule.quiz.length) * 100);
    if (pct >= 67) {
      try {
        await learningApi.completeModule(activeModule.id);
        // Update local state
        setModules(prev => prev.map(m =>
          m.id === activeModule.id ? { ...m, completed: true } : m
        ));
      } catch (err) {
        console.error('Failed to record completion:', err);
      }
    }
  };

  const handleBackToList = () => {
    setView('list');
    setActiveModule(null);
    setLessonIndex(0);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="text-center">
          <BookOpen className="h-16 w-16 text-brand-navy mx-auto mb-4 animate-pulse" />
          <p className="text-brand-navy text-lg">Loading learning modules...</p>
        </div>
      </div>
    );
  }

  if (view === 'lesson' && activeModule) {
    return (
      <LessonView
        module={activeModule}
        lessonIndex={lessonIndex}
        onNext={handleLessonNext}
        onBack={handleLessonBack}
      />
    );
  }

  if (view === 'quiz' && activeModule) {
    return (
      <QuizView
        module={activeModule}
        onComplete={handleQuizComplete}
        onBack={handleBackToList}
      />
    );
  }

  // Module list view
  const completedCount = modules.filter(m => m.completed).length;

  return (
    <div className="max-w-3xl mx-auto">
      {/* Header */}
      <div className="card mb-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-14 h-14 rounded-xl bg-brand-lavender flex items-center justify-center">
              <BookOpen className="h-7 w-7 text-brand-navy" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-brand-navy">Learning Center</h1>
              <p className="text-brand-navy/60">
                Master the psychology of decision-making
              </p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-3xl font-bold text-brand-navy">{completedCount}/{modules.length}</p>
            <p className="text-brand-blue text-sm">modules done</p>
          </div>
        </div>

        {/* Overall progress bar */}
        {modules.length > 0 && (
          <div className="mt-4">
            <div className="h-2 bg-brand-blue/20 rounded-full overflow-hidden">
              <div
                className="h-full bg-brand-navy transition-all duration-500 rounded-full"
                style={{ width: `${(completedCount / modules.length) * 100}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Recommended Banner */}
      {recommendedCategories.length > 0 && (
        <div className="card bg-gradient-to-r from-brand-lavender/50 to-brand-cream border-brand-navy/20 mb-4">
          <div className="flex items-center gap-3">
            <Sparkles className="h-5 w-5 text-brand-navy flex-shrink-0" />
            <div>
              <p className="text-sm font-semibold text-brand-navy">Personalized for You</p>
              <p className="text-xs text-brand-navy/60">
                Based on your detected {topWeakness} patterns, we recommend the highlighted modules below.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Module Cards */}
      {modules.length > 0 ? (
        <div className="space-y-4">
          {[...modules]
            .sort((a, b) => {
              const aRec = recommendedCategories.includes(a.category) && !a.completed ? -1 : 0;
              const bRec = recommendedCategories.includes(b.category) && !b.completed ? -1 : 0;
              return aRec - bRec;
            })
            .map((mod) => (
              <ModuleCard
                key={mod.id}
                module={mod}
                onSelect={openModule}
                isRecommended={recommendedCategories.includes(mod.category)}
              />
            ))}
        </div>
      ) : (
        <div className="card text-center py-12">
          <BookOpen className="h-16 w-16 text-brand-blue mx-auto mb-4" />
          <h2 className="text-xl font-bold text-brand-navy mb-2">No Modules Available</h2>
          <p className="text-brand-navy/60 mb-6">
            Complete some simulations first, and learning modules will unlock based on your patterns.
          </p>
          <button onClick={() => navigate('/dashboard')} className="btn btn-primary">
            Start a Simulation
          </button>
        </div>
      )}
    </div>
  );
}
