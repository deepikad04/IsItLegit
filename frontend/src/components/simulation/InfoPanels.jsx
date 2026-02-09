import {
  Newspaper, Users, BarChart3, Globe, Clock, Twitter, Hash,
  Heart, Repeat2, Radio, Activity, Layers, ShieldCheck, ShieldAlert, ShieldQuestion
} from 'lucide-react';
import clsx from 'clsx';
import { getNewsSource, getSocialUser, getEngagement, relativeTime, isBreaking, hashStr } from './helpers';

const CredibilityBadge = ({ credibility }) => {
  if (credibility == null) return null;
  if (credibility >= 0.7) return (
    <span className="inline-flex items-center gap-1 ml-2 text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 flex-shrink-0">
      <ShieldCheck className="h-3 w-3" /> Reliable
    </span>
  );
  if (credibility >= 0.4) return (
    <span className="inline-flex items-center gap-1 ml-2 text-xs font-semibold px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 flex-shrink-0">
      <ShieldQuestion className="h-3 w-3" /> Uncertain
    </span>
  );
  return (
    <span className="inline-flex items-center gap-1 ml-2 text-xs font-semibold px-2 py-0.5 rounded-full bg-red-100 text-red-800 flex-shrink-0">
      <ShieldAlert className="h-3 w-3" /> Unreliable
    </span>
  );
};

const SentimentTag = ({ sentiment }) => {
  if (!sentiment || sentiment === 'event') return null;
  const colors = {
    bullish: 'text-emerald-800 bg-emerald-100',
    bearish: 'text-red-800 bg-red-100',
    neutral: 'text-gray-700 bg-gray-100',
    fearful: 'text-orange-800 bg-orange-100',
  };
  return (
    <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${colors[sentiment] || colors.neutral}`}>
      {sentiment}
    </span>
  );
};

export default function InfoPanels({ state, activePanel, onSwitchPanel, crowdSentiment, macro, timeElapsed }) {
  return (
    <div className="bg-white rounded-2xl border border-gray-200 shadow-md flex flex-col">
      {/* Tabs */}
      <div className="flex border-b border-gray-200">
        {[
          { id: 'news', icon: Newspaper, label: 'News' },
          { id: 'social', icon: Users, label: 'Social' },
          ...(macro ? [{ id: 'macro', icon: BarChart3, label: 'Macro' }] : []),
        ].map((tab) => (
          <button key={tab.id} onClick={() => onSwitchPanel(tab.id)}
            className={clsx(
              'flex-1 flex items-center justify-center gap-2 py-3.5 text-sm font-semibold border-b-2 transition-colors',
              activePanel === tab.id
                ? 'border-brand-navy text-brand-navy bg-gray-50'
                : 'border-transparent text-gray-400 hover:text-gray-700'
            )}
          >
            <tab.icon className="h-4 w-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Panel content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3 max-h-[260px] sm:max-h-[360px]">

        {/* NEWS */}
        {activePanel === 'news' && (
          state.available_info?.news?.map((item, i) => {
            const source = getNewsSource(item);
            const breaking = isBreaking(item.time, timeElapsed);
            const timeAgo = relativeTime(item.time, timeElapsed);
            return (
              <div key={i} className={clsx(
                'p-3.5 rounded-xl border',
                breaking ? 'bg-red-50 border-red-300' : 'bg-gray-50 border-gray-200'
              )}>
                <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                  <Globe className="h-4 w-4 text-blue-600 flex-shrink-0" />
                  <span className="text-xs font-bold text-blue-700 uppercase tracking-wide">{source}</span>
                  {breaking && (
                    <span className="text-xs font-black px-2 py-0.5 rounded-full bg-red-600 text-white uppercase animate-pulse">
                      Breaking
                    </span>
                  )}
                  {item.unverified && (
                    <span className="text-xs font-black px-2 py-0.5 rounded-full bg-orange-500 text-white uppercase">
                      Unverified
                    </span>
                  )}
                  <CredibilityBadge credibility={item.credibility} />
                </div>
                <p className={clsx(
                  'text-sm leading-relaxed',
                  breaking ? 'text-gray-900 font-bold' : 'text-gray-800 font-medium'
                )}>{item.content}</p>
                <div className="flex items-center gap-3 mt-2">
                  <span className="text-xs font-semibold text-gray-400">{timeAgo}</span>
                  {item.delayed && (
                    <span className="text-xs font-bold text-orange-600 flex items-center gap-1">
                      <Clock className="h-3 w-3" /> Delayed
                    </span>
                  )}
                </div>
              </div>
            );
          })
        )}

        {/* SOCIAL */}
        {activePanel === 'social' && (
          <>
            {crowdSentiment != null && (
              <div className="p-3.5 bg-gray-50 rounded-xl border border-gray-200">
                <div className="flex items-center gap-3">
                  <Users className="h-5 w-5 text-gray-500 flex-shrink-0" />
                  <div className="flex-1">
                    <p className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-1">Market Crowd</p>
                    <div className="flex items-center gap-3">
                      <div className="flex-1 bg-gray-200 rounded-full h-3 overflow-hidden">
                        <div
                          className={clsx(
                            'h-full rounded-full transition-all duration-700',
                            crowdSentiment > 0.6 ? 'bg-emerald-500' :
                            crowdSentiment < 0.4 ? 'bg-red-500' : 'bg-amber-500'
                          )}
                          style={{ width: `${crowdSentiment * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-black text-gray-900 w-20 text-right">
                        {Math.round(crowdSentiment * 100)}% buy
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}
            {state.available_info?.social?.map((item, i) => {
              const user = getSocialUser(item.content);
              const engagement = getEngagement(item.content, item.sentiment);
              const timeAgo = relativeTime(item.time, timeElapsed);
              const isPlatformX = hashStr(item.content) % 3 !== 0;
              return (
                <div key={i} className="p-3.5 bg-gray-50 rounded-xl border border-gray-200">
                  <div className="flex items-start gap-3">
                    <div className={clsx(
                      'w-9 h-9 rounded-full flex items-center justify-center text-white text-xs font-black flex-shrink-0',
                      user.color
                    )}>
                      {user.initials}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-sm font-bold text-gray-900">{user.username}</span>
                        {isPlatformX ? (
                          <Twitter className="h-3.5 w-3.5 text-gray-400" />
                        ) : (
                          <Hash className="h-3.5 w-3.5 text-orange-500" />
                        )}
                        <span className="text-xs text-gray-400">{timeAgo}</span>
                        {item.unverified && (
                          <span className="text-xs font-black px-2 py-0.5 rounded-full bg-orange-500 text-white uppercase">
                            Unverified
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-gray-800 font-medium leading-relaxed">{item.content}</p>
                      <div className="flex items-center gap-4 mt-2">
                        <span className="flex items-center gap-1 text-xs text-gray-400 hover:text-red-500 cursor-default">
                          <Heart className="h-3.5 w-3.5" /> {engagement.likes}
                        </span>
                        <span className="flex items-center gap-1 text-xs text-gray-400 hover:text-emerald-500 cursor-default">
                          <Repeat2 className="h-3.5 w-3.5" /> {engagement.reposts}
                        </span>
                        <SentimentTag sentiment={item.sentiment} />
                        <CredibilityBadge credibility={item.credibility} />
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </>
        )}

        {/* MACRO */}
        {activePanel === 'macro' && macro && (
          <div className="space-y-3">
            {macro.interest_rate_direction && (
              <div className="p-4 bg-gray-50 rounded-xl border border-gray-200">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5 text-gray-500" />
                    <span className="text-sm font-bold text-gray-800">Interest Rates</span>
                  </div>
                  <span className={clsx(
                    'text-sm font-black px-3 py-1 rounded-full',
                    macro.interest_rate_direction === 'up' ? 'bg-red-100 text-red-800' :
                    macro.interest_rate_direction === 'down' ? 'bg-emerald-100 text-emerald-800' :
                    'bg-gray-100 text-gray-700'
                  )}>
                    {macro.interest_rate_direction === 'up' ? 'Rising' :
                     macro.interest_rate_direction === 'down' ? 'Falling' : 'Flat'}
                  </span>
                </div>
              </div>
            )}
            {macro.market_breadth != null && (
              <div className="p-4 bg-gray-50 rounded-xl border border-gray-200">
                <div className="flex items-center gap-2 mb-3">
                  <Layers className="h-5 w-5 text-gray-500" />
                  <span className="text-sm font-bold text-gray-800">Market Breadth</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="flex-1 bg-gray-200 rounded-full h-3 overflow-hidden">
                    <div
                      className={clsx(
                        'h-full rounded-full',
                        macro.market_breadth > 0.6 ? 'bg-emerald-500' :
                        macro.market_breadth < 0.4 ? 'bg-red-500' : 'bg-amber-500'
                      )}
                      style={{ width: `${macro.market_breadth * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-black text-gray-900 w-20 text-right">
                    {Math.round(macro.market_breadth * 100)}% adv
                  </span>
                </div>
              </div>
            )}
            {macro.vix != null && (
              <div className="p-4 bg-gray-50 rounded-xl border border-gray-200">
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <Activity className="h-5 w-5 text-gray-500" />
                    <span className="text-sm font-bold text-gray-800">Volatility Index</span>
                  </div>
                  <span className={clsx(
                    'text-2xl font-black',
                    macro.vix < 15 ? 'text-emerald-600' :
                    macro.vix < 25 ? 'text-amber-600' : 'text-red-600'
                  )}>
                    {macro.vix.toFixed(1)}
                  </span>
                </div>
                <p className="text-xs font-medium text-gray-500">
                  {macro.vix < 15 ? 'Low fear — calm markets' :
                   macro.vix < 25 ? 'Moderate uncertainty' : 'High fear — elevated risk'}
                </p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Recent events */}
      {state.recent_events?.length > 0 && (
        <div className="p-4 border-t border-gray-200">
          <div className="flex items-center gap-2 mb-2">
            <Radio className="h-3.5 w-3.5 text-red-500 animate-pulse" />
            <p className="text-xs font-bold text-red-600 uppercase tracking-wider">Live</p>
          </div>
          {state.recent_events.slice(-2).map((event, i) => (
            <p key={i} className="text-sm text-amber-700 font-bold leading-relaxed">{event.content}</p>
          ))}
        </div>
      )}
    </div>
  );
}
