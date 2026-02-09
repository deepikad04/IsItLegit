import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceDot, Legend,
} from 'recharts';
import clsx from 'clsx';

function DecisionDot({ cx, cy, payload, type }) {
  if (!cx || !cy) return null;
  const colors = {
    user_buy: '#4ade80',   // green
    user_sell: '#f87171',  // red
    user_hold: '#facc15',  // yellow
    pro: '#22d3ee',        // cyan
  };
  const color = colors[type] || '#a78bfa';
  const size = type === 'pro' ? 5 : 7;

  return (
    <circle cx={cx} cy={cy} r={size} fill={color} stroke="#F1EDE2" strokeWidth={2} />
  );
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;

  return (
    <div className="bg-brand-cream border border-brand-blue/30 rounded-lg px-3 py-2 text-sm shadow-xl">
      <p className="text-brand-navy/60">t={d.time}s</p>
      <p className="text-brand-navy font-medium">${d.price?.toFixed(2)}</p>
      {d.userAction && (
        <p className="text-green-700 mt-1">You: {d.userAction}</p>
      )}
      {d.proAction && (
        <p className="text-cyan-700 mt-1">Pro: {d.proAction}</p>
      )}
    </div>
  );
}

export default function ProReplayChart({ priceHistory, userDecisions, proDecisions }) {
  if (!priceHistory?.length) return null;

  // Build a map of user decisions by timestamp
  const userMap = {};
  (userDecisions || []).forEach((d) => {
    userMap[d.at_timestamp] = d.user_action;
  });

  // Build a map of pro decisions by timestamp
  const proMap = {};
  (proDecisions || []).forEach((d) => {
    proMap[d.at_timestamp] = d.pro_action;
  });

  // Merge into chart data
  const data = priceHistory.map((p) => ({
    time: p.time,
    price: p.price,
    userAction: userMap[p.time] || null,
    proAction: proMap[p.time] || null,
  }));

  // Find nearest data points for decisions that don't land exactly on a tick
  const addNearestDecision = (map, key) => {
    Object.keys(map).forEach((ts) => {
      const t = Number(ts);
      const exact = data.find((d) => d.time === t);
      if (exact) {
        exact[key] = exact[key] || map[ts];
      } else {
        // find closest
        const closest = data.reduce((a, b) =>
          Math.abs(b.time - t) < Math.abs(a.time - t) ? b : a
        );
        if (closest) closest[key] = closest[key] || map[ts];
      }
    });
  };
  addNearestDecision(userMap, 'userAction');
  addNearestDecision(proMap, 'proAction');

  return (
    <div className="mt-4 p-4 bg-gray-50 rounded-xl border border-gray-200">
      <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3">Price Chart with Decision Points</p>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={data} margin={{ top: 20, right: 20, bottom: 5, left: 10 }}>
          <XAxis
            dataKey="time"
            tick={{ fill: '#6b7280', fontSize: 11, fontWeight: 500 }}
            tickFormatter={(t) => `${t}s`}
            axisLine={{ stroke: '#d1d5db' }}
          />
          <YAxis
            tick={{ fill: '#6b7280', fontSize: 11, fontWeight: 500 }}
            tickFormatter={(v) => `$${v.toFixed(0)}`}
            axisLine={{ stroke: '#d1d5db' }}
            domain={['auto', 'auto']}
          />
          <Tooltip content={<CustomTooltip />} />
          <Line
            type="monotone"
            dataKey="price"
            stroke="#9ca3af"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 5, fill: '#3A3E61', stroke: 'white', strokeWidth: 2 }}
          />
          {/* User decision markers */}
          {data.filter((d) => d.userAction).map((d, i) => (
            <ReferenceDot
              key={`user-${i}`}
              x={d.time}
              y={d.price}
              r={8}
              fill={d.userAction?.toLowerCase().includes('sell') ? '#ef4444' : '#22c55e'}
              stroke="white"
              strokeWidth={2.5}
            />
          ))}
          {/* Pro decision markers */}
          {data.filter((d) => d.proAction).map((d, i) => (
            <ReferenceDot
              key={`pro-${i}`}
              x={d.time}
              y={d.price}
              r={6}
              fill="#06b6d4"
              stroke="white"
              strokeWidth={2}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mt-3 pt-3 border-t border-gray-200">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-gray-500">
          <span className="w-3.5 h-3.5 rounded-full bg-green-500 inline-block shadow-sm" />
          <span>Your Buy</span>
        </div>
        <div className="flex items-center gap-1.5 text-xs font-semibold text-gray-500">
          <span className="w-3.5 h-3.5 rounded-full bg-red-500 inline-block shadow-sm" />
          <span>Your Sell</span>
        </div>
        <div className="flex items-center gap-1.5 text-xs font-semibold text-gray-500">
          <span className="w-3.5 h-3.5 rounded-full bg-cyan-500 inline-block shadow-sm" />
          <span>Pro Move</span>
        </div>
        <div className="flex items-center gap-1.5 text-xs font-semibold text-gray-500">
          <span className="w-6 h-0.5 bg-gray-400 inline-block rounded-full" />
          <span>Price</span>
        </div>
      </div>
    </div>
  );
}
