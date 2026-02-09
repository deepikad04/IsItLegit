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
    <div className="mt-4">
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data} margin={{ top: 20, right: 20, bottom: 5, left: 10 }}>
          <XAxis
            dataKey="time"
            tick={{ fill: '#3A3E61', fontSize: 11 }}
            tickFormatter={(t) => `${t}s`}
            axisLine={{ stroke: '#A4B9D8' }}
          />
          <YAxis
            tick={{ fill: '#3A3E61', fontSize: 11 }}
            tickFormatter={(v) => `$${v.toFixed(0)}`}
            axisLine={{ stroke: '#A4B9D8' }}
            domain={['auto', 'auto']}
          />
          <Tooltip content={<CustomTooltip />} />
          <Line
            type="monotone"
            dataKey="price"
            stroke="#A4B9D8"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: '#3A3E61' }}
          />
          {/* User decision markers */}
          {data.filter((d) => d.userAction).map((d, i) => (
            <ReferenceDot
              key={`user-${i}`}
              x={d.time}
              y={d.price}
              r={7}
              fill={d.userAction?.toLowerCase().includes('sell') ? '#f87171' : '#4ade80'}
              stroke="#F1EDE2"
              strokeWidth={2}
            />
          ))}
          {/* Pro decision markers */}
          {data.filter((d) => d.proAction).map((d, i) => (
            <ReferenceDot
              key={`pro-${i}`}
              x={d.time}
              y={d.price}
              r={5}
              fill="#22d3ee"
              stroke="#F1EDE2"
              strokeWidth={2}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mt-2 text-xs text-brand-navy/60">
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full bg-green-500 inline-block" />
          <span>Your Buy</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full bg-red-500 inline-block" />
          <span>Your Sell</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full bg-cyan-500 inline-block" />
          <span>Pro Move</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-5 h-0.5 bg-brand-blue inline-block" />
          <span>Price</span>
        </div>
      </div>
    </div>
  );
}
