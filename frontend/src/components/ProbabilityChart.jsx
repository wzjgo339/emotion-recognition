import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

const EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];

const COLORS = {
  angry: '#ef4444',
  disgust: '#22c55e',
  fear: '#8b5cf6',
  happy: '#eab308',
  sad: '#3b82f6',
  surprise: '#f97316',
  neutral: '#9ca3af',
};

export default function ProbabilityChart({ probabilities, predicted }) {
  const data = EMOTIONS.map((e) => ({
    name: e,
    value: (probabilities[e] || 0) * 100,
    isPredicted: e === predicted,
  }));

  return (
    <div className="w-full h-64">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis
            dataKey="name"
            tick={{ fontSize: 12, fill: '#64748b' }}
            tickFormatter={(v) => v.charAt(0).toUpperCase() + v.slice(1)}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fontSize: 12, fill: '#64748b' }}
            tickFormatter={(v) => `${v}%`}
          />
          <Tooltip
            formatter={(value) => [`${value.toFixed(1)}%`, 'Probability']}
            labelFormatter={(label) => label.charAt(0).toUpperCase() + label.slice(1)}
          />
          <Bar dataKey="value" radius={[4, 4, 0, 0]}>
            {data.map((entry) => (
              <Cell
                key={entry.name}
                fill={COLORS[entry.name]}
                opacity={entry.isPredicted ? 1 : 0.35}
                stroke={entry.isPredicted ? '#1e293b' : 'none'}
                strokeWidth={entry.isPredicted ? 2 : 0}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
