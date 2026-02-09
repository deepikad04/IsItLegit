export default function LuckSkillBar({ luck, skill }) {
  const luckPct = Math.round(luck * 100);
  const skillPct = Math.round(skill * 100);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-amber-500" />
          <span className="text-sm font-semibold text-brand-navy">Luck</span>
          <span className="text-lg font-black text-amber-600">{luckPct}%</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-lg font-black text-cyan-600">{skillPct}%</span>
          <span className="text-sm font-semibold text-brand-navy">Skill</span>
          <span className="w-3 h-3 rounded-full bg-cyan-500" />
        </div>
      </div>
      <div className="h-5 bg-gray-100 rounded-full overflow-hidden flex shadow-inner">
        <div
          className="bg-gradient-to-r from-amber-400 to-amber-500 transition-all duration-700 ease-out rounded-l-full"
          style={{ width: `${luckPct}%` }}
        />
        <div
          className="bg-gradient-to-r from-cyan-400 to-cyan-500 transition-all duration-700 ease-out rounded-r-full"
          style={{ width: `${skillPct}%` }}
        />
      </div>
      <p className="text-xs text-center text-brand-navy/50">
        {skillPct > luckPct
          ? 'Your decisions drove the outcome more than market conditions'
          : skillPct === luckPct
            ? 'Equal parts luck and skill in this simulation'
            : 'Market conditions played a bigger role than your decisions'}
      </p>
    </div>
  );
}
