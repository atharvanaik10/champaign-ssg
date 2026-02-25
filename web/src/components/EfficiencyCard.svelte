<script>
  import Info from './Info.svelte'
  export let summary = null

  function fmt(n) {
    if (n == null) return ''
    if (typeof n === 'number') return Number.isInteger(n) ? n : n.toFixed(3)
    return String(n)
  }
</script>

<div class="space-y-4">
  <div class="grid grid-cols-2 gap-3">
    <div class="rounded-lg border bg-white p-3 text-center shadow-sm">
      <div class="flex items-center justify-center gap-2 text-xs text-slate-500">
        <span>Efficiency (SSG)</span>
        <Info text="Mean fraction of risk prevented by the SSG schedule across runs." />
      </div>
      <div class="text-lg font-semibold">{fmt(summary?.efficiency_ssg_mean)}</div>
    </div>
    <div class="rounded-lg border bg-white p-3 text-center shadow-sm">
      <div class="flex items-center justify-center gap-2 text-xs text-slate-500">
        <span>Efficiency (Uniform)</span>
        <Info text="Mean fraction of risk prevented by a uniform-random schedule across runs." />
      </div>
      <div class="text-lg font-semibold">{fmt(summary?.efficiency_uniform_mean)}</div>
    </div>
  </div>

  <div class="grid grid-cols-2 gap-3">
    <div class="rounded-lg border bg-white p-3 text-center shadow-sm">
      <div class="flex items-center justify-center gap-2 text-xs text-slate-500">
        <span>p(event)</span>
        <Info text="Probability that a crime event occurs at each timestep during simulation." />
      </div>
      <div class="text-lg font-semibold">{fmt(summary?.p_event)}</div>
    </div>
    <div class="rounded-lg border bg-white p-3 text-center shadow-sm">
      <div class="flex items-center justify-center gap-2 text-xs text-slate-500">
        <span>Runs</span>
        <Info text="Number of Monte Carlo runs used for evaluation (higher is smoother)." />
      </div>
      <div class="text-lg font-semibold">{fmt(summary?.num_runs)}</div>
    </div>
  </div>
</div>
