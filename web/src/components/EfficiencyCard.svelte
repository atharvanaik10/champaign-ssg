<script>
  import Info from './Info.svelte'
  export let summary = null

  function fmt(n) {
    if (n == null) return ''
    if (typeof n === 'number') return Number.isInteger(n) ? n : n.toFixed(3)
    return String(n)
  }

  function fmtInt(n) {
    if (n == null) return ''
    const v = Number(n)
    return Number.isFinite(v) ? Math.round(v) : ''
  }

  function fmtPct(ratio) {
    if (ratio == null) return ''
    const v = Number(ratio)
    if (!Number.isFinite(v)) return ''
    return `${(v * 100).toFixed(1)}%`
  }

  function fmtList(list) {
    if (!Array.isArray(list)) return ''
    return list.map((x) => fmtInt(x)).join(', ')
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

  <!-- Movement -->
  <div class="grid grid-cols-2 gap-3">
    <div class="rounded-lg border bg-white p-3 text-center shadow-sm">
      <div class="flex items-center justify-center gap-2 text-xs text-slate-500">
        <span>Movement (SSG)</span>
        <Info text="Total hops across all units (node changes between timesteps). Also shows per‑unit hops." />
      </div>
      <div class="text-lg font-semibold">{fmtInt(summary?.movement_ssg_total_hops)}</div>
      {#if Array.isArray(summary?.movement_ssg_by_unit_hops)}
        <div class="mt-1 text-xs text-slate-500">Per‑unit: {fmtList(summary.movement_ssg_by_unit_hops)}</div>
      {/if}
    </div>
    <div class="rounded-lg border bg-white p-3 text-center shadow-sm">
      <div class="flex items-center justify-center gap-2 text-xs text-slate-500">
        <span>Movement (Uniform)</span>
        <Info text="Total hops across all units for the uniform baseline. Also shows per‑unit hops." />
      </div>
      <div class="text-lg font-semibold">{fmtInt(summary?.movement_uniform_total_hops)}</div>
      {#if Array.isArray(summary?.movement_uniform_by_unit_hops)}
        <div class="mt-1 text-xs text-slate-500">Per‑unit: {fmtList(summary.movement_uniform_by_unit_hops)}</div>
      {/if}
    </div>
  </div>

  <!-- Coverage -->
  <div class="grid grid-cols-2 gap-3">
    <div class="rounded-lg border bg-white p-3 text-center shadow-sm">
      <div class="flex items-center justify-center gap-2 text-xs text-slate-500">
        <span>Coverage (SSG)</span>
        <Info text="Unique nodes visited by any unit divided by total nodes in the graph. Also shows per‑unit coverage counts." />
      </div>
      <div class="text-lg font-semibold">
        {fmtInt(summary?.coverage_ssg_total_count)}/{fmtInt(summary?.coverage_ssg_total_nodes)}
        <span class="ml-1 text-sm text-slate-500">{fmtPct(summary?.coverage_ssg_total_ratio)}</span>
      </div>
      {#if Array.isArray(summary?.coverage_ssg_by_unit_count)}
        <div class="mt-1 text-xs text-slate-500">Per‑unit (count): {fmtList(summary.coverage_ssg_by_unit_count)}</div>
      {/if}
    </div>
    <div class="rounded-lg border bg-white p-3 text-center shadow-sm">
      <div class="flex items-center justify-center gap-2 text-xs text-slate-500">
        <span>Coverage (Uniform)</span>
        <Info text="Unique nodes visited by any unit (uniform baseline) divided by total nodes. Also shows per‑unit coverage counts." />
      </div>
      <div class="text-lg font-semibold">
        {fmtInt(summary?.coverage_uniform_total_count)}/{fmtInt(summary?.coverage_uniform_total_nodes)}
        <span class="ml-1 text-sm text-slate-500">{fmtPct(summary?.coverage_uniform_total_ratio)}</span>
      </div>
      {#if Array.isArray(summary?.coverage_uniform_by_unit_count)}
        <div class="mt-1 text-xs text-slate-500">Per‑unit (count): {fmtList(summary.coverage_uniform_by_unit_count)}</div>
      {/if}
    </div>
  </div>
</div>
