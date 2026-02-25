<script>
  import { createEventDispatcher } from 'svelte'
  import Info from './Info.svelte'

  export let graphPath = ''
  export let alpha = 1
  export let beta = 1
  export let gamma = 1
  export let delta = 1
  export let resourceBudget = 10
  export let timeSteps = 120
  export let units = 5
  export let startIndex = 0
  export let seed = 0
  export let pEvent = 0.3
  export let numRuns = 200

  export let status = 'idle'
  export let message = ''
  export let progress = 0

  const dispatch = createEventDispatcher()
  function start() { dispatch('start') }
</script>

<div class="space-y-4 rounded-xl border bg-white p-5 shadow-sm">
  <div>
    <div class="text-xs font-medium uppercase tracking-wide text-slate-500">Data</div>
    <label class="mt-1 block text-sm">
      <div class="flex items-center justify-between">
        <span>Graph path</span>
        <Info text="Path to the adjacency JSON with lat/lon, risk_factor, and neighbors." />
      </div>
      <input class="mt-1 w-full rounded-lg border p-2" bind:value={graphPath} />
    </label>
  </div>

  <div>
    <div class="mb-1 text-xs font-medium uppercase tracking-wide text-slate-500">Game parameters</div>
    <div class="grid grid-cols-2 gap-2">
      <label class="text-sm">
        <div class="flex items-center justify-between"><span>Alpha</span><Info text="Defender reward when the attacked node is covered." /></div>
        <input type="number" step="0.1" class="mt-1 w-full rounded-lg border p-2" bind:value={alpha} />
      </label>
      <label class="text-sm">
        <div class="flex items-center justify-between"><span>Beta</span><Info text="Defender loss when the attacked node is uncovered." /></div>
        <input type="number" step="0.1" class="mt-1 w-full rounded-lg border p-2" bind:value={beta} />
      </label>
      <label class="text-sm">
        <div class="flex items-center justify-between"><span>Gamma</span><Info text="Attacker reward when the attacked node is uncovered." /></div>
        <input type="number" step="0.1" class="mt-1 w-full rounded-lg border p-2" bind:value={gamma} />
      </label>
      <label class="text-sm">
        <div class="flex items-center justify-between"><span>Delta</span><Info text="Attacker loss when the attacked node is covered." /></div>
        <input type="number" step="0.1" class="mt-1 w-full rounded-lg border p-2" bind:value={delta} />
      </label>
    </div>
    <label class="mt-2 block text-sm">
      <div class="flex items-center justify-between"><span>Resource budget</span><Info text="K: Upper bound on the total coverage allocation across nodes (not # of units)." /></div>
      <input type="number" class="mt-1 w-full rounded-lg border p-2" bind:value={resourceBudget} />
    </label>
  </div>

  <div>
    <div class="mb-1 text-xs font-medium uppercase tracking-wide text-slate-500">Patrol</div>
    <div class="grid grid-cols-2 gap-2">
      <label class="text-sm">
        <div class="flex items-center justify-between"><span>Time steps</span><Info text="Length of the simulated route (t = 0..T)." /></div>
        <input type="number" class="mt-1 w-full rounded-lg border p-2" bind:value={timeSteps} />
      </label>
      <label class="text-sm">
        <div class="flex items-center justify-between"><span>Units</span><Info text="Number of patrol units moving simultaneously." /></div>
        <input type="number" class="mt-1 w-full rounded-lg border p-2" bind:value={units} />
      </label>
      <label class="text-sm">
        <div class="flex items-center justify-between"><span>Start index</span><Info text="Index in the node list where all units start (0-based)." /></div>
        <input type="number" class="mt-1 w-full rounded-lg border p-2" bind:value={startIndex} />
      </label>
      <label class="text-sm">
        <div class="flex items-center justify-between"><span>Seed</span><Info text="Random seed for reproducible simulation." /></div>
        <input type="number" class="mt-1 w-full rounded-lg border p-2" bind:value={seed} />
      </label>
    </div>
  </div>

  <div>
    <div class="mb-1 text-xs font-medium uppercase tracking-wide text-slate-500">Evaluation</div>
    <div class="grid grid-cols-2 gap-2">
      <label class="text-sm">
        <div class="flex items-center justify-between"><span>p(event)</span><Info text="Probability a crime occurs at a timestep in the simulation." /></div>
        <input type="number" step="0.01" min="0" max="1" class="mt-1 w-full rounded-lg border p-2" bind:value={pEvent} />
      </label>
      <label class="text-sm">
        <div class="flex items-center justify-between"><span>Runs</span><Info text="Number of Monte Carlo runs for efficiency evaluation." /></div>
        <input type="number" min="1" class="mt-1 w-full rounded-lg border p-2" bind:value={numRuns} />
      </label>
    </div>
  </div>

  <button class="w-full rounded-lg bg-blue-600 px-3 py-2 font-medium text-white hover:bg-blue-700" on:click={start} disabled={status === 'running'}>
    Start
  </button>
  {#if status !== 'idle'}
    <div class="text-sm text-gray-600">{status === 'running' ? 'Running' : status}: {message}</div>
    <div class="h-2 w-full rounded bg-gray-200">
      <div class="h-2 rounded bg-blue-500 transition-all" style={`width:${progress}%`} />
    </div>
  {/if}
</div>

