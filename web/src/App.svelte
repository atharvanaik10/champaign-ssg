<script>
  import L from 'leaflet'
  import 'leaflet/dist/leaflet.css'
  import { onMount } from 'svelte'
  import Info from './components/Info.svelte'
  import ScheduleTable from './components/ScheduleTable.svelte'
  import MapLegend from './components/MapLegend.svelte'
  import EfficiencyCard from './components/EfficiencyCard.svelte'
  import EfficiencyChart from './components/EfficiencyChart.svelte'
  import ConfigCard from './components/ConfigCard.svelte'

  let status = 'idle' // 'idle' | 'running' | 'done' | 'error'
  let message = ''
  let progress = 0

  let graphPath = 'data/uiuc_graph.json'
  // Parameters
  let resourceBudget = 10
  let timeSteps = 120
  let units = 5
  let startIndex = 0
  let seed = 0
  let pEvent = 0.3
  let numRuns = 200
  let alpha = 1,
    beta = 1,
    gamma = 1,
    delta = 1

  let schedule = []
  let geo = null
  let tMax = 0
  let t = 0
  let isPlaying = false
  let fps = 6
  let raf = 0
  let selectedUnit = ''
  let unitColors = new Map()
  let summary = null
  let filteredSchedule = []
  let displayRows = []
  let legendItems = []
  const palette = [
    '#2563eb',
    '#10b981',
    '#f59e0b',
    '#ef4444',
    '#8b5cf6',
    '#14b8a6',
    '#f97316',
    '#ec4899',
    '#22c55e',
    '#0ea5e9',
  ]
  let mapEl
  let map = null
  let edgesLayer = null
  let unitsLayer = null
  let trailsLayer = null
  let trailPolylines = new Map()
  let showTrails = true

  async function fetchGraph() {
    try {
      const res = await fetch(
        `/graph?graph_path=${encodeURIComponent(graphPath)}`,
      )
      if (!res.ok) throw new Error(await res.text())
      geo = await res.json()
    } catch (e) {
      status = 'error'
      message = `Graph load failed: ${String(e)}`
    }
  }

  async function run() {
    status = 'running'
    message = 'Queued'
    progress = 0
    schedule = []
    try {
      const create = await fetch('/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          graph_path: graphPath,
          game: { alpha, beta, gamma, delta, resource_budget: resourceBudget },
          patrol: {
            time_steps: timeSteps,
            num_units: units,
            start_index: startIndex,
            random_seed: seed,
          },
          eval: { p_event: pEvent, num_runs: numRuns },
        }),
      })
      if (!create.ok) throw new Error(await create.text())
      const { job_id } = await create.json()

      const es = new EventSource(`/jobs/${job_id}/events`)
      es.addEventListener('progress', async (evt) => {
        const data = JSON.parse(evt.data)
        status = data.status
        message = data.message
        progress = Math.round((data.progress || 0) * 100)
        if (data.status === 'done') {
          es.close()
          const res = await fetch(`/jobs/${job_id}/schedule?format=json`)
          if (!res.ok) throw new Error(await res.text())
          schedule = await res.json()
          tMax = Math.max(...schedule.map((r) => r.time_step))
          t = 0
          // assign colors per unit
          const uniq = [...new Set(schedule.map((r) => r.unit_id))]
          unitColors = new Map(
            uniq.map((u, i) => [u, palette[i % palette.length]]),
          )
          legendItems = uniq.map((u) => ({ unit: u, color: unitColors.get(u) || '#ef4444' }))
          // fetch summary
          const stat = await fetch(`/jobs/${job_id}`)
          if (stat.ok) {
            const info = await stat.json()
            summary = info.summary || null
          }
          if (!geo) await fetchGraph()
          draw()
        }
        if (data.status === 'error') {
          es.close()
        }
      })
    } catch (e) {
      status = 'error'
      message = String(e)
    }
  }

  function nodeLookup() {
    const m = new Map()
    for (const f of (geo && geo.features) || []) {
      if (
        f.geometry.type === 'Point' &&
        f.properties &&
        typeof f.properties.node_id === 'string'
      ) {
        m.set(String(f.properties.node_id), f.geometry.coordinates)
      }
    }
    return m
  }

  function draw() {
    if (!map) return
    const byNode = nodeLookup()
    const rows = schedule.filter((r) => r.time_step === t)
    const features = rows
      .map((row) => {
        const coords = byNode.get(String(row.node_id))
        if (!coords) return null
        const color = unitColors.get(row.unit_id) || '#ef4444'
        // Leaflet expects [lat, lon]
        return L.circleMarker([coords[1], coords[0]], {
          radius: 6,
          color,
          weight: 2,
          opacity: 1,
          fillOpacity: 0.9,
        })
      })
      .filter(Boolean)
    if (unitsLayer) {
      unitsLayer.clearLayers()
      features.forEach((f) => unitsLayer.addLayer(f))
    }

    // Trails up to current t
    if (!trailsLayer) {
      trailsLayer = L.layerGroup().addTo(map)
    }
    if (!showTrails) {
      trailsLayer.clearLayers()
      trailPolylines.clear()
    } else {
      const unitIds = [...new Set(schedule.map((r) => r.unit_id))]
      unitIds.forEach((u) => {
        const pathRows = schedule
          .filter((r) => r.unit_id === u && r.time_step <= t)
          .sort((a, b) => a.time_step - b.time_step)
        const latlngs = []
        for (const r of pathRows) {
          const c = byNode.get(String(r.node_id))
          if (c) latlngs.push([c[1], c[0]])
        }
        const color = unitColors.get(u) || '#ef4444'
        let poly = trailPolylines.get(u)
        if (latlngs.length > 1) {
          if (!poly) {
            poly = L.polyline(latlngs, { color, weight: 2, opacity: 0.9 })
            trailsLayer.addLayer(poly)
            trailPolylines.set(u, poly)
          } else {
            poly.setLatLngs(latlngs)
            poly.setStyle({ color })
          }
        } else if (poly) {
          trailsLayer.removeLayer(poly)
          trailPolylines.delete(u)
        }
      })
    }
  }

  onMount(async () => {
    await fetchGraph()
    map = L.map(mapEl).setView([40.1106, -88.2272], 14)
    // Light basemap (Carto Positron)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
    }).addTo(map)

    // Add edges as a GeoJSON layer
    const edges = ((geo && geo.features) || []).filter(
      (f) => f.geometry.type === 'LineString',
    )
    edgesLayer = L.geoJSON(
      { type: 'FeatureCollection', features: edges },
      {
        style: { color: '#334155', weight: 2, opacity: 0.9 },
      },
    ).addTo(map)

    // Units layer group
    unitsLayer = L.layerGroup().addTo(map)
  })

  function tick(ts) {
    if (isPlaying && tMax > 0) {
      // advance based on fps
      if (!tick.last || ts - tick.last >= 1000 / Math.max(1, fps)) {
        t = t + 1
        if (t > tMax) t = 0
        draw()
        tick.last = ts
      }
    }
    raf = requestAnimationFrame(tick)
  }

  onMount(() => {
    raf = requestAnimationFrame(tick)
  })

  $: filteredSchedule = selectedUnit === ''
    ? schedule
    : schedule.filter((r) => String(r.unit_id) === String(selectedUnit) || r.unit_id === Number(selectedUnit))

  $: displayRows = (() => {
    const nodes = nodeLookup()
    return filteredSchedule.map((r) => {
      const coords = nodes.get(String(r.node_id))
      const addr = coords ? `${coords[1].toFixed(6)}, ${coords[0].toFixed(6)}` : ''
      return { ...r, address: addr }
    })
  })()

  function downloadCsv() {
    const rows = displayRows
    const header = ['time_step', 'unit_id', 'node_id', 'address']
    const lines = [header.join(',')].concat(
      rows.map(
        (r) => `${r.time_step},${r.unit_id},${r.node_id},"${r.address}"`,
      ),
    )
    const blob = new Blob([lines.join('\n')], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download =
      selectedUnit === ''
        ? 'patrol_schedule.csv'
        : `patrol_schedule_unit_${selectedUnit}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  function openRoute() {
    if (selectedUnit === '') return
    const nodes = nodeLookup()
    const rows = filteredSchedule
      .sort((a, b) => a.time_step - b.time_step)
    if (rows.length < 2) return
    const points = rows.map((r) => nodes.get(r.node_id)).filter(Boolean)
    if (!points.length) return
    const toLatLon = (p) => `${p[1]},${p[0]}`
    const origin = toLatLon(points[0])
    const destination = toLatLon(points[points.length - 1])
    const waypoints = points.slice(1, -1).map(toLatLon).join('|')
    const params = [
      `origin=${encodeURIComponent(origin)}`,
      `destination=${encodeURIComponent(destination)}`,
      'travelmode=driving',
    ]
    if (waypoints) params.push(`waypoints=${encodeURIComponent(waypoints)}`)
    const url = `https://www.google.com/maps/dir/?api=1&${params.join('&')}`
    window.open(url, '_blank')
  }

  function statusLabel() {
    if (!status) return ''
    if (status === 'running') return 'Running'
    return status.charAt(0).toUpperCase() + status.slice(1)
  }

  function fmt(n) {
    if (n == null) return ''
    if (typeof n === 'number') {
      return Number.isInteger(n) ? n.toString() : n.toFixed(3)
    }
    return String(n)
  }

  // Reactive filtered schedule and display rows handled above
</script>

<style>
  /* ensure maplibre styles are bundled */
</style>

<div class="mx-auto max-w-6xl space-y-4 p-4">
  <h1 class="text-xl font-semibold">ALMA Patrol Planner</h1>

  <div class="grid grid-cols-1 gap-4 lg:grid-cols-[340px_1fr]">
    <aside class="relative z-[1500]">
      <ConfigCard
        bind:graphPath={graphPath}
        bind:alpha={alpha}
        bind:beta={beta}
        bind:gamma={gamma}
        bind:delta={delta}
        bind:resourceBudget={resourceBudget}
        bind:timeSteps={timeSteps}
        bind:units={units}
        bind:startIndex={startIndex}
        bind:seed={seed}
        bind:pEvent={pEvent}
        bind:numRuns={numRuns}
        {status}
        {message}
        {progress}
        on:start={run}
      />
    </aside>

    <main class="space-y-4">
      <!-- Map Card -->
      <div class="rounded-xl border bg-white p-4 shadow-sm">
        <div class="mb-2 flex items-center justify-between">
          <div class="text-xs font-medium uppercase tracking-wide text-slate-500">Map</div>
          <MapLegend items={legendItems} />
        </div>
        <div class="mb-3 flex flex-wrap items-center gap-3">
          <button class="rounded-lg border px-3 py-1.5 hover:bg-slate-50" on:click={() => { isPlaying = !isPlaying }}>{isPlaying ? 'Pause' : 'Play'}</button>
          <div class="flex items-center gap-2">
            <span class="text-sm">t</span>
            <input type="range" min="0" max={tMax} bind:value={t} on:input={draw} />
            <span class="text-sm">{t}/{tMax}</span>
          </div>
          <div class="flex items-center gap-2">
            <span class="text-sm">FPS</span>
            <input type="number" min="1" max="30" class="w-20 rounded-lg border p-1" bind:value={fps} />
          </div>
        </div>
        <div bind:this={mapEl} class="h-96 w-full overflow-hidden rounded border" />
      </div>

      {#if schedule.length}
        <!-- Efficiency Card -->
        {#if summary}
          <div class="rounded-xl border bg-white p-4 shadow-sm">
            <div class="mb-4 text-xs font-medium uppercase tracking-wide text-slate-500">Efficiency</div>
            {#if Array.isArray(summary?.eff_by_units_units)}
              <div class="mb-4">
                <EfficiencyChart units={summary.eff_by_units_units} ssg={summary.eff_by_units_ssg} uniform={summary.eff_by_units_uniform} />
              </div>
            {/if}
            <EfficiencyCard {summary} />
            <div class="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <div class="rounded-lg border bg-white p-3 text-center shadow-sm">
                <div class="flex items-center justify-center gap-2 text-xs text-slate-500">
                  <span>Best utility</span>
                  <Info text="Optimal defender utility from the SSG solution (higher is better)." />
                </div>
                <div class="text-lg font-semibold">{fmt(summary.best_defender_utility)}</div>
              </div>
              <div class="rounded-lg border bg-white p-3 text-center shadow-sm">
                <div class="flex items-center justify-center gap-2 text-xs text-slate-500">
                  <span>Nodes</span>
                  <Info text="Number of nodes (intersections) in the road graph." />
                </div>
                <div class="text-lg font-semibold">{fmt(summary.nodes)}</div>
              </div>
              <div class="rounded-lg border bg-white p-3 text-center shadow-sm">
                <div class="flex items-center justify-center gap-2 text-xs text-slate-500">
                  <span>Edges</span>
                  <Info text="Number of edges (roads) in the graph." />
                </div>
                <div class="text-lg font-semibold">{fmt(summary.edges)}</div>
              </div>
              <div class="rounded-lg border bg-white p-3 text-center shadow-sm">
                <div class="flex items-center justify-center gap-2 text-xs text-slate-500">
                  <span>Timesteps</span>
                  <Info text="Length of the simulated schedule (0..T)." />
                </div>
                <div class="text-lg font-semibold">{tMax}</div>
              </div>
            </div>
          </div>
        {/if}

        <!-- Table Card -->
        <div class="rounded-xl border bg-white p-4 shadow-sm">
          <div class="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">Schedule</div>
          <div class="mb-3 flex flex-wrap items-center gap-3">
            <div class="flex items-center gap-2">
              <span class="text-sm">Unit</span>
              <select class="rounded-lg border p-1" bind:value={selectedUnit}>
                <option value="">All</option>
                {#each [...new Set(schedule.map((r) => r.unit_id))] as u}
                  <option value={u}>{u}</option>
                {/each}
              </select>
            </div>
            <button class="rounded-lg border px-3 py-1.5 hover:bg-slate-50" on:click={downloadCsv}>Download CSV</button>
            <button class="rounded-lg border px-3 py-1.5 hover:bg-slate-50 disabled:opacity-50" on:click={openRoute} disabled={selectedUnit === ''}>Open in Google Maps</button>
          </div>
          <ScheduleTable rows={displayRows} />
        </div>
      {/if}
    </main>
  </div>
</div>
