<script lang="ts">
  import { onDestroy } from 'svelte';
  import { get } from 'svelte/store';

  import { getGraph, getJob, getSchedule, getScheduleCsv, startJob, subscribeEvents } from '$lib/api/client';
  import type { JobCreateRequest } from '$lib/api/types';
  import Controls from '$lib/components/Controls.svelte';
  import MapView from '$lib/components/MapView.svelte';
  import ProgressBar from '$lib/components/ProgressBar.svelte';
  import ScheduleTable from '$lib/components/ScheduleTable.svelte';
  import { graphData, jobInfo, scheduleRows } from '$lib/stores/jobStore';
  import { currentT, fps, isPlaying, maxT } from '$lib/stores/playerStore';

  let eventSource: EventSource | null = null;
  let raf = 0;
  let lastTs = 0;

  let request: JobCreateRequest = {
    graph_path: 'data/uiuc_graph.json',
    game: { alpha: 1, beta: 1, gamma: 1, delta: 1, resource_budget: 10 },
    patrol: { time_steps: 120, num_units: 5, start_index: 0, random_seed: 0 }
  };

  async function run(): Promise<void> {
    const jobId = await startJob(request);
    jobInfo.set({ job_id: jobId, status: 'queued', progress: 0, message: 'Queued' });

    eventSource?.close();
    eventSource = subscribeEvents(jobId, async (evt) => {
      jobInfo.set(evt);
      if (evt.status === 'done') {
        scheduleRows.set(await getSchedule(jobId));
        graphData.set(await getGraph(request.graph_path));
        const tMax = Math.max(...get(scheduleRows).map((r) => r.time_step));
        maxT.set(tMax);
      }
      if (evt.status === 'error') {
        eventSource?.close();
      }
    });

    eventSource.onerror = async () => {
      // fallback polling
      const info = await getJob(jobId);
      jobInfo.set(info);
      if (info.status === 'done') {
        scheduleRows.set(await getSchedule(jobId));
        graphData.set(await getGraph(request.graph_path));
        const tMax = Math.max(...get(scheduleRows).map((r) => r.time_step));
        maxT.set(tMax);
      }
    };
  }

  async function downloadCsv(): Promise<void> {
    const id = get(jobInfo)?.job_id;
    if (!id) return;
    const text = await getScheduleCsv(id);
    const blob = new Blob([text], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `patrol_schedule_${id}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function tick(ts: number): void {
    const playing = get(isPlaying);
    if (playing) {
      const interval = 1000 / Math.max(1, get(fps));
      if (ts - lastTs >= interval) {
        const next = get(currentT) + 1;
        const limit = Math.max(0, get(maxT));
        currentT.set(next > limit ? 0 : next);
        lastTs = ts;
      }
    }
    raf = requestAnimationFrame(tick);
  }

  function onKey(event: KeyboardEvent): void {
    if (event.code === 'Space') {
      event.preventDefault();
      isPlaying.update((v) => !v);
    }
    if (event.code === 'ArrowRight') currentT.update((v) => Math.min(v + 1, get(maxT)));
    if (event.code === 'ArrowLeft') currentT.update((v) => Math.max(v - 1, 0));
  }

  if (typeof window !== 'undefined') {
    window.addEventListener('keydown', onKey);
    raf = requestAnimationFrame(tick);
  }

  onDestroy(() => {
    eventSource?.close();
    cancelAnimationFrame(raf);
    if (typeof window !== 'undefined') window.removeEventListener('keydown', onKey);
  });
</script>

<div class="grid min-h-screen grid-cols-1 gap-4 p-4 lg:grid-cols-[340px_1fr]">
  <aside class="space-y-3 rounded border bg-white p-4">
    <h1 class="text-xl font-semibold">ALMA Patrol Planner</h1>
    <label class="block text-sm">Graph path
      <input class="mt-1 w-full rounded border p-2" bind:value={request.graph_path} />
    </label>
    <label class="block text-sm">Resource budget
      <input type="number" class="mt-1 w-full rounded border p-2" bind:value={request.game.resource_budget} />
    </label>
    <label class="block text-sm">Time steps
      <input type="number" class="mt-1 w-full rounded border p-2" bind:value={request.patrol.time_steps} />
    </label>
    <label class="block text-sm">Units
      <input type="number" class="mt-1 w-full rounded border p-2" bind:value={request.patrol.num_units} />
    </label>
    <button class="w-full rounded bg-blue-600 px-3 py-2 text-white" on:click={run}>Start Job</button>
    {#if $jobInfo}
      <ProgressBar progress={$jobInfo.progress} message={$jobInfo.message} />
    {/if}
  </aside>

  <main class="space-y-3">
    <Controls />
    <MapView graph={$graphData} schedule={$scheduleRows} />
    <div class="flex items-center justify-between">
      <h2 class="text-lg font-semibold">Schedule</h2>
      <button class="rounded border px-3 py-1" on:click={downloadCsv}>Download CSV</button>
    </div>
    <ScheduleTable rows={$scheduleRows} />
  </main>
</div>
