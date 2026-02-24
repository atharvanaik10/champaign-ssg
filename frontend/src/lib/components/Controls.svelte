<script lang="ts">
  import { currentT, fps, isPlaying, maxT } from '$lib/stores/playerStore';
  import { get } from 'svelte/store';

  function toggle(): void {
    isPlaying.update((v) => !v);
  }

  function seek(delta: number): void {
    const next = Math.max(0, Math.min(get(maxT), get(currentT) + delta));
    currentT.set(next);
  }
</script>

<div class="flex flex-wrap items-center gap-3 rounded border bg-white p-3">
  <button class="rounded bg-slate-800 px-3 py-1 text-white" on:click={toggle}>{$isPlaying ? 'Pause' : 'Play'}</button>
  <button class="rounded border px-2 py-1" on:click={() => seek(-1)}>◀</button>
  <button class="rounded border px-2 py-1" on:click={() => seek(1)}>▶</button>
  <label class="flex items-center gap-2 text-sm">FPS
    <input type="range" min="1" max="30" bind:value={$fps} />
    <span>{$fps}</span>
  </label>
  <label class="flex grow items-center gap-2 text-sm">Time
    <input class="w-full" type="range" min="0" max={$maxT || 0} step="1" bind:value={$currentT} />
    <span>{$currentT}</span>
  </label>
</div>
