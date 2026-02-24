<script lang="ts">
  import type { ScheduleRow } from '$lib/api/types';

  export let rows: ScheduleRow[] = [];
</script>

<div class="max-h-80 overflow-auto rounded border bg-white">
  <table class="min-w-full text-sm">
    <thead class="sticky top-0 bg-slate-100">
      <tr>
        <th class="px-3 py-2 text-left">time_step</th>
        <th class="px-3 py-2 text-left">unit_id</th>
        <th class="px-3 py-2 text-left">node_id</th>
      </tr>
    </thead>
    <tbody>
      {#each rows as row}
        <tr class="border-t">
          <td class="px-3 py-1">{row.time_step}</td>
          <td class="px-3 py-1">{row.unit_id}</td>
          <td class="px-3 py-1">{row.node_id}</td>
        </tr>
      {/each}
    </tbody>
  </table>
</div>
