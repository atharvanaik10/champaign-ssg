<script>
  export let rows = []
</script>

<div class="rounded border bg-white shadow-sm">
  <div class="max-h-96 overflow-y-auto">
    <table class="w-full table-fixed text-sm">
      <thead class="sticky top-0 bg-gray-100">
        <tr>
          <th class="w-24 p-2 text-left">time</th>
          <th class="w-20 text-left">unit</th>
          <th class="w-32 text-left">node</th>
          <th class="text-left">address</th>
        </tr>
      </thead>
      <tbody>
        {#each rows as r}
          <tr class="border-t">
            <td class="p-2">{r.time_step}</td>
            <td>{r.unit_id}</td>
            <td>{r.node_id}</td>
            <td>{r.address}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
</div>

