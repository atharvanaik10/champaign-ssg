<script>
  // items: [{ unit: number|string, color: string }]
  export let items = []
</script>

<div class="ml-auto flex flex-wrap items-center gap-3">
  {#each items as it}
    <div class="flex items-center gap-1 text-xs text-slate-600">
      <span class="inline-block h-2 w-2 rounded-full" style="background:{it.color}"></span>
      Unit {it.unit}
    </div>
  {/each}
</div>

