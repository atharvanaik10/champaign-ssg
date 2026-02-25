<script>
  export let text = '';
</script>

<div class="relative inline-block group" aria-label="info">
  <span class="ml-2 inline-flex h-4 w-4 items-center justify-center rounded-full bg-slate-200 text-[10px] font-semibold text-slate-600 select-none">i</span>
  <div class="pointer-events-none absolute z-[2000] -left-1/2 mt-2 w-56 rounded-md bg-slate-900 px-2 py-1 text-xs leading-snug text-white opacity-0 shadow-lg transition-opacity duration-150 group-hover:opacity-100">
    {text}
  </div>
</div>
