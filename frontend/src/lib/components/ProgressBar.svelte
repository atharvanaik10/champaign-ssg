<script lang="ts">
  export let progress = 0;
  export let message = 'Idle';
</script>

<div class="space-y-1">
  <div class="flex justify-between text-sm">
    <span>{message}</span>
    <span>{Math.round(progress * 100)}%</span>
  </div>
  <div class="h-3 w-full rounded bg-slate-300">
    <div class="h-3 rounded bg-blue-600 transition-all" style={`width:${Math.max(0, Math.min(100, progress * 100))}%`}></div>
  </div>
</div>
