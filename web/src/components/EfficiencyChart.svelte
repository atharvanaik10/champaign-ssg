<script>
  import ApexCharts from 'apexcharts'
  import { onMount, onDestroy } from 'svelte'
  export let units = []
  export let ssg = []
  export let uniform = []

  let el
  let chart

  $: categories = (units || []).map((u) => String(u))
  $: series = [
    { name: 'SSG', data: (ssg || []).map((v) => Number(v) || 0) },
    { name: 'Uniform', data: (uniform || []).map((v) => Number(v) || 0) },
  ]
  $: ymax = Math.max(0.01, ...(ssg || []), ...(uniform || []))

  function buildOptions() {
    return {
      chart: { type: 'line', toolbar: { show: true }, animations: { enabled: true } },
      stroke: { curve: 'straight', width: 3 },
      markers: { size: 3 },
      dataLabels: { enabled: false },
      grid: { borderColor: '#e2e8f0' },
      colors: ['#2563eb', '#10b981'],
      xaxis: {
        categories,
        title: { text: 'UNITS', style: { color: '#475569', fontSize: '12px' } },
        axisBorder: { color: '#cbd5e1' },
        axisTicks: { color: '#cbd5e1' },
        labels: { style: { colors: '#64748b' } },
      },
      yaxis: {
        min: 0,
        max: ymax * 1.1,
        title: { text: 'EFFICIENCY', style: { color: '#475569', fontSize: '12px' } },
        decimalsInFloat: 5,
        labels: { formatter: (val) => Number(val).toFixed(5) },
      },
      tooltip: {
        theme: 'dark',
        style: { fontSize: '12px' },
        y: { formatter: (val) => Number(val).toFixed(5) }
      },
      legend: { position: 'top' },
      series,
    }
  }

  onMount(() => {
    chart = new ApexCharts(el, buildOptions())
    chart.render()
  })

  $: if (chart) {
    chart.updateOptions({ xaxis: { categories }, yaxis: { max: ymax * 1.1, decimalsInFloat: 5, labels: { formatter: (val) => Number(val).toFixed(5) } } }, false, true)
    chart.updateSeries(series, true)
  }

  onDestroy(() => { if (chart) chart.destroy() })
</script>

<div bind:this={el} class="w-full"></div>
