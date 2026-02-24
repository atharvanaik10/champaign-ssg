<script lang="ts">
  import maplibregl from 'maplibre-gl';
  import 'maplibre-gl/dist/maplibre-gl.css';
  import { onDestroy, onMount } from 'svelte';
  import { currentT } from '$lib/stores/playerStore';
  import type { ScheduleRow } from '$lib/api/types';

  export let graph: GeoJSON.FeatureCollection | null = null;
  export let schedule: ScheduleRow[] = [];

  let mapHost: HTMLDivElement;
  let map: maplibregl.Map | null = null;
  let currentTime = 0;

  const unsub = currentT.subscribe((v) => {
    currentTime = v;
    refreshUnits();
  });

  function nodeLookup(): Map<string, number[]> {
    const lookup = new Map<string, number[]>();
    for (const feature of graph?.features ?? []) {
      if (feature.geometry.type === 'Point' && typeof feature.properties?.node_id === 'string') {
        lookup.set(feature.properties.node_id, feature.geometry.coordinates as number[]);
      }
    }
    return lookup;
  }

  function refreshUnits(): void {
    if (!map) return;
    const byNode = nodeLookup();
    const rows = schedule.filter((r) => r.time_step === currentTime);
    const points = rows
      .map((row) => {
        const coords = byNode.get(row.node_id);
        if (!coords) return null;
        return {
          type: 'Feature',
          properties: { unit_id: row.unit_id, node_id: row.node_id },
          geometry: { type: 'Point', coordinates: coords }
        };
      })
      .filter(Boolean);

    const source = map.getSource('units') as maplibregl.GeoJSONSource;
    source?.setData({ type: 'FeatureCollection', features: points as GeoJSON.Feature[] });
  }

  onMount(() => {
    map = new maplibregl.Map({
      container: mapHost,
      style: 'https://demotiles.maplibre.org/style.json',
      center: [-88.2272, 40.1106],
      zoom: 14
    });

    map.on('load', () => {
      const edgeFeatures = (graph?.features ?? []).filter((f) => f.geometry.type === 'LineString');
      map?.addSource('edges', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: edgeFeatures }
      });
      map?.addLayer({
        id: 'edges-layer',
        type: 'line',
        source: 'edges',
        paint: { 'line-color': '#475569', 'line-width': 2 }
      });

      map?.addSource('units', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] }
      });
      map?.addLayer({
        id: 'units-layer',
        type: 'circle',
        source: 'units',
        paint: { 'circle-color': '#ef4444', 'circle-radius': 6 }
      });
      refreshUnits();
    });
  });

  $: if (map && graph) {
    const edgeFeatures = graph.features.filter((f) => f.geometry.type === 'LineString');
    const source = map.getSource('edges') as maplibregl.GeoJSONSource;
    source?.setData({ type: 'FeatureCollection', features: edgeFeatures });
    refreshUnits();
  }

  onDestroy(() => {
    unsub();
    map?.remove();
  });
</script>

<div bind:this={mapHost} class="h-96 w-full rounded border"></div>
