import { writable } from 'svelte/store';

import type { JobInfo, ScheduleRow } from '$lib/api/types';

export const jobInfo = writable<JobInfo | null>(null);
export const scheduleRows = writable<ScheduleRow[]>([]);
export const graphData = writable<GeoJSON.FeatureCollection | null>(null);
