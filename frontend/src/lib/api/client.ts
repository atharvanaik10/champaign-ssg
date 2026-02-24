import type { JobCreateRequest, JobInfo, ScheduleRow } from './types';

const API_BASE = 'http://localhost:8000';

export async function startJob(payload: JobCreateRequest): Promise<string> {
  const response = await fetch(`${API_BASE}/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  if (!response.ok) throw new Error(await response.text());
  const data = await response.json();
  return data.job_id as string;
}

export async function getJob(jobId: string): Promise<JobInfo> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}`);
  if (!response.ok) throw new Error(await response.text());
  return (await response.json()) as JobInfo;
}

export async function getSchedule(jobId: string): Promise<ScheduleRow[]> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}/schedule?format=json`);
  if (!response.ok) throw new Error(await response.text());
  return (await response.json()) as ScheduleRow[];
}

export async function getScheduleCsv(jobId: string): Promise<string> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}/schedule?format=csv`);
  if (!response.ok) throw new Error(await response.text());
  return await response.text();
}

export async function getGraph(graphPath: string): Promise<GeoJSON.FeatureCollection> {
  const response = await fetch(`${API_BASE}/graph?graph_path=${encodeURIComponent(graphPath)}`);
  if (!response.ok) throw new Error(await response.text());
  return (await response.json()) as GeoJSON.FeatureCollection;
}

export function subscribeEvents(jobId: string, onMessage: (value: JobInfo) => void): EventSource {
  const source = new EventSource(`${API_BASE}/jobs/${jobId}/events`);
  source.addEventListener('progress', (event) => {
    const payload = JSON.parse((event as MessageEvent).data) as JobInfo;
    onMessage(payload);
  });
  return source;
}
