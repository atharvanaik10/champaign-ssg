export type JobStatus = 'queued' | 'running' | 'done' | 'error';

export interface JobCreateRequest {
  graph_path: string;
  game: {
    alpha: number;
    beta: number;
    gamma: number;
    delta: number;
    resource_budget: number;
  };
  patrol: {
    time_steps: number;
    num_units: number;
    start_index: number;
    random_seed: number;
  };
}

export interface JobInfo {
  job_id: string;
  status: JobStatus;
  progress: number;
  message: string;
  summary?: Record<string, number>;
  error?: string;
}

export interface ScheduleRow {
  time_step: number;
  unit_id: number;
  node_id: string;
}
