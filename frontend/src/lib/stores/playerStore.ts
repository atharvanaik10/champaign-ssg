import { writable } from 'svelte/store';

export const currentT = writable(0);
export const isPlaying = writable(false);
export const fps = writable(8);
export const maxT = writable(0);
