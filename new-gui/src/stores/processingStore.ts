import { create } from 'zustand';

export type ProcessingStatus = 'idle' | 'processing' | 'stopping' | 'completed' | 'error';

export interface ProcessingLog {
  type: 'info' | 'success' | 'warning' | 'error' | 'stdout' | 'stderr';
  message: string;
  timestamp: number;
}

export interface ProcessingState {
  status: ProcessingStatus;
  progress: number;
  currentFrame: number;
  totalFrames: number;
  fps: number;
  elapsed: string;
  eta: string;
  logs: ProcessingLog[];
  error: string | null;
  isPaused: boolean;
}

interface ProcessingStore extends ProcessingState {
  startProcessing: () => void;
  stopProcessing: () => void;
  pauseProcessing: () => void;
  resumeProcessing: () => void;
  completeProcessing: () => void;
  failProcessing: (error: string) => void;
  resetProcessing: () => void;
  updateProgress: (progress: number) => void;
  setFrameInfo: (currentFrame: number, totalFrames: number) => void;
  setPerformance: (fps: number, elapsed: string, eta: string) => void;
  addLog: (log: ProcessingLog) => void;
  clearLogs: () => void;
  setError: (error: string) => void;
}

const initialState: ProcessingState = {
  status: 'idle',
  progress: 0,
  currentFrame: 0,
  totalFrames: 0,
  fps: 0,
  elapsed: '00:00',
  eta: '00:00',
  logs: [],
  error: null,
  isPaused: false,
};

export const useProcessingStore = create<ProcessingStore>((set) => ({
  ...initialState,
  
  startProcessing: () => set({
    status: 'processing',
    progress: 0,
    currentFrame: 0,
    error: null,
    isPaused: false,
    logs: [],
  }),
  
  stopProcessing: () => set({ status: 'stopping' }),
  
  pauseProcessing: () => set({ isPaused: true }),
  
  resumeProcessing: () => set({ isPaused: false }),
  
  completeProcessing: () => set({ status: 'completed', isPaused: false }),
  
  failProcessing: (error) => set({ status: 'error', error, isPaused: false }),
  
  resetProcessing: () => set({
    status: 'idle',
    progress: 0,
    currentFrame: 0,
    totalFrames: 0,
    fps: 0,
    error: null,
    isPaused: false,
    logs: [],
  }),
  
  updateProgress: (progress) => set({ progress }),
  
  setFrameInfo: (currentFrame, totalFrames) => set({ currentFrame, totalFrames }),
  
  setPerformance: (fps, elapsed, eta) => set({ fps, elapsed, eta }),
  
  addLog: (log) => set((state) => ({
    logs: [...state.logs, log].slice(-500),
  })),
  
  clearLogs: () => set({ logs: [] }),
  
  setError: (error) => set({ error, status: 'error' }),
}));
