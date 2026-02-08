import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type TabId = 
  | 'general' 
  | 'objdetect'
  | 'upscale' 
  | 'imageupscale'
  | 'interpolate' 
  | 'restore' 
  | 'scene' 
  | 'dedup' 
  | 'segment' 
  | 'depth' 
  | 'encoding'
  | 'performance';

interface NavigationState {
  activeTab: TabId;
  sidebarCollapsed: boolean;
}

interface NavigationStore extends NavigationState {
  setActiveTab: (tab: TabId) => void;
  toggleSidebar: () => void;
}

export const useNavigationStore = create<NavigationStore>()(
  persist(
    (set) => ({
      activeTab: 'general',
      sidebarCollapsed: false,
      setActiveTab: (tab) => set({ activeTab: tab }),
      toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
    }),
    {
      name: 'tas-navigation',
    }
  )
);