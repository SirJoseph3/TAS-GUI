import React from 'react';
import { useNavigationStore } from '../stores/navigationStore';
import { GeneralTab } from './tabs/GeneralTab';

import { UpscaleTab } from './tabs/UpscaleTab';
import { InterpolateTab } from './tabs/InterpolateTab';
import { ObjectDetectionTab } from './tabs/ObjectDetectionTab';
import { ImageUpscaleTab } from './tabs/ImageUpscaleTab';
import { RestoreTab } from './tabs/RestoreTab';
import { SceneTab } from './tabs/SceneTab';
import { DedupTab } from './tabs/DedupTab';
import { SegmentTab } from './tabs/SegmentTab';
import { DepthTab } from './tabs/DepthTab';
import { EncodingTab } from './tabs/EncodingTab';
import { PerformanceTab } from './tabs/PerformanceTab';
import { ProcessingPanel } from './ProcessingPanel';

const tabComponents: Record<string, React.FC> = {
  general: GeneralTab,

  upscale: UpscaleTab,
  interpolate: InterpolateTab,
  objdetect: ObjectDetectionTab,
  imageupscale: ImageUpscaleTab,
  restore: RestoreTab,
  scene: SceneTab,
  dedup: DedupTab,
  segment: SegmentTab,
  depth: DepthTab,
  encoding: EncodingTab,
  performance: PerformanceTab,
};

export const MainContent: React.FC = () => {
  const { activeTab } = useNavigationStore();
  const TabComponent = tabComponents[activeTab] || GeneralTab;

  return (
    <div className="flex-1 flex overflow-hidden">
      {/* Settings Content */}
      <div className="flex-1 overflow-auto p-6">
        <TabComponent />
      </div>
      
      {/* Processing Panel (Right Side) */}
      <ProcessingPanel />
    </div>
  );
};
