import type { SettingsState } from '../stores/settingsStore';

export interface ProcessingArgsOptions {
  /** Override --output path (used for preview segment output). */
  outputOverride?: string;
  /** --inpoint (seconds) */
  inpoint?: number;
  /** --outpoint (seconds) */
  outpoint?: number;
}

export function buildPreviewSegmentOutputPath(
  baseOutputPath: string,
  startS: number,
  durationS: number
): string {
  const safeStart = Math.max(0, Math.floor(Number(startS) || 0));
  const safeDur = Math.max(1, Math.floor(Number(durationS) || 1));

  const dot = baseOutputPath.lastIndexOf('.');
  if (dot > 0) {
    const base = baseOutputPath.slice(0, dot);
    const ext = baseOutputPath.slice(dot);
    return `${base}_preview_${safeStart}s_${safeDur}s${ext}`;
  }

  return `${baseOutputPath}_preview_${safeStart}s_${safeDur}s`;
}

export function buildProcessingArgs(
  settings: SettingsState,
  opts: ProcessingArgsOptions = {}
): string[] {
  const args: string[] = [];

  // Input/Output
  if (settings.youtubeUrl) {
    args.push('--input', settings.youtubeUrl);
  } else {
    args.push('--input', settings.inputPath);
  }

  const outputPath = opts.outputOverride ?? settings.outputPath;
  if (outputPath) args.push('--output', outputPath);

  if (settings.benchmark) args.push('--benchmark');

  if (typeof opts.inpoint === 'number') args.push('--inpoint', String(opts.inpoint));
  if (typeof opts.outpoint === 'number') args.push('--outpoint', String(opts.outpoint));

  // Upscale
  if (settings.upscaleEnabled) {
    args.push('--upscale');
    args.push('--upscale_factor', String(settings.upscaleFactor));
    if (settings.upscaleMethod === 'custom') {
      if (!settings.upscaleCustomModel) {
        throw new Error('Upscale method is set to custom, but no custom upscale model was selected.');
      }
      // CLI accepts a model path as --upscale_method.
      args.push('--upscale_method', settings.upscaleCustomModel);
      if (settings.upscaleCustomBackend !== 'default') {
        args.push('--custom_upscale_backend', settings.upscaleCustomBackend);
      }
    } else {
      args.push('--upscale_method', settings.upscaleMethod);
    }
  }

  // Interpolate
  if (settings.interpolateEnabled) {
    args.push('--interpolate');
    args.push('--interpolate_factor', String(settings.interpolateFactor));
    if (settings.interpolateMethod === 'custom') {
      if (!settings.interpolateCustomModel) {
        throw new Error('Interpolate method is set to custom, but no custom interpolate model was selected.');
      }
      args.push('--interpolate_method', settings.interpolateCustomModel);
      if (settings.interpolateCustomBackend !== 'default') {
        args.push('--custom_interpolate_backend', settings.interpolateCustomBackend);
      }
    } else {
      args.push('--interpolate_method', settings.interpolateMethod);
    }
    if (settings.ensemble) args.push('--ensemble');
    if (settings.dynamicScale) args.push('--dynamic_scale');
    if (settings.slowmo) args.push('--slowmo');
  }

  // Restore
  if (settings.restoreEnabled) {
    args.push('--restore');
    const rawRestore = settings.restoreChain.length > 0 ? settings.restoreChain : [settings.restoreMethod];
    const hasCustom = rawRestore.includes('custom');
    const mappedRestore = rawRestore.map((m) => {
      if (m !== 'custom') return m;
      if (!settings.restoreCustomModel) {
        throw new Error('Restore method chain includes custom, but no custom restore model was selected.');
      }
      return settings.restoreCustomModel;
    });
    args.push('--restore_method', ...mappedRestore);

    if (hasCustom && settings.restoreCustomBackend !== 'default') {
      args.push('--custom_restore_backend', settings.restoreCustomBackend);
    }
    if (settings.sharpenEnabled) {
      args.push('--sharpen');
      args.push('--sharpen_sens', String(settings.sharpenSens));
    }
  }

  // Scene Detection
  if (settings.scenechangeEnabled) {
    args.push('--scenechange');
    if (settings.useCustomScenechangeModel && settings.scenechangeCustomModel) {
      args.push('--scenechange_method', settings.scenechangeCustomModel);
    } else {
      args.push('--scenechange_method', settings.scenechangeMethod);
    }
    args.push('--scenechange_sens', String(settings.scenechangeSens / 100));
    if (settings.autoclip) args.push('--autoclip');
  }

  // Deduplication
  if (settings.dedupEnabled) {
    args.push('--dedup');
    if (settings.useCustomDedupModel && settings.dedupCustomModel) {
      args.push('--dedup_method', settings.dedupCustomModel);
    } else {
      args.push('--dedup_method', settings.dedupMethod);
    }
    args.push('--dedup_sens', String(settings.dedupSens / 100));
  }

  // Segmentation
  if (settings.segmentEnabled) {
    args.push('--segment');
    if (settings.segmentMethod === 'custom') {
      if (!settings.segmentCustomModel) {
        throw new Error('Segment method is set to custom, but no custom segment model was selected.');
      }
      args.push('--segment_method', settings.segmentCustomModel);
    } else {
      args.push('--segment_method', settings.segmentMethod);
    }
  }

  // Depth
  if (settings.depthEnabled) {
    args.push('--depth');
    if (settings.depthMethod === 'custom') {
      if (!settings.depthCustomModel) {
        throw new Error('Depth method is set to custom, but no custom depth model was selected.');
      }
      args.push('--depth_method', settings.depthCustomModel);
    } else {
      args.push('--depth_method', settings.depthMethod);
    }
    args.push('--depth_quality', settings.depthQuality);
  }

  // Object Detection
  if (settings.objDetectEnabled) {
    args.push('--obj_detect');
    args.push('--obj_detect_method', settings.objDetectMethod);
    if (settings.objDetectDisableAnnotations) {
      args.push('--obj_detect_disable_annotations', 'true');
    }
  }

  // Encoding
  args.push('--encode_method', settings.encodeMethod);
  args.push('--bit_depth', settings.bitDepth);
  if (settings.resizeEnabled) {
    args.push('--resize');
    args.push('--resize_factor', String(settings.resizeFactor));
  }

  // Performance
  args.push('--half', String(settings.half));
  args.push('--decode_method', settings.decodeMethod);
  args.push('--compile_mode', settings.compileMode);
  if (settings.static) args.push('--static');
  if (settings.tileRendering) {
    args.push('--tile_rendering');
    args.push('--tile_size', String(settings.tileSize));
  }
  if (settings.nvdecCompat) args.push('--nvdec_compat');

  // Preview
  if (settings.livePreview) args.push('--preview');

  return args;
}
