
/**
 * Think state as global in the scope of ContFeature.svelte
 */
export let state = {
  curXScale: null,
  curYScale: null,
  curTransform: null,
  selectedInfo: null,
  pointData: null,
  additiveData: null,
  pointDataBuffer: null,
  additiveDataBuffer: null,
  oriXScale: null,
  oriYScale: null,
  bboxPadding: 5,
};