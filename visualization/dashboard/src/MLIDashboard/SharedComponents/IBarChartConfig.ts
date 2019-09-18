export enum FeatureKeys {
    absoluteGlobal = 'absoluteGlobal',
    absoluteLocal = 'absoluteLocal' 
}

export type FeatureSortingKey = number | FeatureKeys;

export interface IBarChartConfig {
    topK: number;
    sortingKey?: FeatureSortingKey;
    defaultVisibleClasses?: number[];
}