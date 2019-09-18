import IStringsParam from "./IStringsParam";

// This is the interface of the data to be provided by any glue code, be it the ModelExplanationController, the Jupyter widget,
// or some future extension. The Explanation Dashboard opperates on this data object, and an optional chart config that specifies
// configurable view information that is orthogonal to this data.
/**
 * @typedef {Object} IKernelExplanationData
 * @property {any[][]} [trainingData] - sample dataset. Dim(rows) x Dim(features)
 * @property {string[]} [featureNames] - pretty-print names for the feature columns. Dim(features)
 * @property {string[]} [classNames] - pretty-print names for the classes. Dim(classes)
 * @property {number[][] | number[][][]} [localExplanations] - local explanations for sample data. [Dim(classes)] x Dim(rows) x Dim(features)
 * @property {number[]} globalExplanation - global explanation data averaged across classes. Dim(features)
 * @property {Array<number | string>} [trueY] - true values for sample data output. Dim(rows)
 * @property {Array<number | string>} [predictedY] - model outputs for sample dataset. Dim(rows)
 * @property {number[][] | number[]} [probabilityY] - model probabilities for output values. Dim(rows) x [Dim(classes)]
 */

export interface IExplanationDashboardProps {
    modelInformation: IModelInformation;
    dataSummary: IDatasetSummary;
    testData?: any[][];
    predictedY?: number[];
    probabilityY?: number[][];
    trueY?: number[];
    precomputedExplanations?: IPrecomputedExplanations;
    theme?: any;
    stringParams?: IStringsParam;
    requestPredictions?: (request: any[], abortSignal: AbortSignal) => Promise<any[]>;
    requestLocalFeatureExplanations?: (request: any[], abortSignal: AbortSignal, explanationAlgorithm?: string) => Promise<any[]>;
}

export interface IModelInformation {
    modelClass: 'Tree' | 'EBM' | 'blackbox';
    method: 'classifier' | 'regressor'
}

export interface IDatasetSummary {
    featureNames?: string[];
    classNames?: string[];
    categoricalMap?: {[key: number]: string[]};
}

export interface IPrecomputedExplanations {
    localFeatureImportance?: LocalFeatureImportance;
    globalFeatureImportance?: GlobalFeatureImportance;
}

export type LocalFeatureImportance = number[][][] | number[][];

export type GlobalFeatureImportance = number[][] | number[];