import { IModelMetadata } from "../Shared/IModelMetadata";

export enum ModelTypes {
    regression = 'regression',
    binary = 'binary',
    multiclass = 'multiclass'
}

export interface IExplanationContext {
    modelMetadata: IExplanationModelMetadata;
    explanationGenerators: IExplanationGenerators;
    localExplanation?: ILocalExplanation;
    testDataset?: ITestDataset;
    globalExplanation?: IGlobalExplanation;
    isGlobalDerived: boolean;
}

// The interface containing either the local explanations matrix, 
// or information on the fetcing of the local explanation. 
export interface ILocalExplanation {
    values?: number[][][];
    flattenedValues?: number[][];
    percentComplete?: number;
}

// The Global explanation. Either derived from local, or passed in as independent prop.
// User provided shall take precidence over our computed. Features x Class
export interface IGlobalExplanation {
    perClassFeatureImportances?: number[][];
    flattenedFeatureImportances?: number[];
}

export interface IExplanationGenerators {
    requestPredictions?: (request: any[], abortSignal: AbortSignal) => Promise<any[]>;
    requestLocalFeatureExplanations?: (request: any[], abortSignal: AbortSignal, explanationAlgorithm?: string) => Promise<any[]>;
}

export interface ITestDataset {
    dataset: any[][];
    predictedY: number[];
    probabilityY?: number[][];
    trueY?: number[];
}

export interface IExplanationModelMetadata extends IModelMetadata {
    modelType: ModelTypes;
    explainerType?: string;
}