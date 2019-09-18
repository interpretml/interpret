import _ from "lodash";
import React from "react";
import { localization } from "../../Localization/localization";
import { FeatureEditingTile, PredictionLabel, NoDataMessage } from "../SharedComponents";
import { ICategoricalRange } from "../../Shared/ICategoricalRange";
import { IExplanationContext } from "../IExplanationContext";
import { HelpMessageDict } from "../Interfaces";

require('./PerturbationExploration.css');

export interface IPerturbationExplorationProps {
    invokeModel?: (data: any[], abortSignal: AbortSignal) => Promise<any[]>;
    datapointIndex: number;
    explanationContext: IExplanationContext;
    messages?: HelpMessageDict;
    theme?: string;
}

export interface IPerturbationExplorationState {
    // the dictionary of edited values, keyed on the feature inedex
    perturbedDictionary: {[key: number]: any};
    // the dictionary of validation errors
    featureErrors: boolean[];
    prediction?: number;
    predictionProbabilities?: number[];
    abortController: AbortController | undefined;
    errorMessage?: string;
}

export class PerturbationExploration extends  React.Component<IPerturbationExplorationProps, IPerturbationExplorationState> {

    constructor(props: IPerturbationExplorationProps) {
        super(props);
        this.state = {
            perturbedDictionary: {},
            featureErrors: new Array(props.explanationContext.modelMetadata.featureNames.length),
            abortController: undefined
        };
        this.onValueEdit = this.onValueEdit.bind(this);
        this.fetchData = _.debounce(this.fetchData.bind(this), 500);
    }

    public componentDidUpdate(prevProps: IPerturbationExplorationProps): void {
        if (this.props.datapointIndex !== prevProps.datapointIndex) {
            if (this.state.abortController) {
                this.state.abortController.abort();
            }
            this.setState({
                perturbedDictionary: {},
                featureErrors: new Array(this.props.explanationContext.modelMetadata.featureNames.length),
                abortController: undefined,
                prediction: undefined,
                predictionProbabilities: undefined
            })
        }
    }

    public render(): React.ReactNode {
        if (this.props.invokeModel === undefined) {
            const explanationStrings = this.props.messages ? this.props.messages.PredictorReq : undefined;
            return <NoDataMessage explanationStrings={explanationStrings}/>;
        }
        const hasErrors = this.state.featureErrors.some(val => val);
        return (
            <div className="flex-wrapper">
                <div className="label-group">
                    <div className="label-group-label">Base:</div>
                    <div className="flex-full">
                        <PredictionLabel
                            prediction={this.props.explanationContext.testDataset.predictedY[this.props.datapointIndex]}
                            classNames={this.props.explanationContext.modelMetadata.classNames}
                            modelType={this.props.explanationContext.modelMetadata.modelType}
                            predictedProbabilities={this.props.explanationContext.testDataset.probabilityY ? 
                                this.props.explanationContext.testDataset.probabilityY[this.props.datapointIndex] :  undefined}
                        />
                    </div>
                </div>
                {(this.state.abortController && !hasErrors) && <div className="loading-message">{localization.PerturbationExploration.loadingMessage}</div>}
                {(this.state.errorMessage) && <div className="loading-message"> {this.state.errorMessage}</div>}
                {(hasErrors) && <div className="loading-message">{localization.IcePlot.topLevelErrorMessage}</div>}
                {(!hasErrors && this.state.prediction !== undefined && this.state.abortController === undefined) && <div className="label-group">
                    <div className="label-group-label">{localization.PerturbationExploration.perturbationLabel}</div>
                    <div className="flex-full">
                        <PredictionLabel
                            prediction={this.state.prediction}
                            classNames={this.props.explanationContext.modelMetadata.classNames}
                            modelType={this.props.explanationContext.modelMetadata.modelType}
                            predictedProbabilities={this.state.predictionProbabilities}
                        />
                    </div>
                </div>}
                <div className="tile-scroller">
                    {_.cloneDeep(this.props.explanationContext.testDataset.dataset[this.props.datapointIndex]).map((featureValue, featureIndex) => {
                        return (<FeatureEditingTile 
                            key={featureIndex}
                            index={featureIndex}
                            featureName={this.props.explanationContext.modelMetadata.featureNames[featureIndex]}
                            defaultValue={featureValue}
                            onEdit={this.onValueEdit}
                            enumeratedValues={(this.props.explanationContext.modelMetadata.featureRanges[featureIndex] as ICategoricalRange).uniqueValues}
                            rangeType={this.props.explanationContext.modelMetadata.featureRanges[featureIndex].rangeType}
                        />);
                    })}
                </div>
            </div>
        );
    }

    private onValueEdit(featureIndex: number, val: string | number, error?: string): void {
        const perturbedDictionary = _.cloneDeep(this.state.perturbedDictionary);
        const featureErrors = _.cloneDeep(this.state.featureErrors);
        featureErrors[featureIndex] = error !== undefined;
        // unset in the case that the user reverts.
        if (val === this.props.explanationContext.testDataset.dataset[this.props.datapointIndex][featureIndex]) {
            perturbedDictionary[featureIndex] = undefined;
        }
        else {
            perturbedDictionary[featureIndex] = val;
        }
        this.setState({perturbedDictionary, featureErrors}, () => { 
            this.fetchData();
        });
    }

    private fetchData(): void {
        if (this.state.abortController !== undefined) {
            this.state.abortController.abort();
        }
        // skip if there are any errors.
        if (this.state.featureErrors.some(val => val)) {
            return;
        }
        const data = _.cloneDeep(this.props.explanationContext.testDataset.dataset[this.props.datapointIndex]);
        for (var key of Object.keys(this.state.perturbedDictionary)) {
            const index = +key;
            if (this.state.perturbedDictionary[index] !== undefined) {
                data[index] = this.state.perturbedDictionary[index];
            }
        }
        const abortController = new AbortController();
        const promise = this.props.invokeModel([data], abortController.signal);
        this.setState({abortController, errorMessage: undefined}, async () => {
            try {
                const fetchedData = await promise;
                if (abortController.signal.aborted) {
                    return;
                }
                if (Array.isArray(fetchedData[0])) {
                    const predictionVector = fetchedData[0];
                    let predictedClass = 0;
                    let maxProb = predictionVector[0];
                    for (let i = 1; i < predictionVector.length; i++) {
                        if (predictionVector[i] > maxProb) {
                            predictedClass = i;
                            maxProb = predictionVector[i];
                        }
                    }
                    this.setState({prediction: predictedClass, predictionProbabilities: predictionVector, abortController: undefined});
                } else {
                    this.setState({prediction: fetchedData[0], abortController: undefined});
                }
            } catch(err) {
                if (err.name === 'AbortError') {
                    return;
                }
                if (err.name === 'PythonError') {
                    this.setState({errorMessage: localization.formatString(localization.IcePlot.errorPrefix, err.message) as string})
                }
            }
        });
    }
}