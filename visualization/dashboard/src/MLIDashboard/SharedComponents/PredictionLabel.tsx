import React from "react";
import { localization } from "../../Localization/localization";
import { ModelTypes } from "../IExplanationContext";
require('./PredictionLabel.css');

export interface IPredictionLabelProps {
    modelType: ModelTypes;
    prediction: number;
    classNames: string[];
    predictedProbabilities?: number[];
}

export class PredictionLabel extends  React.Component<IPredictionLabelProps> {
    public render(): React.ReactNode {
        return (
        <div className="prediction-area">
            <div className="prediction-label">{
                this.makePredictionLabel()
            }</div>
            { (this.props.predictedProbabilities !== undefined) && <div className="probability-label">{
                this.makeProbabilityLabel()
            }</div>}
        </div>)
    }

    private makePredictionLabel(): string {
        if (this.props.modelType === ModelTypes.regression) {
            return localization.formatString(localization.PredictionLabel.predictedValueLabel, this.props.prediction.toLocaleString(undefined, {minimumFractionDigits: 3})) as string;
        }
        return localization.formatString(localization.PredictionLabel.predictedClassLabel, this.props.classNames[this.props.prediction]) as string;
    }

    private makeProbabilityLabel(): string {
        const probability = this.props.predictedProbabilities[this.props.prediction];
        return localization.formatString(localization.IcePlot.probabilityLabel, probability.toLocaleString(undefined, {minimumFractionDigits: 3})) as string;
    }
}