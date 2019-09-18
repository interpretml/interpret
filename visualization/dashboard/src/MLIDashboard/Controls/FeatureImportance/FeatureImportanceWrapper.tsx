import React from "react";
import { SelectionContext } from "../../../Shared";
import { IDashboardContext } from "../../ExplanationDashboard";
import { IBarChartConfig } from "../../SharedComponents";
import { FeatureImportanceBar } from "./FeatureImportanceBar";
import { Beehive } from "./Beenhive";
import { IHelpMessage, HelpMessageDict } from "../../Interfaces/IStringsParam";
import { Violin } from "./Violin";
import { IComboBoxOption } from "office-ui-fabric-react/lib/ComboBox";
import { localization } from "../../../Localization/localization";

export const BarId = 'bar_id';
export const GlobalFeatureImportanceId = 'global_feature_id';

export enum FeatureImportanceModes {
    bar = 'bar',
    beehive = 'beehive',
    violin = 'violin',
    box = 'box'
}

export interface IFeatureImportanceConfig extends IBarChartConfig {
    displayMode: FeatureImportanceModes;
    id: string;
}

export interface IGlobalFeatureImportanceProps {
    dashboardContext: IDashboardContext;
    config: IFeatureImportanceConfig;
    selectionContext: SelectionContext;
    chartTypeOptions?: IComboBoxOption[];
    messages?: HelpMessageDict;
    theme?: string;
    onChange: (config: IFeatureImportanceConfig, id: string) => void;
}

export class FeatureImportanceWrapper extends React.PureComponent<IGlobalFeatureImportanceProps> {

    public render(): React.ReactNode {
        const chartTypeOptions: IComboBoxOption[] = this.props.dashboardContext.explanationContext.localExplanation !== undefined ?
        [
            {text: localization.FeatureImportanceWrapper.boxText, key: FeatureImportanceModes.box},
            {text: localization.FeatureImportanceWrapper.beehiveText, key: FeatureImportanceModes.beehive},
            {text: localization.FeatureImportanceWrapper.violinText, key: FeatureImportanceModes.violin}
        ] :
        [];

        switch (this.props.config.displayMode) {
            case FeatureImportanceModes.bar:
                return <FeatureImportanceBar
                    {...this.props}
                    chartTypeOptions={chartTypeOptions}
                />;
            case FeatureImportanceModes.beehive:
                return <Beehive
                    {...this.props}
                    chartTypeOptions={chartTypeOptions}
                />;
            case FeatureImportanceModes.violin:
            case FeatureImportanceModes.box:
                return <Violin
                    {...this.props}
                    chartTypeOptions={chartTypeOptions}
                />;
        }
    }
}