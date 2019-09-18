import { ComboBox, IComboBox, IComboBoxOption } from "office-ui-fabric-react/lib/ComboBox";
import React from "react";
import { AccessibleChart } from "../../../ChartTools";
import { localization } from "../../../Localization/localization";
import { IPlotlyProperty } from "../../../Shared";
import { FabricStyles } from "../../FabricStyles";
import {  ScatterUtils, IScatterProps } from "./ScatterUtils";
import _ from "lodash";
import { NoDataMessage, LoadingSpinner } from "../../SharedComponents";
require('./Scatter.css');

export const DataScatterId = 'data_scatter_id';

export class DataExploration extends React.PureComponent<IScatterProps> {
    private plotlyProps: IPlotlyProperty;

    constructor(props: IScatterProps) {
        super(props);
        this.onXSelected = this.onXSelected.bind(this);
        this.onYSelected = this.onYSelected.bind(this);
        this.onColorSelected = this.onColorSelected.bind(this);
        this.onDismiss = this.onDismiss.bind(this);
    }

    public render(): React.ReactNode {
        if (this.props.dashboardContext.explanationContext.testDataset) {
            const projectedData = ScatterUtils.projectData(this.props.dashboardContext.explanationContext);
            this.plotlyProps = this.props.plotlyProps !== undefined ?
                _.cloneDeep(this.props.plotlyProps) :
                ScatterUtils.defaultDataExpPlotlyProps(this.props.dashboardContext.explanationContext);
            const dropdownOptions = ScatterUtils.buildOptions(this.props.dashboardContext.explanationContext, false);
            const initialColorOption = ScatterUtils.getselectedColorOption(this.plotlyProps, dropdownOptions);
            return (
                <div className="explanation-chart">
                    <div className="top-controls">
                        <div className="path-selector x-value">
                            <ComboBox
                                options={dropdownOptions}
                                onChange={this.onXSelected}
                                label={localization.ExplanationScatter.xValue}
                                ariaLabel="x picker"
                                selectedKey={this.plotlyProps.data[0].xAccessor}
                                useComboBoxAsMenuWidth={true}
                                styles={ScatterUtils.xStyle}
                            />
                        </div>
                        <div className="path-selector">
                            <ComboBox
                                options={dropdownOptions}
                                onChange={this.onColorSelected}
                                label={localization.ExplanationScatter.colorValue}
                                ariaLabel="color picker"
                                selectedKey={initialColorOption}
                                useComboBoxAsMenuWidth={true}
                                styles={FabricStyles.defaultDropdownStyle}
                            />
                        </div>
                    </div>
                    <div className="top-controls">
                        <div className="path-selector y-value">
                            <ComboBox
                                options={dropdownOptions}
                                onChange={this.onYSelected}
                                label={localization.ExplanationScatter.yValue}
                                ariaLabel="y picker"
                                selectedKey={this.plotlyProps.data[0].yAccessor}
                                useComboBoxAsMenuWidth={true}
                                styles={ScatterUtils.yStyle}
                            />
                        </div>
                    </div>
                    <AccessibleChart
                        plotlyProps={ScatterUtils.populatePlotlyProps(projectedData, _.cloneDeep(this.plotlyProps))}
                        sharedSelectionContext={this.props.selectionContext}
                        theme={this.props.theme}
                    />
                </div>
            );
        }
        const explanationStrings = this.props.messages ? this.props.messages.TestReq : undefined;
        return <NoDataMessage explanationStrings={explanationStrings}/>;
    }

    private onXSelected(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        ScatterUtils.updateNewXAccessor(this.props, this.plotlyProps, item, DataScatterId);
    }

    private onYSelected(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        ScatterUtils.updateNewYAccessor(this.props, this.plotlyProps, item, DataScatterId);
    }

    // Color is done in one of two ways: if categorical, we set the groupBy property, creating a series per class
    // If it is numeric, we set the color property and display a colorbar. when setting one, clear the other.
    private onColorSelected(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        ScatterUtils.updateColorAccessor(this.props, this.plotlyProps, item, DataScatterId);
    }

    private onDismiss(): void {
        this.setState({ isCalloutVisible: false });
    }
}