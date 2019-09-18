import React from "react";
import { ScatterUtils, IScatterProps } from "./ScatterUtils";
import { Callout } from "office-ui-fabric-react/lib/Callout";
import { localization } from "../../../Localization/localization";
import { DefaultButton, IconButton } from "office-ui-fabric-react/lib/Button";
import { AccessibleChart } from "../../../ChartTools";
import { FabricStyles } from "../../FabricStyles";
import { ComboBox, IComboBox, IComboBoxOption } from "office-ui-fabric-react/lib/ComboBox";
import _ from "lodash";
import { IPlotlyProperty } from "../../../Shared";
import { ModelTypes } from "../../IExplanationContext";
import { LoadingSpinner, NoDataMessage } from "../../SharedComponents";
require('./Scatter.css');

export const ExplanationScatterId = 'explanation_scatter_id';

export interface IExplanationExplorationState {
    isCalloutVisible: boolean;
}

export class ExplanationExploration extends React.PureComponent<IScatterProps, IExplanationExplorationState> {
    private readonly iconId = "data-exploration-help-icon1";
    private plotlyProps: IPlotlyProperty;

    constructor(props: IScatterProps) {
        super(props);
        this.state = { isCalloutVisible: false};
        this.onXSelected = this.onXSelected.bind(this);
        this.onYSelected = this.onYSelected.bind(this);
        this.onColorSelected = this.onColorSelected.bind(this);
        this.onDismiss = this.onDismiss.bind(this);
        this.onIconClick = this.onIconClick.bind(this);
    }

    public render(): React.ReactNode {
        if (this.props.dashboardContext.explanationContext.testDataset
            && this.props.dashboardContext.explanationContext.localExplanation
            && this.props.dashboardContext.explanationContext.localExplanation.values) {
            const projectedData = ScatterUtils.projectData(this.props.dashboardContext.explanationContext);
            this.plotlyProps = this.props.plotlyProps !== undefined ?
                _.cloneDeep(this.props.plotlyProps) :
                ScatterUtils.defaultExplanationPlotlyProps(this.props.dashboardContext.explanationContext);
            const dropdownOptions = ScatterUtils.buildOptions(this.props.dashboardContext.explanationContext, true);
            const initialColorOption = ScatterUtils.getselectedColorOption(this.plotlyProps, dropdownOptions);
            const weightContext = this.props.dashboardContext.weightContext;
            const includeWeightDropdown = this.props.dashboardContext.explanationContext.modelMetadata.modelType === ModelTypes.multiclass;
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
                        {(includeWeightDropdown) && 
                        <div className="selector">
                            <div className="selector-label">
                                <div className="label-text">{localization.CrossClass.label}</div>
                                <IconButton
                                    id={this.iconId}
                                    iconProps={{ iconName: 'Info' }}
                                    title={localization.CrossClass.info}
                                    ariaLabel="Info"
                                    onClick={this.onIconClick}
                                    styles={{ root: { marginBottom: -3, color: 'rgb(0, 120, 212)' } }}
                                />
                            </div>
                            <ComboBox
                                selectedKey={weightContext.selectedKey}
                                onChange={weightContext.onSelection}
                                options={weightContext.options}
                                ariaLabel={"Cross-class weighting selector"}
                                useComboBoxAsMenuWidth={true}
                                styles={FabricStyles.defaultDropdownStyle}
                            />
                        </div>}
                    </div>
                    {this.state.isCalloutVisible && (
                    <Callout
                        target={'#' + this.iconId}
                        setInitialFocus={true}
                        onDismiss={this.onDismiss}
                        role="alertdialog">
                        <div className="callout-info">
                            <div className="class-weight-info">
                                <span>{localization.CrossClass.overviewInfo}</span> 
                                <ul>
                                    <li>{localization.CrossClass.absoluteValInfo}</li>
                                    <li>{localization.CrossClass.predictedClassInfo}</li>
                                    <li>{localization.CrossClass.enumeratedClassInfo}</li>
                                </ul>
                            </div>
                            <DefaultButton onClick={this.onDismiss}>{localization.CrossClass.close}</DefaultButton>
                        </div>
                    </Callout>
                    )}
                    <AccessibleChart
                        plotlyProps={ScatterUtils.populatePlotlyProps(projectedData, _.cloneDeep(this.plotlyProps))}
                        sharedSelectionContext={this.props.selectionContext}
                        theme={this.props.theme}
                    />
                </div>
            );
        }
        if (this.props.dashboardContext.explanationContext.localExplanation &&
            this.props.dashboardContext.explanationContext.localExplanation.percentComplete !== undefined) {
            return <LoadingSpinner/>
        }
        const explanationStrings = this.props.messages ? this.props.messages.LocalExpAndTestReq : undefined;
        return <NoDataMessage explanationStrings={explanationStrings}/>;
    }

    private onXSelected(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        ScatterUtils.updateNewXAccessor(this.props, this.plotlyProps, item, ExplanationScatterId);
    }

    private onYSelected(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        ScatterUtils.updateNewYAccessor(this.props, this.plotlyProps, item, ExplanationScatterId);
    }

    // Color is done in one of two ways: if categorical, we set the groupBy property, creating a series per class
    // If it is numeric, we set the color property and display a colorbar. when setting one, clear the other.
    private onColorSelected(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        ScatterUtils.updateColorAccessor(this.props, this.plotlyProps, item, ExplanationScatterId);
    }

    private onIconClick(): void {
        this.setState({isCalloutVisible: !this.state.isCalloutVisible});
    }

    private onDismiss(): void {
        this.setState({ isCalloutVisible: false });
    }
}