import _ from "lodash";
import { ComboBox, IComboBox, IComboBoxOption } from "office-ui-fabric-react/lib/ComboBox";
import { IDropdownOption } from "office-ui-fabric-react/lib/Dropdown";
import { Slider } from "office-ui-fabric-react/lib/Slider";
import React from "react";
import { localization } from "../../../Localization/localization";
import { FabricStyles } from "../../FabricStyles";
import { ModelTypes } from "../../IExplanationContext";
import { ModelExplanationUtils } from "../../ModelExplanationUtils";
import { NoDataMessage, LoadingSpinner, FeatureKeys, FeatureSortingKey, BarChart } from "../../SharedComponents";
import { IGlobalFeatureImportanceProps } from "./FeatureImportanceWrapper";
import { Callout } from "office-ui-fabric-react/lib/Callout";
import { DefaultButton, IconButton } from "office-ui-fabric-react/lib/Button";

require('./FeatureImportanceBar.css');

export interface IFeatureImportanceBarState {
    selectedSorting: FeatureSortingKey;
    isCalloutVisible: boolean;
}

export class FeatureImportanceBar extends React.PureComponent<IGlobalFeatureImportanceProps, IFeatureImportanceBarState> {
    private sortOptions: IDropdownOption[];
    private readonly _iconId = 'icon-id';

    constructor(props:IGlobalFeatureImportanceProps) {
        super(props);
        this.sortOptions = this.buildSortOptions();
        this.onSortSelect = this.onSortSelect.bind(this);
        this.setTopK = this.setTopK.bind(this);
        this.setChart = this.setChart.bind(this);
        this.onDismiss = this.onDismiss.bind(this);
        this.onIconClick = this.onIconClick.bind(this);
        this.state = {
            selectedSorting: FeatureKeys.absoluteGlobal,
            isCalloutVisible: false
        };
    }

    public render(): React.ReactNode {
        const expContext = this.props.dashboardContext.explanationContext;
        const globalExplanation = expContext.globalExplanation
        if (globalExplanation !== undefined && 
            (globalExplanation.flattenedFeatureImportances !== undefined ||
                globalExplanation.perClassFeatureImportances !== undefined)) {
            const featuresByClassMatrix = this.getFeatureByClassMatrix();
            let sortVector = this.getSortVector(featuresByClassMatrix);

            return (
                <div className="feature-bar-explanation-chart">
                    <div className="top-controls">
                        {(this.props.chartTypeOptions && this.props.chartTypeOptions.length > 1) &&<ComboBox
                            label={localization.FeatureImportanceWrapper.chartType}
                            className="pathSelector"
                            selectedKey={this.props.config.displayMode}
                            onChange={this.setChart}
                            options={this.props.chartTypeOptions}
                            ariaLabel={"chart type picker"}
                            useComboBoxAsMenuWidth={true}
                            styles={FabricStyles.smallDropdownStyle}
                        />}
                        <div className="slider-control">
                            <div className="slider-label">
                                <span className="label-text">{localization.AggregateImportance.topKFeatures}</span>
                                <IconButton
                                    id={this._iconId}
                                    iconProps={{ iconName: 'Info' }}
                                    title={localization.CrossClass.info}
                                    ariaLabel="Info"
                                    onClick={this.onIconClick}
                                    styles={{ root: { marginBottom: -3, color: 'rgb(0, 120, 212)' } }}
                                />
                            </div>
                            <Slider
                                className="feature-slider"
                                max={Math.min(30, expContext.modelMetadata.featureNames.length)}
                                min={1}
                                step={1}
                                value={this.props.config.topK}
                                onChange={this.setTopK}
                                showValue={true}
                            />
                        </div>
                        {(this.sortOptions.length > 0) && <ComboBox
                                className="pathSelector"
                                label={localization.BarChart.sortBy}
                                selectedKey={this.state.selectedSorting}
                                onChange={this.onSortSelect}
                                options={this.sortOptions}
                                ariaLabel={"sort selector"}
                                useComboBoxAsMenuWidth={true}
                                styles={FabricStyles.smallDropdownStyle}
                            />}
                    </div>
                    {this.state.isCalloutVisible && (
                        <Callout
                            target={'#' + this._iconId}
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
                    <BarChart 
                        featureByClassMatrix={featuresByClassMatrix}
                        sortedIndexVector={sortVector}
                        topK={this.props.config.topK}
                        modelMetadata={expContext.modelMetadata}
                        barmode='stack'
                    />
                </div>
            )

        }
        if (expContext.localExplanation && expContext.localExplanation.percentComplete !== undefined) {
            return <LoadingSpinner/>;
        }
        
        const explanationStrings = this.props.messages ? this.props.messages.LocalOrGlobalAndTestReq: undefined;
        return <NoDataMessage explanationStrings={explanationStrings}/>;
    }

    private getSortVector(featureByClassMatrix: number[][]): number[] {
        if (this.state.selectedSorting === FeatureKeys.absoluteGlobal) {
            return ModelExplanationUtils.buildSortedVector(featureByClassMatrix);
        }
        return ModelExplanationUtils.buildSortedVector(featureByClassMatrix, this.state.selectedSorting as number);
    }

    private getFeatureByClassMatrix(): number[][] {
        return this.props.dashboardContext.explanationContext.globalExplanation.perClassFeatureImportances || 
        this.props.dashboardContext.explanationContext.globalExplanation.flattenedFeatureImportances
            .map(value => [value]);
    }

    private buildSortOptions(): IDropdownOption[] {
        if (this.props.dashboardContext.explanationContext.modelMetadata.modelType !== ModelTypes.multiclass ||
            (this.props.dashboardContext.explanationContext.globalExplanation === undefined || 
             this.props.dashboardContext.explanationContext.globalExplanation.perClassFeatureImportances === undefined)) {
            return [];
        }
        const result: IDropdownOption[] = [{key: FeatureKeys.absoluteGlobal, text: localization.BarChart.absoluteGlobal}];
        result.push(...this.props.dashboardContext.explanationContext.modelMetadata.classNames
            .map((className, index) => ({key: index, text: className})));
        return result;
    }

    private setTopK(newValue: number): void {
        const newConfig = _.cloneDeep(this.props.config);
        newConfig.topK = newValue;
        this.props.onChange(newConfig, this.props.config.id);
    }

    private setChart(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        const newConfig = _.cloneDeep(this.props.config);
        newConfig.displayMode = item.key as any;
        this.props.onChange(newConfig, this.props.config.id);
    }

    private onSortSelect(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        this.setState({selectedSorting: item.key as any})
    }

    private onIconClick(): void {
        this.setState({isCalloutVisible: !this.state.isCalloutVisible});
    }

    private onDismiss(): void {
        this.setState({ isCalloutVisible: false });
    }
}