import _ from "lodash";
import * as memoize from 'memoize-one';
import { DefaultButton, IconButton } from "office-ui-fabric-react/lib/Button";
import { Callout } from "office-ui-fabric-react/lib/Callout";
import { ComboBox, IComboBox, IComboBoxOption } from "office-ui-fabric-react/lib/ComboBox";
import { IDropdownOption } from "office-ui-fabric-react/lib/Dropdown";
import { Slider } from "office-ui-fabric-react/lib/Slider";
import React from "react";
import { ChartBuilder, AccessibleChart } from "../../../ChartTools";
import { localization } from "../../../Localization/localization";
import { IPlotlyProperty } from "../../../Shared";
import { FabricStyles } from "../../FabricStyles";
import { IExplanationContext, ModelTypes } from "../../IExplanationContext";
import { ModelExplanationUtils } from "../../ModelExplanationUtils";
import { NoDataMessage, LoadingSpinner, FeatureKeys, FeatureSortingKey } from "../../SharedComponents";
import { FeatureImportanceModes, IGlobalFeatureImportanceProps } from "./FeatureImportanceWrapper";

require('./Violin.css');

export enum GroupByOptions {
    none = 'none',
    predictedY = 'predictedY',
    trueY = 'trueY' 
}

export interface IViolinState {
    groupBy: GroupByOptions;
    selectedSorting: FeatureSortingKey;
    calloutContent?: React.ReactNode;
    calloutId?: string;
}

interface IDataArray {
    class: string;
    x: string[];
    y: number[];
    text?: string[];
}

export class Violin extends React.PureComponent<IGlobalFeatureImportanceProps, IViolinState> {
    private static maxFeatures = 30;
    private static maxDefaultSeries = 3;
    private static plotlyColorPalette =  [
        '#1f77b4',  // muted blue
        '#ff7f0e',  // safety orange
        '#2ca02c',  // cooked asparagus green
        '#d62728',  // brick red
        '#9467bd',  // muted purple
        '#8c564b',  // chestnut brown
        '#e377c2',  // raspberry yogurt pink
        '#7f7f7f',  // middle gray
        '#bcbd22',  // curry yellow-green
        '#17becf'   // blue-teal
    ];

    private static buildBoxPlotlyProps: (data: IExplanationContext, sortVector: number[], groupBy: GroupByOptions) => IPlotlyProperty
    = (memoize as any).default((data: IExplanationContext, sortVector: number[], groupBy: GroupByOptions): IPlotlyProperty => {
        const plotlyProps = _.cloneDeep(Violin.boxPlotlyProps);
        const classesArray = Violin.getClassesArray(data, groupBy);
        const mappedData = data.localExplanation.flattenedValues.map((featureArray, rowIndex) => {
            return {
                x: sortVector.map(featureIndex => data.modelMetadata.featureNames[featureIndex]),
                y: sortVector.map(featureIndex => featureArray[featureIndex]),
                classIndex: classesArray[rowIndex],
                class: data.modelMetadata.classNames[classesArray[rowIndex]]
            };
        });
        const computedSeries = ChartBuilder.buildPlotlySeries(plotlyProps.data[0], mappedData);
        if (computedSeries.length === 1) {
            plotlyProps.layout.showlegend = false;
        }
        plotlyProps.data = computedSeries;
        return plotlyProps;
    }, _.isEqual);

    private static buildViolinPlotlyProps: (data: IExplanationContext, sortVector: number[], groupBy: GroupByOptions) => IPlotlyProperty
    = (memoize as any).default((data: IExplanationContext, sortVector: number[], groupBy: GroupByOptions): IPlotlyProperty => {
        const plotlyProps = _.cloneDeep(Violin.violinPlotlyProps);
        const classesArray = Violin.getClassesArray(data, groupBy);
        const featuresByRows = ModelExplanationUtils.transpose2DArray(data.localExplanation.flattenedValues);
        const computedSeries = sortVector.map(featureIndex => {
            const baseSeries: any = _.cloneDeep(plotlyProps.data[0]);
            baseSeries.scalegroup = featureIndex.toString();
            // Only add a legend item for the first instance
            if (featureIndex !== 0) {
                baseSeries.showlegend = false;
            }
            const rowItems = featuresByRows[featureIndex].map((value, rowIndex) => {
                return {
                    y: value,
                    class: data.modelMetadata.classNames[classesArray[rowIndex]]
                }
            });
            baseSeries.x0 = data.modelMetadata.featureNames[featureIndex];
            baseSeries.alignmentgroup = featureIndex;
            const series = ChartBuilder.buildPlotlySeries(baseSeries, rowItems);
            series.forEach((singleSeries: any) => {
                const className = singleSeries.name;
                let classIndex = data.modelMetadata.classNames.indexOf(className);
                if (classIndex === -1) {
                    classIndex = 0;
                }
                singleSeries.legendgroup = className;
                singleSeries.alignmentgroup = featureIndex;
                singleSeries.offsetgroup = className;
                singleSeries.line = {color: Violin.plotlyColorPalette[classIndex % Violin.plotlyColorPalette.length]};
                if (classIndex >= Violin.maxDefaultSeries) {
                    singleSeries.visible = 'legendonly';
                }
            });
            return series;
        }).reduce((prev, curr) => { return prev.concat(...curr)}, []);
        computedSeries.sort((a, b) => {
            return data.modelMetadata.classNames.indexOf(a.name) - data.modelMetadata.classNames.indexOf(b.name)
        })
        plotlyProps.data = computedSeries;
        // a single series, no need for a legend
        if (computedSeries.length === sortVector.length) {
            plotlyProps.layout.showlegend = false;
        }
        return plotlyProps;
    }, _.isEqual);

    private static getClassesArray: (data: IExplanationContext, groupBy: GroupByOptions) => number[]
    = (memoize as any).default((data: IExplanationContext, groupBy: GroupByOptions): number[] => {
        switch(groupBy){
            case GroupByOptions.predictedY: {
                return data.testDataset.predictedY;
            }
            case GroupByOptions.trueY: {
                return data.testDataset.trueY;
            }
            case GroupByOptions.none:
            default: 
                return new Array(data.testDataset.predictedY.length).fill(0);
        }
    }, _.isEqual);

    private static violinPlotlyProps: IPlotlyProperty = {
        config: { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian', 'lasso2d', 'select2d']  } as any,
        data: ([{
            type: 'violin' as any,
            yAccessor: 'y',
            hoveron: "points+kde",
            groupBy:'class',
            meanline: {
              visible: true
            },
            box: {
                visible: true
            },
            scalemode: 'count',
            spanmode: 'hard',
            span: [
                0
            ],
        }] as any[]),
        layout: {
            autosize: true,
            font: {
                size: 10
            },
            hovermode: 'closest',
            margin: {
                t: 10, b: 30
            },
            legend: {
                tracegroupgap: 0
            },
            showlegend: true,
            violinmode: 'group',
            violingap: 40,
            violingroupgap: 0,
            xaxis: {
                automargin: true
            },
            yaxis: {
                automargin: true,
                title: localization.featureImportance
            }
        }  as any
    };

    private static boxPlotlyProps: IPlotlyProperty = {
        config: { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian', 'lasso2d', 'select2d']  } as any,
        data: ([{
            type: 'box' as any,
            xAccessor: 'x',
            xAccessorPrefix: `sort_by(@, &classIndex)`,
            yAccessor: 'y',
            groupBy:'class',
            boxpoints: 'Outliers',
            boxmean: 'sd'
        }] as any[]),
        layout: {
            autosize: true,
            font: {
                size: 10
            },
            hovermode: 'closest',
            margin: {
                t: 10, b: 30
            },
            showlegend: true,
            boxmode: 'group',
            xaxis: {
                automargin: true
            },
            yaxis: {
                automargin: true,
                title: localization.featureImportance
            }
        }  as any
    };

    private sortOptions: IDropdownOption[];
    private groupByOptions: IDropdownOption[];
    private readonly _crossClassIconId = 'cross-class-icon-id';
    private readonly _globalSortIconId = 'global-sort-icon-id';

    constructor(props:IGlobalFeatureImportanceProps) {
        super(props);
        this.sortOptions = this.buildSortOptions();
        this.groupByOptions = this.buildGroupOptions();
        this.onSortSelect = this.onSortSelect.bind(this);
        this.onGroupSelect = this.onGroupSelect.bind(this);
        this.setTopK = this.setTopK.bind(this);
        this.setChart = this.setChart.bind(this);
        this.onDismiss = this.onDismiss.bind(this);
        this.showCrossClassInfo = this.showCrossClassInfo.bind(this);
        this.showGlobalSortInfo = this.showGlobalSortInfo.bind(this);
        this.state = {
            selectedSorting: FeatureKeys.absoluteGlobal,
            groupBy: props.dashboardContext.explanationContext.modelMetadata.modelType === ModelTypes.regression ?
                GroupByOptions.none : GroupByOptions.predictedY
        };
    }

    public render(): React.ReactNode {
        if (this.props.dashboardContext.explanationContext.testDataset !== undefined &&
            this.props.dashboardContext.explanationContext.localExplanation !== undefined &&
            this.props.dashboardContext.explanationContext.localExplanation.values !== undefined) {
            const sortVector = this.getSortVector().slice(-1 * Violin.maxFeatures).reverse()

            const plotlyProps = this.props.config.displayMode === FeatureImportanceModes.violin ?
                Violin.buildViolinPlotlyProps(this.props.dashboardContext.explanationContext, sortVector, this.state.groupBy) :
                Violin.buildBoxPlotlyProps(this.props.dashboardContext.explanationContext, sortVector, this.state.groupBy);
            const weightContext = this.props.dashboardContext.weightContext;
            const relayoutArg = {'xaxis.range': [-0.5, this.props.config.topK - 0.5]};
            _.set(plotlyProps, 'layout.xaxis.range', [-0.5, this.props.config.topK - 0.5]);
            return (
                <div className="aggregate-chart">
                    <div className="top-controls">
                        <ComboBox
                            label={localization.FeatureImportanceWrapper.chartType}
                            className="path-selector"
                            selectedKey={this.props.config.displayMode}
                            onChange={this.setChart}
                            options={this.props.chartTypeOptions}
                            ariaLabel={"chart type picker"}
                            useComboBoxAsMenuWidth={true}
                            styles={FabricStyles.smallDropdownStyle}
                        />
                        {(this.props.dashboardContext.explanationContext.modelMetadata.modelType !== ModelTypes.regression) &&
                        <ComboBox
                            label={localization.Violin.groupBy}
                            className="path-selector"
                            selectedKey={this.state.groupBy}
                            onChange={this.onGroupSelect}
                            options={this.groupByOptions}
                            ariaLabel={"chart type picker"}
                            useComboBoxAsMenuWidth={true}
                            styles={FabricStyles.smallDropdownStyle}
                        />}
                        <div className="slider-control">
                            <div className="slider-label">
                                <span className="label-text">{localization.AggregateImportance.topKFeatures}</span>
                                <IconButton
                                    id={this._globalSortIconId}
                                    iconProps={{ iconName: 'Info' }}
                                    title={localization.CrossClass.info}
                                    ariaLabel="Info"
                                    onClick={this.showGlobalSortInfo}
                                    styles={{ root: { marginBottom: -3, color: 'rgb(0, 120, 212)' } }}
                                />
                            </div>
                            <Slider
                                className="feature-slider"
                                max={Math.min(Violin.maxFeatures, this.props.dashboardContext.explanationContext.modelMetadata.featureNames.length)}
                                min={1}
                                step={1}
                                value={this.props.config.topK}
                                onChange={this.setTopK}
                                showValue={true}
                            />
                        </div>
                        {(this.props.dashboardContext.explanationContext.modelMetadata.modelType === ModelTypes.multiclass) &&
                        <div className="selector">
                            <div className="selector-label">
                                <span>{localization.CrossClass.label}</span>
                                <IconButton
                                    id={this._crossClassIconId}
                                    iconProps={{ iconName: 'Info' }}
                                    title={localization.CrossClass.info}
                                    ariaLabel="Info"
                                    onClick={this.showCrossClassInfo}
                                    styles={{ root: { marginBottom: -3, color: 'rgb(0, 120, 212)' } }}
                                />
                            </div>
                            <ComboBox
                                className="pathSelector"
                                selectedKey={weightContext.selectedKey}
                                onChange={weightContext.onSelection}
                                options={weightContext.options}
                                ariaLabel={"Cross-class weighting selector"}
                                useComboBoxAsMenuWidth={true}
                                styles={FabricStyles.smallDropdownStyle}
                            />
                        </div>}
                    </div>
                    {this.state.calloutContent && (
                    <Callout
                        target={'#' + this.state.calloutId}
                        setInitialFocus={true}
                        onDismiss={this.onDismiss}
                        role="alertdialog">
                        <div className="callout-info">
                            {this.state.calloutContent}
                            <DefaultButton onClick={this.onDismiss}>{localization.CrossClass.close}</DefaultButton>
                        </div>
                    </Callout>
                    )}
                    <AccessibleChart
                        plotlyProps={plotlyProps}
                        sharedSelectionContext={this.props.selectionContext}
                        theme={this.props.theme}
                        relayoutArg={relayoutArg as any}
                    />
                </div>)

        }
        if (this.props.dashboardContext.explanationContext.localExplanation &&
            this.props.dashboardContext.explanationContext.localExplanation.percentComplete !== undefined) {
            return <LoadingSpinner/>;
        }        

        const explanationStrings = this.props.messages ? this.props.messages.LocalExpAndTestReq: undefined;
        return <NoDataMessage explanationStrings={explanationStrings}/>;
    }

    private getSortVector(): number[] {
        if (this.state.selectedSorting === FeatureKeys.absoluteGlobal) {
            return ModelExplanationUtils.buildSortedVector(this.props.dashboardContext.explanationContext.globalExplanation.perClassFeatureImportances);
        } if (this.state.groupBy === GroupByOptions.none) {
            return ModelExplanationUtils.buildSortedVector(this.props.dashboardContext.explanationContext.localExplanation.flattenedValues);
        }
        const classLabels = Violin.getClassesArray(this.props.dashboardContext.explanationContext, this.state.groupBy);
        const importanceSums =  this.props.dashboardContext.explanationContext.localExplanation.flattenedValues
            .filter((row, index) => { classLabels[index] === this.state.selectedSorting})
            .reduce((prev: number[], current: number[]) => {
                return prev.map((featureImp, featureIndex) => {
                    return featureImp + current[featureIndex];
                });
            }, new Array(this.props.dashboardContext.explanationContext.modelMetadata.featureNames.length).fill(0));
        return ModelExplanationUtils.getSortIndices(importanceSums);
    }

    private buildSortOptions(): IDropdownOption[] {
        if (this.props.dashboardContext.explanationContext.modelMetadata.modelType === ModelTypes.regression) {
            return [];
        }
        const result: IDropdownOption[] = [{key: FeatureKeys.absoluteGlobal, text: localization.BarChart.absoluteGlobal}];
        result.push(...this.props.dashboardContext.explanationContext.modelMetadata.classNames
            .map((className, index) => ({key: index, text: className})));
        return result;
    }

    private buildGroupOptions(): IDropdownOption[] {
        if (this.props.dashboardContext.explanationContext.modelMetadata.modelType === ModelTypes.regression) {
            return [];
        }
        const result: IDropdownOption[] = [
            {key: GroupByOptions.none, text: localization.Violin.groupNone},
            {key: GroupByOptions.predictedY, text: localization.Violin.groupPredicted}];
        if (this.props.dashboardContext.explanationContext.testDataset &&
            this.props.dashboardContext.explanationContext.testDataset.trueY !== undefined) {
                result.push({key: GroupByOptions.trueY, text: localization.Violin.groupTrue})
            }
        return result;
    }

    private setChart(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        const newConfig = _.cloneDeep(this.props.config);
        newConfig.displayMode = item.key as any;
        this.props.onChange(newConfig, this.props.config.id);
    }

    private setTopK(newValue: number): void {
        const newConfig = _.cloneDeep(this.props.config);
        newConfig.topK = newValue;
        this.props.onChange(newConfig, this.props.config.id);
    }

    private showCrossClassInfo(): void {
        if (this.state.calloutContent) {
            this.onDismiss();
        }
        else {
            const calloutContent = <div className="class-weight-info">
                    <span>{localization.CrossClass.overviewInfo}</span> 
                    <ul>
                        <li>{localization.CrossClass.absoluteValInfo}</li>
                        <li>{localization.CrossClass.predictedClassInfo}</li>
                        <li>{localization.CrossClass.enumeratedClassInfo}</li>
                    </ul>
                </div>;
            this.setState({calloutContent, calloutId: this._crossClassIconId});
        }
    }

    private showGlobalSortInfo(): void {
        if (this.state.calloutContent) {
            this.onDismiss();
        }
        else {
            const calloutContent = <div className="class-weight-info">
                    <span>{localization.FeatureImportanceWrapper.globalImportanceExplanation}</span> 
                    {this.props.dashboardContext.explanationContext.modelMetadata.modelType === ModelTypes.multiclass && (
                        <span>{localization.FeatureImportanceWrapper.multiclassImportanceAddendum}</span> 
                    )}
                </div>;
            this.setState({calloutContent, calloutId: this._globalSortIconId});
        }
    }

    private onDismiss(): void {
        this.setState({ calloutContent: undefined, calloutId: undefined });
    }

    private onSortSelect(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        this.setState({selectedSorting: item.key as any})
    }

    private onGroupSelect(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        this.setState({groupBy: item.key as any})
    }
}