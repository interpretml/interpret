import * as _ from 'lodash';
import * as memoize from 'memoize-one';
import { DefaultButton, IconButton } from 'office-ui-fabric-react/lib/Button';
import { Callout } from 'office-ui-fabric-react/lib/Callout';
import { ComboBox, IComboBox, IComboBoxOption } from 'office-ui-fabric-react/lib/ComboBox';
import { IDropdownOption } from 'office-ui-fabric-react/lib/Dropdown';
import { Slider } from 'office-ui-fabric-react/lib/Slider';
import * as React from 'react';
import { ChartBuilder, AccessibleChart } from '../../../ChartTools';
import { localization } from '../../../Localization/localization';
import { PlotlyMode, IPlotlyProperty } from '../../../Shared';
import { FabricStyles } from '../../FabricStyles';
import { IExplanationContext, ModelTypes } from '../../IExplanationContext';
import { ModelExplanationUtils } from '../../ModelExplanationUtils';
import { PlotlyUtils, NoDataMessage, LoadingSpinner } from '../../SharedComponents';
import { FeatureImportanceModes, IGlobalFeatureImportanceProps } from './FeatureImportanceWrapper';

require('./Beehive.css');

export interface IBeehiveState {
    calloutContent?: React.ReactNode;
    calloutId?: string;
    selectedColorOption: string;
}

interface IDataArray {
    rowIndex: string[];
    color: number[];
    x: number[];
    y: number[];
    text: string[];
}

interface IProjectedData {
    rowIndex: string;
    normalizedFeatureValue: number;
    featureIndex: number;
    ditheredFeatureIndex: number;
    featureImportance: number;
    predictedClass?: string | number;
    predictedClassIndex?: number;
    trueClass?: string | number;
    trueClassIndex?: number;
    tooltip: string;
}

export class Beehive extends React.PureComponent<IGlobalFeatureImportanceProps, IBeehiveState> {
    private static maxFeatures = 30;

    private static generateSortVector: (data: IExplanationContext) => number[] 
    = (memoize as any).default((data: IExplanationContext): number[] => {
        return ModelExplanationUtils.buildSortedVector(data.globalExplanation.perClassFeatureImportances);
    });

    private static projectData: (data: IExplanationContext, sortVector: number[]) => IProjectedData[]
    = (memoize as any).default((data: IExplanationContext, sortVector: number[]): IProjectedData[] => {
        const mappers: Array<(value: string | number) => number> = Beehive.populateMappers(data);
        const isClassifier = data.modelMetadata.modelType !== ModelTypes.regression;
        return sortVector.map((featureIndex, sortVectorIndex) => {
            return data.localExplanation.flattenedValues.map((row, rowIndex) => {
                const predictedClassIndex = data.testDataset.predictedY[rowIndex];
                const predictedClass = isClassifier ? data.modelMetadata.classNames[predictedClassIndex] : predictedClassIndex;
                const trueClassIndex = data.testDataset.trueY ? data.testDataset.trueY[rowIndex] : undefined;
                const trueClass = data.testDataset.trueY ? 
                    (isClassifier ? data.modelMetadata.classNames[trueClassIndex] : trueClassIndex) : undefined;
                return {
                    rowIndex: rowIndex.toString(),
                    normalizedFeatureValue: mappers[featureIndex](data.testDataset.dataset[rowIndex][featureIndex]),
                    featureIndex: sortVectorIndex,
                    ditheredFeatureIndex: sortVectorIndex + (0.2 * Math.random() - 0.1),
                    featureImportance: row[featureIndex],
                    predictedClass,
                    predictedClassIndex,
                    trueClassIndex,
                    trueClass,
                    tooltip: Beehive.buildTooltip(data, rowIndex, featureIndex)
                };
            });
        }).reduce((prev, curr) => prev.concat(...curr), []);
    }, _.isEqual);

    private static buildPlotlyProps: (explanationContext: IExplanationContext, sortVector: number[], selectedOption: IComboBoxOption) => IPlotlyProperty
    = (memoize as any).default(
    (explanationContext: IExplanationContext, sortVector: number[], selectedOption: IComboBoxOption): IPlotlyProperty => {
        const plotlyProps = _.cloneDeep(Beehive.BasePlotlyProps);
        const rows = Beehive.projectData(explanationContext, sortVector);
        _.set(plotlyProps, 'layout.xaxis.ticktext', sortVector.map(i => explanationContext.modelMetadata.featureNames[i]));
        _.set(plotlyProps, 'layout.xaxis.tickvals', sortVector.map((val, index) => index));
        if (explanationContext.modelMetadata.modelType === ModelTypes.binary) {
            _.set(plotlyProps, 'layout.yaxis.title',
                `${localization.featureImportance}<br> ${localization.ExplanationScatter.class} : ${explanationContext.modelMetadata.classNames[0]}`);
        }
        PlotlyUtils.setColorProperty(plotlyProps, selectedOption, explanationContext.modelMetadata, selectedOption.text);
        if (selectedOption.data.isNormalized) {
            plotlyProps.data[0].marker.colorscale = [[0, 'rgba(0,0,255,0.5)'], [1, 'rgba(255,0,0,0.5)']];
            _.set(plotlyProps.data[0], 'marker.colorbar.tickvals', [0, 1]);
            _.set(plotlyProps.data[0], 'marker.colorbar.ticktext', [localization.AggregateImportance.low, localization.AggregateImportance.high]);
        } else {
            _.set(plotlyProps.data[0],'marker.opacity', 0.6);
        }
        plotlyProps.data = ChartBuilder.buildPlotlySeries(plotlyProps.data[0], rows);
        return plotlyProps;
    }, _.isEqual);

    private static buildTooltip(data: IExplanationContext, rowIndex: number, featureIndex: number): string {
        const result = [];
        result.push(localization.formatString(localization.AggregateImportance.featureLabel, data.modelMetadata.featureNames[featureIndex]));
        result.push(localization.formatString(localization.AggregateImportance.valueLabel,data.testDataset.dataset[rowIndex][featureIndex]));
        result.push(localization.formatString(
            localization.AggregateImportance.importanceLabel, 
            data.localExplanation.flattenedValues[rowIndex][featureIndex].toLocaleString(undefined, {minimumFractionDigits: 3})
        ));
        if (data.modelMetadata.modelType === ModelTypes.regression) {
            result.push(localization.formatString(localization.AggregateImportance.predictedOutputTooltip, data.testDataset.predictedY[rowIndex]));
            if (data.testDataset.trueY) {
                result.push(localization.formatString(localization.AggregateImportance.trueOutputTooltip, data.testDataset.trueY[rowIndex]));
            }
        } else {
            result.push(localization.formatString(localization.AggregateImportance.predictedClassTooltip, data.modelMetadata.classNames[data.testDataset.predictedY[rowIndex]]));
            if (data.testDataset.trueY) {
                result.push(localization.formatString(localization.AggregateImportance.trueClassTooltip, data.modelMetadata.classNames[data.testDataset.trueY[rowIndex]]));
            }
        }
        return result.join('<br>');
    }

    // To present all colors on a uniform color scale, the min and max of each feature are calculated
    // once per dataset and
    private static populateMappers: (data: IExplanationContext) => Array<(value: number | string) => number>
        = (memoize as any).default(
        (data: IExplanationContext): Array<(value: number | string) => number> => {
            return data.modelMetadata.featureNames.map((unused, featureIndex) => {
                if (data.modelMetadata.featureIsCategorical[featureIndex]) {
                    const values = _.uniq(data.testDataset.dataset.map(row => row[featureIndex])).sort();
                    return (value) => { return  values.length > 1 ? values.indexOf(value) / (values.length - 1) : 0};
                }
                const featureArray = data.testDataset.dataset.map((row: number[]) => row[featureIndex]);
                const min = Math.min(...featureArray);
                const max = Math.max(...featureArray);
                const range = max - min;
                return (value) => { return (range !== 0 && typeof value === 'number') ?  (value - min) / range : 0 };
            });
        }
    );

    private static BasePlotlyProps: IPlotlyProperty = {
        config: { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian', 'lasso2d', 'select2d']  } as any,
        data: [
            {
                hoverinfo: 'text',
                datapointLevelAccessors: {
                    customdata: {
                        path: ['rowIndex'],
                        plotlyPath: 'customdata'
                    },
                    text: {
                        path: ['tooltip'],
                        plotlyPath: 'text'
                    }
                },
                mode: PlotlyMode.markers,
                type: 'scatter',
                yAccessor: 'featureImportance',
                xAccessor: 'ditheredFeatureIndex'
            }
        ] as any,
        layout: {
            autosize: true,
            font: {
                size: 10
            },
            hovermode: 'closest',
            margin: {
                t: 10, b: 30
            },
            showlegend: false,
            yaxis: {
                automargin: true,
                title: localization.featureImportance
            },
            xaxis: {
                automargin: true
            }
        }
    };

    private readonly _crossClassIconId = 'cross-class-icon-id';
    private readonly _globalSortIconId = 'global-sort-icon-id';
    private colorOptions: IDropdownOption[];

    constructor(props: IGlobalFeatureImportanceProps) {
        super(props);
        this.state = { 
            selectedColorOption: 'normalizedFeatureValue'
        };
        this.onDismiss = this.onDismiss.bind(this);
        this.showCrossClassInfo = this.showCrossClassInfo.bind(this);
        this.showGlobalSortInfo = this.showGlobalSortInfo.bind(this);
        this.setChart = this.setChart.bind(this);
        this.setK = this.setK.bind(this);
        this.setColor = this.setColor.bind(this);
        this.colorOptions = this.buildColorOptions();
    }

    public render(): React.ReactNode {
        if (this.props.dashboardContext.explanationContext.testDataset !== undefined &&
            this.props.dashboardContext.explanationContext.localExplanation !== undefined &&
            this.props.dashboardContext.explanationContext.localExplanation.values !== undefined) {
            const sortVector = Beehive.generateSortVector(this.props.dashboardContext.explanationContext).slice(-1 * Beehive.maxFeatures).reverse();
            const plotlyProps = Beehive.buildPlotlyProps(this.props.dashboardContext.explanationContext, sortVector, this.colorOptions.find((option => (option.key as any) === this.state.selectedColorOption)));
            const weightContext = this.props.dashboardContext.weightContext;
            const relayoutArg = {'xaxis.range': [-0.5, this.props.config.topK - 0.5]};
            _.set(plotlyProps, 'layout.xaxis.range', [-0.5, this.props.config.topK - 0.5]);
            return (
                <div className="aggregate-chart">
                    <div className="top-controls">
                        <ComboBox
                            label={localization.FeatureImportanceWrapper.chartType}
                            className="path-selector"
                            selectedKey={FeatureImportanceModes.beehive}
                            onChange={this.setChart}
                            options={this.props.chartTypeOptions}
                            ariaLabel={"chart type picker"}
                            useComboBoxAsMenuWidth={true}
                            styles={FabricStyles.smallDropdownStyle}
                        />
                        <ComboBox
                            label={localization.ExplanationScatter.colorValue}
                            className="path-selector"
                            selectedKey={this.state.selectedColorOption}
                            onChange={this.setColor}
                            options={this.colorOptions}
                            ariaLabel={"color picker"}
                            useComboBoxAsMenuWidth={true}
                            styles={FabricStyles.smallDropdownStyle}
                        />
                        <div className="slider-control">
                            <div className="slider-label">
                                <span className="label-text">{localization.AggregateImportance.topKFeatures}</span>
                                {(this.props.dashboardContext.explanationContext.isGlobalDerived) &&
                                <IconButton
                                    id={this._globalSortIconId}
                                    iconProps={{ iconName: 'Info' }}
                                    title={localization.CrossClass.info}
                                    ariaLabel="Info"
                                    onClick={this.showGlobalSortInfo}
                                    styles={{ root: { marginBottom: -3, color: 'rgb(0, 120, 212)' } }}
                                />}
                            </div>
                            <Slider
                                className="feature-slider"
                                max={Math.min(Beehive.maxFeatures, this.props.dashboardContext.explanationContext.modelMetadata.featureNames.length)}
                                min={1}
                                step={1}
                                value={this.props.config.topK}
                                onChange={(value: number) => this.setK(value)}
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
                                className="path-selector"
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
                </div>
            );  
        }
        if (this.props.dashboardContext.explanationContext.localExplanation &&
            this.props.dashboardContext.explanationContext.localExplanation.percentComplete !== undefined) {
            return <LoadingSpinner/>;
        }        

        const explanationStrings = this.props.messages ? this.props.messages.LocalExpAndTestReq: undefined;
        return <NoDataMessage explanationStrings={explanationStrings}/>;
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
                    <div>
                        <br/>
                    </div>
                </div>;
            this.setState({calloutContent, calloutId: this._globalSortIconId});
        }
    }

    private buildColorOptions(): IComboBoxOption[] {
        const isRegression = this.props.dashboardContext.explanationContext.modelMetadata.modelType === ModelTypes.regression;
        const result = [
            {
                key: 'normalizedFeatureValue',
                text: localization.AggregateImportance.scaledFeatureValue,
                data: {isCategorical: false, isNormalized: true}
            },
            {
                key: 'predictedClass',
                text: isRegression ? localization.AggregateImportance.predictedValue : localization.AggregateImportance.predictedClass,
                data: {
                    isCategorical: !isRegression, 
                    sortProperty: !isRegression ? 'predictedClassIndex' : undefined
                }
            }
        ];
        if (this.props.dashboardContext.explanationContext.testDataset.trueY) {
            result.push({
                key: 'trueClass',
                text: isRegression ? localization.AggregateImportance.trueValue  : localization.AggregateImportance.trueClass ,
                data: {
                    isCategorical: !isRegression,
                    sortProperty: !isRegression ? 'trueClassIndex' : undefined
                }
            });
        }
        return result;
    }

    private setChart(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        const newConfig = _.cloneDeep(this.props.config);
        newConfig.displayMode = item.key as any;
        this.props.onChange(newConfig, this.props.config.id);
    }

    private onDismiss(): void {
        this.setState({ calloutContent: undefined, calloutId: undefined });
    }

    private setColor(event: React.FormEvent<IComboBox>, item: IComboBoxOption): void {
        this.setState({ selectedColorOption: item.key as any });
    }

    private setK(newValue: number): void {
        const newConfig = _.cloneDeep(this.props.config);
        newConfig.topK = newValue;
        this.props.onChange(newConfig, this.props.config.id);
    }
}
