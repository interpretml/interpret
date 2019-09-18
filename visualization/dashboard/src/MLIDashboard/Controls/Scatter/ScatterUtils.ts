import { IDashboardContext } from "../../ExplanationDashboard";
import { IComboBoxOption, IComboBoxStyles } from "office-ui-fabric-react/lib/ComboBox";
import _ from "lodash";
import { IDropdownOption, DropdownMenuItemType } from "office-ui-fabric-react/lib/Dropdown";
import { localization } from "../../../Localization/localization";
import * as memoize from 'memoize-one';
import { IExplanationContext, IExplanationModelMetadata, ModelTypes } from "../../IExplanationContext";
import { PlotlyMode, IPlotlyProperty, SelectionContext } from "../../../Shared";
import { ChartBuilder, AccessorMappingFunctionNames } from "../../../ChartTools";
import { FabricStyles } from "../../FabricStyles";
import { IHelpMessage, HelpMessageDict } from "../../Interfaces";
import { PlotlyUtils } from "../../SharedComponents";

export interface IScatterProps {
    plotlyProps: IPlotlyProperty;
    selectionContext: SelectionContext;
    theme?: string;
    messages?: HelpMessageDict
    dashboardContext: IDashboardContext;
    onChange: (props: IPlotlyProperty, id: string) => void;
}


export interface IProjectedData {
    TrainingData: any[];
    LocalExplanation?: number[];
    ProbabilityY?: number[];
    Index: string;
    PredictedY: string | number;
    TrueY?: string | number;
    PredictedYClassIndex?: number;
    TrueYClassIndex?: number;
}

export class ScatterUtils {

    private static baseScatterProperties: IPlotlyProperty = {
        config: { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian', 'lasso2d', 'select2d'] },
        data: [
            {
                datapointLevelAccessors: {
                    customdata: {
                        path: ['Index'],
                        plotlyPath: 'customdata'
                    },
                    text: {
                        mapFunction: AccessorMappingFunctionNames.stringifyText,
                        path: [],
                        plotlyPath: 'text'
                    }
                },
                hoverinfo: 'text',
                mode: PlotlyMode.markers,
                type: 'scatter'
            }
        ],
        layout: {
            autosize: true,
            font: {
                size: 10
            },
            margin: {
                t: 10
            },
            hovermode: 'closest',
            showlegend: false,
            xaxis: {
                tickcolor: '#118844',
                tickfont: {
                    color: '#118844'
                },
                title:{
                    font: {
                        color: '#118844'
                    }
                }
            },
            yaxis: {
                automargin: true,
                tickcolor: '#2255aa',
                tickfont: {
                    color: '#2255aa'
                },
                title:{
                    font: {
                        color: '#2255aa'
                    }
                }
            },
        } as any
    };

    public static defaultDataExpPlotlyProps(exp: IExplanationContext): IPlotlyProperty {
        let maxIndex: number = 0;
        let maxVal: number = Number.MIN_SAFE_INTEGER;

        if (exp.globalExplanation && exp.globalExplanation.perClassFeatureImportances) {
            // Find the top metric
            exp.globalExplanation.perClassFeatureImportances
                .map(classArray => classArray.reduce((a, b) => a + b), 0)
                .forEach((val, index) => {
                    if (val >= maxVal) {
                        maxIndex = index;
                        maxVal = val;
                    }
                });
        }

        const props = _.cloneDeep(ScatterUtils.baseScatterProperties);
        const xAccessor = `Index`;
        const yAccessor = `TrainingData[${maxIndex}]`; 
        const colorAccessor = 'PredictedY';
        const colorOption = {
            key: colorAccessor,
            text: localization.ExplanationScatter.predictedY,
            data: {
                isCategorical: exp.modelMetadata.modelType !== ModelTypes.regression,
                sortProperty: exp.modelMetadata.modelType !== ModelTypes.regression ? 'PredictedYClassIndex' : undefined
            }};
        const modelData = exp.modelMetadata;
        const colorbarTitle = ScatterUtils.formatItemTextForAxis(colorOption, modelData)
        PlotlyUtils.setColorProperty(props, colorOption, modelData, colorbarTitle);
        props.data[0].yAccessor = yAccessor
        props.data[0].xAccessor = xAccessor
        props.data[0].datapointLevelAccessors!['text'].path = [xAccessor, yAccessor, colorAccessor];
        props.data[0].datapointLevelAccessors!['text'].mapArgs = [
            localization.ExplanationScatter.index,
            modelData.featureNames[maxIndex],
            localization.ExplanationScatter.predictedY];

        _.set(props, 'layout.xaxis.title.text', localization.ExplanationScatter.index);
        _.set(
            props,
            'layout.yaxis.title.text',
            modelData.featureNames[maxIndex]
        );

        return props;
    }

    public static defaultExplanationPlotlyProps(exp: IExplanationContext): IPlotlyProperty {
        let maxIndex: number = 0;
        let secondIndex: number = 0;
        let maxVal: number = Number.MIN_SAFE_INTEGER;

        // Find the top two metrics
        exp.globalExplanation.perClassFeatureImportances
            .map(classArray => classArray.reduce((a, b) => a + b), 0)
            .forEach((val, index) => {
                if (val >= maxVal) {
                    secondIndex = maxIndex;
                    maxIndex = index;
                    maxVal = val;
                }
            });

        const props = _.cloneDeep(ScatterUtils.baseScatterProperties);
        const yAccessor = `LocalExplanation[${maxIndex}]`;
        const xAccessor = `TrainingData[${maxIndex}]`;
        const colorAccessor = `TrainingData[${secondIndex}]`;
        const colorOption = {
            key: colorAccessor,
            text: exp.modelMetadata.featureNames[secondIndex],
            data: {
                isCategorical: exp.modelMetadata.featureIsCategorical[secondIndex]
            }};
        const modelData = exp.modelMetadata;
        const colorbarTitle = ScatterUtils.formatItemTextForAxis(colorOption, modelData)
        PlotlyUtils.setColorProperty(props, colorOption, modelData, colorbarTitle);
        props.data[0].xAccessor = xAccessor;
        props.data[0].yAccessor = yAccessor;
        props.data[0].datapointLevelAccessors!['text'].path = [xAccessor, yAccessor, colorAccessor];
        props.data[0].datapointLevelAccessors!['text'].mapArgs = [
            localization.formatString(localization.ExplanationScatter.dataLabel, modelData.featureNames[maxIndex]),
            localization.formatString(localization.ExplanationScatter.importanceLabel, modelData.featureNames[maxIndex]),
            localization.formatString(localization.ExplanationScatter.dataLabel, modelData.featureNames[secondIndex])];

        const yAxisLabel = modelData.modelType === ModelTypes.binary ?
            localization.formatString(localization.ExplanationScatter.importanceLabel, modelData.featureNames[maxIndex]) + ` : ${modelData.classNames[0]}` :
            localization.formatString(localization.ExplanationScatter.importanceLabel, modelData.featureNames[maxIndex]);
        _.set(
            props,
            'layout.yaxis.title.text',
            yAxisLabel
        );
        _.set(
            props,
            'layout.xaxis.title.text',
            localization.formatString(localization.ExplanationScatter.dataLabel, modelData.featureNames[maxIndex])
        );

        return props;
    }

    public static populatePlotlyProps: (data: IProjectedData[], plotlyProps: IPlotlyProperty) => IPlotlyProperty
        = (memoize as any).default(
        (data: IProjectedData[], plotlyProps: IPlotlyProperty): IPlotlyProperty => {
            const result = _.cloneDeep(plotlyProps);
            result.data = result.data
                .map(series => ChartBuilder.buildPlotlySeries(series, data) as any)
                .reduce((prev, curr) => prev.concat(...curr), []);
            return result as any;
        },
        _.isEqual
    );

    public static buildOptions: (explanationContext: IExplanationContext, includeFeatureImportance: boolean) => IDropdownOption[]
        = (memoize as any).default(
        (explanationContext: IExplanationContext, includeFeatureImportance: boolean): IDropdownOption[] => {
            const result: IDropdownOption[] = [];
            if (includeFeatureImportance) {
                result.push({key: 'Header0', text: localization.featureImportance, itemType: DropdownMenuItemType.Header});
                explanationContext.modelMetadata.featureNames.forEach((featureName, index) => {
                    result.push({
                        key: `LocalExplanation[${index}]`,
                        text: localization.formatString(localization.ExplanationScatter.importanceLabel, featureName) as string,
                        data: {isCategorical: false, isFeatureImportance: true}
                    });
                });
            }
            result.push({ key: 'divider1', text: '-', itemType: DropdownMenuItemType.Divider });
            result.push({ key: 'Header1', text: localization.ExplanationScatter.dataGroupLabel, itemType: DropdownMenuItemType.Header });
            explanationContext.modelMetadata.featureNames.forEach((featureName, index) => {
                result.push({
                    key: `TrainingData[${index}]`,
                    text: includeFeatureImportance ? 
                        localization.formatString(localization.ExplanationScatter.dataLabel, featureName) as string :
                        featureName,
                    data: {isCategorical: explanationContext.modelMetadata.featureIsCategorical[index]}
                });
            });
            result.push({
                key: `Index`,
                text: localization.ExplanationScatter.index,
                data: {isCategorical: false}
            });
            result.push({ key: 'divider2', text: '-', itemType: DropdownMenuItemType.Divider });
            result.push({ key: 'Header2', text: localization.ExplanationScatter.output, itemType: DropdownMenuItemType.Header });
            result.push({
                key: `PredictedY`,
                text: localization.ExplanationScatter.predictedY,
                data: {
                    isCategorical: explanationContext.modelMetadata.modelType !== ModelTypes.regression,
                    sortProperty: explanationContext.modelMetadata.modelType !== ModelTypes.regression ? 'PredictedYClassIndex' : undefined
                }
            });
            if (explanationContext.testDataset.probabilityY) {
                explanationContext.testDataset.probabilityY[0].forEach((probClass, index) => {
                    result.push({
                        key: `ProbabilityY[${index}]`,
                        text: localization.formatString(localization.ExplanationScatter.probabilityLabel, explanationContext.modelMetadata.classNames[index]) as string,
                        data: {isCategorical: false}
                    });
                });
            }
            if (explanationContext.testDataset.trueY) {
                result.push({
                    key: `TrueY`,
                    text: localization.ExplanationScatter.trueY,
                    data :{
                        isCategorical: explanationContext.modelMetadata.modelType !== ModelTypes.regression,
                        sortProperty: explanationContext.modelMetadata.modelType !== ModelTypes.regression ? 'TrueYClassIndex' : undefined
                    }
                });
            }
            return result;
        }
    );

    // The chartBuilder util works best with arrays of objects, rather than an object with array props.
    // Just re-zipper to form;
    public static projectData: (explanationContext: IExplanationContext) => IProjectedData[]
    = (memoize as any).default(
        (explanationContext: IExplanationContext): IProjectedData[] => {
            return explanationContext.testDataset.dataset.map((featuresArray, rowIndex)=> {
                let PredictedY: string | number;
                let PredictedYClassIndex: number;
                const rawPrediction = explanationContext.testDataset.predictedY[rowIndex];
                if (explanationContext.modelMetadata.modelType === ModelTypes.regression) {
                    PredictedY = rawPrediction;
                } else {
                    PredictedYClassIndex = rawPrediction;
                    PredictedY = explanationContext.modelMetadata.classNames[rawPrediction];
                }
                const result: IProjectedData = {
                    TrainingData: featuresArray,
                    Index: rowIndex.toString(),
                    PredictedY,
                    PredictedYClassIndex
                };
                if (explanationContext.localExplanation && explanationContext.localExplanation.flattenedValues) {
                    result.LocalExplanation = explanationContext.localExplanation.flattenedValues[rowIndex];
                }
                if (explanationContext.testDataset.probabilityY) {
                    result.ProbabilityY = explanationContext.testDataset.probabilityY[rowIndex];
                }
                if (explanationContext.testDataset.trueY) {
                    const rawTruth = explanationContext.testDataset.trueY[rowIndex];
                    if (explanationContext.modelMetadata.modelType === ModelTypes.regression) {
                        result.TrueY = rawTruth;
                    } else {
                        result.TrueY = explanationContext.modelMetadata.classNames[rawTruth];
                        result.TrueYClassIndex = rawTruth;
                    }
                }
                return result;
            });
        },
        _.isEqual
    );

    public static xStyle: Partial<IComboBoxStyles> = _.extend({
        input: {
            color: '#118844',
            selectors: {
                ':hover': {
                    color: '#118844'
                },
                ':active': {
                    color: '#118844'
                },
                ':focus': {
                    color: '#118844'
                }
            }
        }}, FabricStyles.defaultDropdownStyle);


    public static yStyle: Partial<IComboBoxStyles> = _.extend({
        input: {
            color: '#2255aa',
            selectors: {
                ':hover': {
                    color: '#2255aa'
                },
                ':active': {
                    color: '#2255aa'
                },
                ':focus': {
                    color: '#2255aa'
                }
            }
        }}, FabricStyles.defaultDropdownStyle);

    public static updateNewXAccessor(props: IScatterProps, plotlyProps: IPlotlyProperty, item: IComboBoxOption, id: string): void {
        if (item.key !== plotlyProps.data[0].xAccessor) {
            plotlyProps.data[0].xAccessor = item.key.toString();
            ScatterUtils.updateTooltipArgs(plotlyProps, item.key.toString(), item.text, 0);
            _.set(plotlyProps, 'layout.xaxis.title.text', ScatterUtils.formatItemTextForAxis(
                item, 
                props.dashboardContext.explanationContext.modelMetadata));
            props.onChange(plotlyProps, id);
        }
    }

    public static updateNewYAccessor(props: IScatterProps, plotlyProps: IPlotlyProperty, item: IComboBoxOption, id: string): void {
        if (item.key !== plotlyProps.data[0].yAccessor) {
            plotlyProps.data[0].yAccessor = item.key.toString();
            ScatterUtils.updateTooltipArgs(plotlyProps, item.key.toString(), item.text, 1);
            _.set(plotlyProps, 'layout.yaxis.title.text', ScatterUtils.formatItemTextForAxis(
                item, 
                props.dashboardContext.explanationContext.modelMetadata));
            props.onChange(plotlyProps, id);
        }
    }

    public static updateColorAccessor(props: IScatterProps, plotlyProps: IPlotlyProperty, item: IComboBoxOption, id: string): void {
        const colorbarTitle = ScatterUtils.formatItemTextForAxis(item, props.dashboardContext.explanationContext.modelMetadata)
        PlotlyUtils.setColorProperty(plotlyProps, item, props.dashboardContext.explanationContext.modelMetadata, colorbarTitle);
        ScatterUtils.updateTooltipArgs(plotlyProps, item.key.toString(), item.text, 2);
        props.onChange(plotlyProps, id);
    }

    public static getselectedColorOption(plotlyProps: IPlotlyProperty, options: IDropdownOption[]): string | undefined {
        let foundOption = options.find(
            option => _.isEqual([option.key], _.get(plotlyProps.data[0], 'datapointLevelAccessors.color.path'))
        );
        if (foundOption !== undefined) {
            return foundOption.key.toString();
        }
        if(plotlyProps.data[0].groupBy === undefined ||
            plotlyProps.data[0].groupBy!.length < 1
        ) {
            return undefined;
        }
        foundOption = options.find(option => option.key === plotlyProps.data[0].groupBy![0]);
        return foundOption ? foundOption.key.toString() : undefined;
    }

    private static updateTooltipArgs(props: IPlotlyProperty, accessor: string, label: string, index: number): void {
        props.data[0].datapointLevelAccessors['text'].mapArgs[index] = label;
        props.data[0].datapointLevelAccessors['text'].path[index] = accessor;
    }

    private static formatItemTextForAxis(item: IDropdownOption, modelMetadata: IExplanationModelMetadata): string {
        if (modelMetadata.modelType === ModelTypes.binary && item.data.isFeatureImportance) {
            // Add the first class's name to the text for binary case, to clarify
            const className = modelMetadata.classNames[0];
            return `${item.text}<br> ${localization.ExplanationScatter.class} : ${className}`;
        }
        return item.text;
    }
}