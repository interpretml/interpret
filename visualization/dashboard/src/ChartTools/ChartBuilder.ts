import { Data, Datum } from "plotly.js";
import { IData } from "../Shared";
import * as jmespath from 'jmespath';
import * as _ from 'lodash';
import { AccessorMappingFunctions } from './AccessorMappingFunctions';

export class ChartBuilder {
    public static buildPlotlySeries<T>(datum: IData, rows: T[]): Array<Partial<Data>> {
        const groupingDictionary: { [key: string]: Partial<Data> } = {};
        let defaultSeries: Partial<Data> | undefined;
        const datumLevelPaths: string = datum.datapointLevelAccessors
            ? ', ' +
            Object.keys(datum.datapointLevelAccessors)
                .map(key => {
                    return `${key}: [${datum.datapointLevelAccessors![key].path.join(', ')}]`;
                })
                .join(', ')
            : '';
        const projectedRows: Array<{ x: any; y: any; group: any; size: any }> = jmespath.search(
            rows,
            `${datum.xAccessorPrefix || ''}[*].{x: ${datum.xAccessor}, y: ${datum.yAccessor}, group: ${
                datum.groupBy
            }, size: ${datum.sizeAccessor}${datumLevelPaths}}`
        );
        // for bubble charts, we scale all sizes to the max size, only needs to be done once since its global
        // Due to https://github.com/plotly/plotly.js/issues/2080 we have to set size explicitly rather than use
        // the prefered solution of sizeref
        const maxBubbleValue = Math.max(...projectedRows.map(run => Math.abs(run.size))) || 10;
        projectedRows.forEach((row, rowIndex) => {
            let series: Partial<Data> = _.cloneDeep(datum);
            // defining an x/y accessor will overwrite any hardcoded x or y values.
            if (datum.xAccessor) {
                series.x = [];
            }
            if (datum.yAccessor) {
                series.y = [];
            }
            if (datum.sizeAccessor) {
                series.marker!.size = [];
            }

            if (datum.datapointLevelAccessors !== undefined) {
                Object.keys(datum.datapointLevelAccessors).forEach(key => {
                    const plotlyPath = datum.datapointLevelAccessors![key].plotlyPath;
                    _.set(series, plotlyPath, []);
                });
            }

            // Handle mutiple group by in the future
            if (datum.groupBy && datum.groupBy.length > 0) {
                const key = row.group;
                if (key === undefined || key === null) {
                    if (defaultSeries === undefined) {
                        defaultSeries = series;
                    }
                    series = defaultSeries;
                } else {
                    if (groupingDictionary[key] === undefined) {
                        series.name = key;
                        groupingDictionary[key] = series;
                    }
                    series = groupingDictionary[key];
                }
            } else {
                if (defaultSeries === undefined) {
                    defaultSeries = series;
                }
                series = defaultSeries;
            }

            // Due to logging supporting heterogeneous metric types, a metric can be a scalar on one run and a vector on another
            // Support these cases in the minimally surprising way by upcasting a scalar point to match the highest dimension for that row (like numpy does)
            // If two arrays are logged, but of different lengths, pad the shorter ones with undefined to avoid series having different lengths concatted.
            // We always have a size of at least one, this avoids corner case of one array being empty
            let maxLength: number = 1;
            let hasVectorValues = false;
            if (Array.isArray(row.x)) {
                hasVectorValues = true;
                maxLength = Math.max(maxLength, row.x.length);
            }
            if (Array.isArray(row.y)) {
                hasVectorValues = true;
                maxLength = Math.max(maxLength, row.y.length);
            }
            if (Array.isArray(row.size)) {
                hasVectorValues = true;
                maxLength = Math.max(maxLength, row.size.length);
            }
            if (hasVectorValues) {
                // for making scalars into a vector, fill the vector with that scalar value
                if (!Array.isArray(row.x)) {
                    row.x = new Array(maxLength).fill(row.x);
                }
                if (!Array.isArray(row.y)) {
                    row.y = new Array(maxLength).fill(row.y);
                }
                if (!Array.isArray(row.size)) {
                    row.size = new Array(maxLength).fill(row.size);
                }

                // for padding too-short of arrays, set length to be uniform
                row.x.length = maxLength;
                row.y.length = maxLength;
                row.size.length = maxLength;
            }

            if (datum.xAccessor) {
                series.x = (series.x as Datum[]).concat(row.x);
            }
            if (datum.yAccessor) {
                series.y = (series.y as Datum[]).concat(row.y);
            }
            if (datum.sizeAccessor) {
                const size = (row.size * (datum.maxMarkerSize || 40) ** 2) / (2.0 * maxBubbleValue);
                series.marker!.size = (series.marker!.size as number[]).concat(Math.abs(size));
            }
            if (datum.datapointLevelAccessors !== undefined) {
                Object.keys(datum.datapointLevelAccessors).forEach(key => {
                    const accessor = datum.datapointLevelAccessors![key];
                    const plotlyPath = accessor.plotlyPath;
                    let value =
                        accessor.mapFunction !== undefined
                            ? AccessorMappingFunctions[accessor.mapFunction!](
                                row[key],
                                datum,
                                accessor.mapArgs || []
                            )
                            : row[key];
                    if (hasVectorValues) {
                        if (!Array.isArray(value)) {
                            value = new Array(maxLength).fill(value);
                        }
                        value.length = maxLength;
                    }
                    const newArray = _.get(series, plotlyPath).concat(value);
                    _.set(series, plotlyPath, newArray);
                });
            }
        });
        const result = defaultSeries !== undefined ? [defaultSeries] : [];
        Object.keys(groupingDictionary).forEach(key => {
            result.push(groupingDictionary[key]);
        });
        return result;
    }
}