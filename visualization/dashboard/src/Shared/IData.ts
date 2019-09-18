import { Data } from 'plotly.js';
import { IAccessor } from './IAccessor';

export interface IData extends Data {
    xAccessor?: string;
    xAccessorPrefix?: string;
    yAccessor?: string;
    yAccessorPrefix?: string;
    groupBy?: string[];
    groupByPrefix?: string;
    sizeAccessor?: string;
    maxMarkerSize?: number;
    seriesLevelAccessors?: { [key: string]: IAccessor };
    datapointLevelAccessors?: { [key: string]: IAccessor };
}
