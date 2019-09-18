import { AccessorMappingFunctionNames } from '../ChartTools/AccessorMappingFunctionNames';

export interface IAccessor {
    mapArgs?: any[];
    mapFunction?: AccessorMappingFunctionNames;
    path: string[];
    plotlyPath: string;
}
