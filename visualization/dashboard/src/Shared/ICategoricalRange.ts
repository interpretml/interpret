import { RangeTypes } from "./RangeTypes";

export interface ICategoricalRange {
    uniqueValues: string[];
    rangeType: RangeTypes.categorical;
}