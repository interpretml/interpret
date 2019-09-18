import { RangeTypes } from "./RangeTypes";

export interface INumericRange {
    // if the feature is numeric
    min: number;
    max: number;
    rangeType: RangeTypes.integer | RangeTypes.numeric;
}