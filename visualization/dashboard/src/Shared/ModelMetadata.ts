import { INumericRange } from "./INumericRange";
import { ICategoricalRange } from "./ICategoricalRange";
import { RangeTypes } from "./RangeTypes";
import _ from "lodash";
import { localization } from "../Localization/localization";

export class ModelMetadata {
    public static buildFeatureRanges(testData: any[][], isCategoricalArray: boolean[] | undefined, categoricalMap?: {[key: number]: string[]}):
        Array<INumericRange | ICategoricalRange> | undefined {
        if (testData === undefined || isCategoricalArray === undefined) {
            return undefined;
        }
        return isCategoricalArray.map((isCategorical, featureIndex) => {
            if (isCategorical) {
                if (categoricalMap && categoricalMap[featureIndex] !== undefined) {
                    return {
                        uniqueValues: categoricalMap[featureIndex],
                        rangeType: RangeTypes.categorical
                    } as ICategoricalRange;
                }
                const featureVector = testData.map(row => row[featureIndex]);
                return {uniqueValues: _.uniq(featureVector), rangeType: RangeTypes.categorical} as ICategoricalRange;
            }
            const featureVector = testData.map(row => row[featureIndex]);
            return {
                min: Math.min(...featureVector),
                max: Math.max(...featureVector),
                rangeType: featureVector.every(val => Number.isInteger(val)) ? RangeTypes.integer : RangeTypes.numeric
            } as INumericRange;
        });
    }

    public static buildIsCategorical(featureLength: number, testData?: any[][], categoricalMap?: {[key: number]: string[]}): boolean[] | undefined {
        const featureIndexArray = Array.from(Array(featureLength).keys());
        if (categoricalMap) {
            return featureIndexArray.map(i => categoricalMap[i] !== undefined);
        }
        if (testData && testData.length > 0) {
            return featureIndexArray.map((featureIndex) => {
                return !testData.every(row => typeof row[featureIndex] === "number");
            });
        }
        return undefined;
    }

    public static buildIndexedNames(numberOfClasses: number, baseString: string): string[] {
        return Array.from(Array(numberOfClasses).keys())
        .map(i => localization.formatString(baseString, i.toString()) as string);
    }
}