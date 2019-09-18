import * as _ from 'lodash';
import moment from 'moment';

export const NonNumericPlaceholder: string = '-';


export function formatValue(value: any): string {
    if (typeof value === 'string' || !value) {
        return value;
    }
    if (_.isDate(value)) {
        return moment(value).format();
    }
    if (!_.isNaN(_.toNumber(value))) {
        const numericalVal = _.toNumber(value);
        if (_.isInteger(numericalVal)) {
            return numericalVal.toString();
        }
        return numericalVal.toPrecision(4);
    }
    if (_.isArray(value)) {
        return `vaector[${value.length}]`;
    }
    return JSON.stringify(value);
}
