import { IData } from "../Shared";
import { AccessorMappingFunctionNames } from "./AccessorMappingFunctionNames";

// No one hates 'any' more than me, but it is the strongest type we can say given it is the result of
// a jmes query.
const stringify = (value: any[], datum: IData, args: any[]): string => {
    const minLength = Math.min(args.length, value.length);
    const result: string[] = [];
    for (let i = 0; i < minLength; i++) {
        let formattedValue = value[i];
        if (typeof formattedValue === 'number') {
            formattedValue = formattedValue.toLocaleString(undefined, {minimumFractionDigits: 2})
        }
        result[i] = `${args[i]}: ${formattedValue}`;
    }
    return result.join('<br>');
};

export const AccessorMappingFunctions: { [key: string]: (value: any[], datum: IData, args: any[]) => any } = {
    [AccessorMappingFunctionNames.stringifyText]: stringify
};
