export default interface IStringsParam {
    localizations?: any;
    contextualHelp?: HelpMessageDict
}

// For future adding helpful links rather than just text
export interface IHelpMessage {
    format: 'text' | 'link' | 'lineBreak';
    displayText: string;
    args?: any;
}

export enum HelperKeys {
    LocalExpAndTestReq = 'LocalExpAndTestReq',
    LocalOrGlobalAndTestReq = 'LocalOrGlobalAndTestReq',
    TestReq = 'TestReq',
    PredictorReq = 'PredictorReq'
}

export type HelpMessageDict = {
    [key in HelperKeys]?: IHelpMessage[]
}
