import React from "react";
import { ComboBox, IComboBoxOption, IComboBox } from "office-ui-fabric-react/lib/ComboBox";
import { TextField } from "office-ui-fabric-react/lib/TextField";
import { RangeTypes } from "../../Shared/RangeTypes";
import { localization } from "../../Localization/localization";
import _ from "lodash";
import { ICategoricalRange } from "../../Shared/ICategoricalRange";
import { INumericRange, IModelMetadata } from "../../Shared";
import { FabricStyles } from "../FabricStyles";
import { IDropdownOption } from "office-ui-fabric-react/lib/Dropdown";
require("./BinningControl.css");

export interface IBinningProps {
    modelMetadata: IModelMetadata;
    featureOptions: IDropdownOption[];
    selectedFeatureIndex: number;
    defaultSteps?: number;
    onError?: () => string;
    onChange: (value: IBinnedResponse) => void;
}

export interface IBinningState {
    featureIndex: number;
    type: RangeTypes;
    min?: string;
    minErrorMessage?: string;
    max?: string;
    maxErrorMessage?: string;
    steps?: string;
    stepsErrorMessage?: string;
    selectedOptionKeys?: string[];
    categoricalOptions?: IComboBoxOption[];
}

export interface IBinnedResponse {
    hasError: boolean;
    array: Array<number | string>;
    featureIndex: number;
    rangeType: RangeTypes;
}

export class BinningControl extends React.PureComponent<IBinningProps, IBinningState> {
    constructor(props: IBinningProps) {
        super(props);
        
        this.onFeatureSelected = this.onFeatureSelected.bind(this);
        this.onCategoricalRangeChanged = this.onCategoricalRangeChanged.bind(this);
        this.onMinRangeChanged = this.onMinRangeChanged.bind(this);
        this.onMaxRangeChanged = this.onMaxRangeChanged.bind(this);
        this.onStepsRangeChanged = this.onStepsRangeChanged.bind(this);
        this.state = undefined;
    }

    public componentDidMount(): void {
        if (!this.state) {
            this.setState(this.buildRangeView(this.props.selectedFeatureIndex), () => {
                this.pushChange();
            });
        }
    }

    public render(): React.ReactNode {
        return (
            <div className="feature-picker">
                <div className="path-selector">
                    <ComboBox
                        options={this.props.featureOptions}
                        onChange={this.onFeatureSelected}
                        label={localization.IcePlot.featurePickerLabel}
                        ariaLabel="feature picker"
                        selectedKey={!!this.state ? this.state.featureIndex : undefined }
                        useComboBoxAsMenuWidth={true}
                        styles={FabricStyles.defaultDropdownStyle}
                    />
                </div>
                {!!this.state &&
                <div className="rangeview">
                    {this.state.type === RangeTypes.categorical &&
                        <ComboBox
                            multiSelect
                            selectedKey={this.state.selectedOptionKeys}
                            allowFreeform={true}
                            autoComplete="on"
                            options={this.state.categoricalOptions}
                            onChange={this.onCategoricalRangeChanged}
                            styles={FabricStyles.defaultDropdownStyle}
                        />
                    }
                    {this.state.type !== RangeTypes.categorical &&
                        <div className="parameter-set">
                            <TextField 
                                label={localization.IcePlot.minimumInputLabel}
                                styles={FabricStyles.textFieldStyle}
                                value={this.state.min}
                                onChange={this.onMinRangeChanged}
                                errorMessage={this.state.minErrorMessage}/>
                            <TextField 
                                label={localization.IcePlot.maximumInputLabel}
                                styles={FabricStyles.textFieldStyle}
                                value={this.state.max}
                                onChange={this.onMaxRangeChanged}
                                errorMessage={this.state.maxErrorMessage}/>
                            <TextField 
                                label={localization.IcePlot.stepInputLabel}
                                styles={FabricStyles.textFieldStyle}
                                value={this.state.steps}
                                onChange={this.onStepsRangeChanged}
                                errorMessage={this.state.stepsErrorMessage}/>
                        </div>
                    }
                </div>}
            </div>
        );
    }

    private onFeatureSelected(event: React.FormEvent<IComboBox>, item: IDropdownOption): void {
        this.setState(this.buildRangeView(item.key as number), ()=> {
            this.pushChange();
        });
    }

    private onMinRangeChanged(ev: React.FormEvent<HTMLInputElement>, newValue?: string): void {
        const val = + newValue;
        const rangeView = _.cloneDeep(this.state) as IBinningState;
        rangeView.min = newValue;
        if (Number.isNaN(val) || (this.state.type === RangeTypes.integer && !Number.isInteger(val))) {
            rangeView.minErrorMessage = this.state.type === RangeTypes.integer ?
                localization.IcePlot.integerError : localization.IcePlot.numericError;
            this.setState(rangeView);
        } else {
            rangeView.minErrorMessage = undefined;
            this.setState(rangeView, () => {this.pushChange();});
        }
    }

    private onMaxRangeChanged(ev: React.FormEvent<HTMLInputElement>, newValue?: string): void {
        const val = + newValue;
        const rangeView = _.cloneDeep(this.state) as IBinningState;
        rangeView.max = newValue;
        if (Number.isNaN(val) || (this.state.type === RangeTypes.integer && !Number.isInteger(val))) {
            rangeView.maxErrorMessage = this.state.type === RangeTypes.integer ?
                localization.IcePlot.integerError : localization.IcePlot.numericError;
            this.setState(rangeView);
        } else {
            rangeView.maxErrorMessage = undefined;
            this.setState(rangeView, () => {this.pushChange();});
        }
    }

    private onStepsRangeChanged(ev: React.FormEvent<HTMLInputElement>, newValue?: string): void {
        const val = + newValue;
        const rangeView = _.cloneDeep(this.state) as IBinningState;
        rangeView.steps = newValue;
        if (!Number.isInteger(val)) {
            rangeView.stepsErrorMessage = localization.IcePlot.integerError;
            this.setState(rangeView);
        } else {
            rangeView.stepsErrorMessage = undefined;
            this.setState(rangeView, () => {this.pushChange();});
        }
    }

    private onCategoricalRangeChanged(event: React.FormEvent<IComboBox>, option?: IComboBoxOption, index?: number, value?: string): void {
        const rangeView = _.cloneDeep(this.state) as IBinningState;
        const currentSelectedKeys = rangeView.selectedOptionKeys || [];
        if (option) {
            // User selected/de-selected an existing option
            rangeView.selectedOptionKeys = this.updateSelectedOptionKeys(currentSelectedKeys, option);
        } else if (value !== undefined) {
            // User typed a freeform option
            const newOption: IComboBoxOption = { key: value, text: value };
            rangeView.selectedOptionKeys = [...currentSelectedKeys, newOption.key as string];
            rangeView.categoricalOptions.push(newOption);
        }
        this.setState(rangeView, () => {this.pushChange();});
    }

    private updateSelectedOptionKeys = (selectedKeys: string[], option: IComboBoxOption): string[] => {
        selectedKeys = [...selectedKeys]; // modify a copy
        const index = selectedKeys.indexOf(option.key as string);
        if (option.selected && index < 0) {
          selectedKeys.push(option.key as string);
        } else {
          selectedKeys.splice(index, 1);
        }
        return selectedKeys;
      }

    private buildRangeView(featureIndex: number): IBinningState {
        if (this.props.modelMetadata.featureIsCategorical[featureIndex]) {
            const summary = this.props.modelMetadata.featureRanges[featureIndex] as ICategoricalRange;
            if (summary.uniqueValues) {
                return {
                    featureIndex,
                    selectedOptionKeys: summary.uniqueValues,
                    categoricalOptions: summary.uniqueValues.map(text => {return {key: text, text};}),
                    type: RangeTypes.categorical
                };
            }
        } else {
            const summary = this.props.modelMetadata.featureRanges[featureIndex] as INumericRange;
            return {
                featureIndex,
                min: summary.min.toString(),
                max: summary.max.toString(),
                steps: this.props.defaultSteps !== undefined ? this.props.defaultSteps.toString() : "20",
                type: summary.rangeType
            };
        }
    }

    private pushChange(): void {
        if (this.state === undefined ||
            this.state.minErrorMessage !== undefined ||
            this.state.maxErrorMessage !== undefined ||
            this.state.stepsErrorMessage !== undefined) {
            this.props.onChange({
                hasError: true,
                array: [],
                featureIndex: this.state.featureIndex,
                rangeType: undefined
            });
        }
        const min = +this.state.min;
        const max = +this.state.max;
        const steps = +this.state.steps;

        if (this.state.type === RangeTypes.categorical && Array.isArray(this.state.selectedOptionKeys)) {
            this.props.onChange({
                hasError: false,
                array: this.state.selectedOptionKeys,
                featureIndex: this.state.featureIndex,
                rangeType: RangeTypes.categorical
            });
        } else if (!Number.isNaN(min) && !Number.isNaN(max) && Number.isInteger(steps)) {
            let delta = steps > 0 ? (max - min) / steps :
                max - min;
            const array = _.uniq(Array.from({length: steps}, (x, i)=> this.state.type === RangeTypes.integer ?
                Math.round(min + i * delta) :
                min + i * delta));
            this.props.onChange({
                hasError: false,
                array,
                featureIndex: this.state.featureIndex,
                rangeType: this.state.type
            });
        } else {
            this.props.onChange({
                hasError: true,
                array: [],
                featureIndex: this.state.featureIndex,
                rangeType: undefined
            });
        }
    }
}