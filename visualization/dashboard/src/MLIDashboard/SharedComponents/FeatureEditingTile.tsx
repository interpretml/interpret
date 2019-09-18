import React from 'react';
import { TextField } from 'office-ui-fabric-react/lib/TextField';
import { FabricStyles } from '../FabricStyles';
import { ComboBox, IComboBox } from 'office-ui-fabric-react/lib/ComboBox';
import { IDropdownOption } from 'office-ui-fabric-react/lib/Dropdown';
import { RangeTypes } from '../../Shared/RangeTypes';
import { localization } from '../../Localization/localization';

require('./FeatureEditingTile.css');

export interface IFeatureEditingTileProps {
    onEdit: (index: number, val: string | number, error?: string) => void;
    defaultValue: string | number;
    featureName: string;
    index: number;
    enumeratedValues: string[] | undefined;
    rangeType: RangeTypes;
}

export interface IFeatureEditingTileState {
    value: string;
    errorMessage?: string;
}

export class FeatureEditingTile extends  React.Component<IFeatureEditingTileProps, IFeatureEditingTileState> {
    private options: IDropdownOption[] = this.props.enumeratedValues !== undefined ?
        this.props.enumeratedValues.map(value => { return {text: value, key: value}}) :
        undefined;
    
    constructor(props: IFeatureEditingTileProps) {
        super(props);
        this.state = {
            value: this.props.defaultValue.toString()
        };
        this.onValueChanged = this.onValueChanged.bind(this);
        this.onComboSelected = this.onComboSelected.bind(this);
    }

    public componentDidUpdate(prevProps: IFeatureEditingTileProps): void {
        if (this.props.defaultValue !== prevProps.defaultValue) {
            this.setState({
                value: this.props.defaultValue.toString(),
                errorMessage: undefined
            });
        }
    }
    
    public render(): React.ReactNode {
        let tileClass = "tile";
        if (this.state.value !== this.props.defaultValue.toString() && this.state.errorMessage === undefined) {
            tileClass += " edited";
        }
        if (this.state.errorMessage !== undefined) {
            tileClass += ' error'
        }

        return(
            <div className={tileClass}>
                <div className="tile-label">
                    {this.props.featureName}
                </div>
                {(this.props.enumeratedValues === undefined) && <TextField
                    styles={FabricStyles.textFieldStyle}
                    value={this.state.value}
                    onChange={this.onValueChanged}
                    errorMessage={this.state.errorMessage}
                />}
                {(this.props.enumeratedValues !== undefined) && <ComboBox
                    text={this.state.value}
                    allowFreeform={true}
                    autoComplete="on"
                    options={this.options}
                    onChange={this.onComboSelected}
                    styles={FabricStyles.defaultDropdownStyle}
                />}
            </div>
        );
    }

    private onValueChanged(ev: React.FormEvent<HTMLInputElement>, newValue?: string): void {
        const val = + newValue;
        let errorMessage: string | undefined;
        if (Number.isNaN(val) || (this.props.rangeType === RangeTypes.integer && !Number.isInteger(val))) {
            errorMessage = this.props.rangeType === RangeTypes.integer ? localization.IcePlot.integerError : localization.IcePlot.numericError;
        }
        this.props.onEdit(this.props.index, val, errorMessage);
        this.setState({value: newValue, errorMessage});
    }

    private onComboSelected(event: React.FormEvent<IComboBox>, item: IDropdownOption, index: number, userProvidedValue: string): void {
        const newVal = item !== undefined ? item.text : userProvidedValue;
        this.props.onEdit(this.props.index, newVal);
        this.setState({value: newVal});
    }
}