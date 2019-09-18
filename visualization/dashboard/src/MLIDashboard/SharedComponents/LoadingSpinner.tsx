import React from "react";
import { Spinner, SpinnerSize } from 'office-ui-fabric-react/lib/Spinner';
import { localization } from "../../Localization/localization";

require('./LoadingSpinner.css');

export class LoadingSpinner extends React.PureComponent {
    public render(): React.ReactNode {
        return (
            <Spinner className={'explanation-spinner'} size={SpinnerSize.large} label={localization.BarChart.calculatingExplanation}/>
        );
    }
}