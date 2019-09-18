import { IComboBox, IComboBoxOption } from "office-ui-fabric-react/lib/ComboBox";

export enum WeightVectors {
    equal = 'equal',
    absAvg = 'absAvg',
    predicted = 'predicted'
}
export type WeightVectorOption = number | WeightVectors.equal | WeightVectors.predicted

export interface IWeightedDropdownContext {
    options: IComboBoxOption[];
    selectedKey: WeightVectorOption;
    onSelection: (event: React.FormEvent<IComboBox>, item: IComboBoxOption) => void;
}