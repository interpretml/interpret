import { IComboBoxStyles } from "office-ui-fabric-react/lib/ComboBox";
import { IPivotStyles } from "office-ui-fabric-react/lib/Pivot";
import { ITextFieldStyles } from "office-ui-fabric-react/lib/TextField";

export class FabricStyles {
    public static defaultDropdownStyle: Partial<IComboBoxStyles> = {
        container: {
            display: "inline-flex",
            width: "100%",
        },
        root: {
            flex: 1
        },
        label: {
            padding: "5px 10px 0 10px"
        },
        callout: {
            maxHeight: "256px",
            minWidth: "200px"

        },
        optionsContainerWrapper: {
            maxHeight: "256px",
            minWidth: "200px"
        }
    }

    public static smallDropdownStyle: Partial<IComboBoxStyles> = {
        container: {
            display: "inline-flex",
            flexWrap: "wrap",
            width: "150px",
        },
        root: {
            flex: 1,
            minWidth: "150px"
        },
        label: {
            paddingRight: "10px"
        },
        callout: {
            maxHeight: "256px",
            minWidth: "200px"

        },
        optionsContainerWrapper: {
            maxHeight: "256px",
            minWidth: "200px"
        }
    }

    public static verticalTabsStyle: Partial<IPivotStyles> = {
        root: {
            height: "100%",
            width: "100px",
            display: "flex",
            flexDirection: "column"
        },
        text: {
            whiteSpace: 'normal',
            lineHeight: '28px'
        },
        link: {
            flex: 1,
            backgroundColor: '#f4f4f4',
            selectors: {
                '&:not(:last-child)': {
                    borderBottom: '1px solid grey'
                },
                '.ms-Button-flexContainer': {
                    justifyContent: 'center'
                },
                '&:focus, &:focus:not(:last-child)': {
                    border: '3px solid rgb(102, 102, 102)'
                }
            }    
        },
        linkIsSelected: {
            flex: 1,
            selectors: {
                '&:not(:last-child)': {
                    borderBottom: '1px solid grey'
                },
                '.ms-Button-flexContainer': {
                    justifyContent: 'center'
                },
                '&:focus, &:focus:not(:last-child)': {
                    border: '3px solid rgb(235, 235, 235)'
                }
            }   
        }
    }

    public static textFieldStyle: Partial<ITextFieldStyles> = {
        root: {
            minWidth: '150px',
            padding: '0 5px'
        },
        wrapper :{
            display: "inline-flex"
        },
        subComponentStyles:{
            label: {
                padding: '5px 10px 0 10px'
            },
        },
    }
}