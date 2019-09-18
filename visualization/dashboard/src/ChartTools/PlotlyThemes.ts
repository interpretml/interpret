import { IPlotlyProperty } from '../Shared';

interface IPlotlyTheme {
    axisColor: string;
    axisGridColor: string;
    backgroundColor: string;
    fontColor: string;
}

const plotlyLightTheme: IPlotlyTheme = {
    axisColor: '#444',
    axisGridColor: '#eee',
    backgroundColor: '#fff',
    fontColor: '#000'
};

const plotlyDarkTheme: IPlotlyTheme = {
    axisColor: '#aaa',
    axisGridColor: '#222',
    backgroundColor: '#000',
    fontColor: '#fff'
};

const plotlyWhiteTheme: IPlotlyTheme = {
    axisColor: '#000',
    axisGridColor: '#000',
    backgroundColor: '#fff',
    fontColor: '#000'
};

const plotlyBlackTheme: IPlotlyTheme = {
    axisColor: '#fff',
    axisGridColor: '#fff',
    backgroundColor: '#000',
    fontColor: '#fff'
};

export class PlotlyThemes {
    public static applyTheme(props: IPlotlyProperty, theme?: string): IPlotlyProperty {
        const plotlyTheme = this.getTheme(theme);

        const newProps = { ...props };
        newProps.layout = props.layout ? { ...props.layout } : ({} as Plotly.Layout);
        newProps.layout.font = newProps.layout.font ? { ...newProps.layout.font } : ({} as Plotly.Font);
        newProps.layout.xaxis = newProps.layout.xaxis ? { ...newProps.layout.xaxis } : ({} as Plotly.LayoutAxis);
        newProps.layout.yaxis = newProps.layout.yaxis ? { ...newProps.layout.yaxis } : ({} as Plotly.LayoutAxis);

        newProps.layout.font.color = plotlyTheme.fontColor;
        newProps.layout.paper_bgcolor = plotlyTheme.backgroundColor;
        newProps.layout.plot_bgcolor = plotlyTheme.backgroundColor;
        newProps.layout.xaxis.color = plotlyTheme.axisColor;
        newProps.layout.xaxis.gridcolor = plotlyTheme.axisGridColor;
        newProps.layout.yaxis.color = plotlyTheme.axisColor;
        newProps.layout.yaxis.gridcolor = plotlyTheme.axisGridColor;

        return newProps;
    }

    private static getTheme(theme?: string): IPlotlyTheme {
        switch (theme) {
            case undefined:
            case 'light':
                return plotlyLightTheme;
            case 'dark':
                return plotlyDarkTheme;
            case 'white':
                return plotlyWhiteTheme;
            case 'black':
                return plotlyBlackTheme;
            default:
                return plotlyLightTheme;
        }
    }

    private constructor() {}
}
