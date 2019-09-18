import * as _ from 'lodash';
import * as Plotly from 'plotly.js/dist/plotly-cartesian';
import { PlotlyHTMLElement, Layout } from 'plotly.js';
import * as React from 'react';
import uuidv4 from 'uuid/v4';
import { formatValue } from './DisplayFormatters';
import { PlotlyThemes } from './PlotlyThemes';

import { IPlotlyProperty, SelectionContext } from '../Shared';

type SelectableChartType = 'scatter' | 'multi-line' | 'non-selectable';

const s = require('./AccessibleChart.css');
export interface AccessibleChartProps {
    plotlyProps: IPlotlyProperty;
    theme: string;
    sharedSelectionContext: SelectionContext;
    relayoutArg?: Partial<Layout>;
    localizedStrings?: any;
}

export class AccessibleChart extends React.Component<AccessibleChartProps, { loading: boolean }> {
    private guid: string = uuidv4();
    private timer: number;
    private subscriptionId: string;
    private plotlyRef: PlotlyHTMLElement;
    private isClickHandled: boolean = false;

    constructor(props: AccessibleChartProps) {
        super(props);
        this.state = { loading: true };
        this.onChartClick = this.onChartClick.bind(this);
    }

    public componentDidMount(): void {
        if (this.hasData()) {
            this.resetRenderTimer();
            this.subscribeToSelections();
        }
    }

    public componentDidUpdate(prevProps: AccessibleChartProps): void {
        if (
            (!_.isEqual(prevProps.plotlyProps, this.props.plotlyProps) || this.props.theme !== prevProps.theme) &&
            this.hasData()
        ) {
            this.resetRenderTimer();
            if (this.plotSelectionType(prevProps.plotlyProps) !== this.plotSelectionType(this.props.plotlyProps)) {
                // The callback differs based on chart type, if the chart is now a different type, un and re subscribe.
                if (this.subscriptionId && this.props.sharedSelectionContext) {
                    this.props.sharedSelectionContext.unsubscribe(this.subscriptionId);
                }
                this.subscribeToSelections();
            }
            if (!this.state.loading) {
                this.setState({ loading: true });
            }
        } else if (!_.isEqual(this.props.relayoutArg, prevProps.relayoutArg) && this.guid) {
            Plotly.relayout(this.guid, this.props.relayoutArg);
        }
    }

    public componentWillUnmount(): void {
        if (this.subscriptionId && this.props.sharedSelectionContext) {
            this.props.sharedSelectionContext.unsubscribe(this.subscriptionId);
        }
        if (this.timer) {
            window.clearTimeout(this.timer);
        }
    }

    public render(): React.ReactNode {
        if (this.hasData()) {
            return (
                <>
                    {this.state.loading && <div className="LoadingScreen">{'Loading...'}</div>}
                    <div
                        className="GridChart"
                        id={this.guid}
                        style={{ visibility: this.state.loading ? 'hidden' : 'visible' }}
                    />
                    {this.createTableWithPlotlyData(this.props.plotlyProps.data)}
                </>
            );
        }
        return <div className="centered">{'No Data'}</div>;
    }

    private hasData(): boolean {
        return (
            this.props.plotlyProps &&
            this.props.plotlyProps.data.length > 0 &&
            _.some(this.props.plotlyProps.data, datum => !_.isEmpty(datum.y) || !_.isEmpty(datum.x))
        );
    }

    private subscribeToSelections(): void {
        if (this.plotSelectionType(this.props.plotlyProps) !== 'non-selectable' && this.props.sharedSelectionContext) {
            this.subscriptionId = this.props.sharedSelectionContext.subscribe({
                selectionCallback: selections => {
                    this.applySelections(selections);
                }
            });
        }
    }

    private resetRenderTimer(): void {
        if (this.timer) {
            window.clearTimeout(this.timer);
        }
        const themedProps = this.props.theme
            ? PlotlyThemes.applyTheme(this.props.plotlyProps, this.props.theme)
            : _.cloneDeep(this.props.plotlyProps);
        this.timer = window.setTimeout(async () => {
            this.plotlyRef = await Plotly.react(this.guid, themedProps.data, themedProps.layout, themedProps.config);
            if (this.props.sharedSelectionContext) {
                this.applySelections(this.props.sharedSelectionContext.selectedIds);
            }

            if (!this.isClickHandled) {
                this.isClickHandled = true;
                this.plotlyRef.on('plotly_click', this.onChartClick);
            }
            this.setState({ loading: false });
        }, 0);
    }

    private onChartClick(data: any): void {
        const selectionType = this.plotSelectionType(this.props.plotlyProps);
        if (selectionType !== 'non-selectable' && this.props.sharedSelectionContext) {
            if (this.props.sharedSelectionContext === undefined) {
                return;
            }
            const clickedId =
                selectionType === 'multi-line'
                    ? (data.points[0].data as any).customdata[0]
                    : (data.points[0] as any).customdata;
            const selections: string[] = this.props.sharedSelectionContext.selectedIds.slice();
            const existingIndex = selections.indexOf(clickedId);
            if (existingIndex !== -1) {
                selections.splice(existingIndex, 1);
            } else {
                selections.push(clickedId);
            }
            this.props.sharedSelectionContext.onSelect(selections);
        }
    }

    private plotSelectionType(plotlyProps: IPlotlyProperty): SelectableChartType {
        if (plotlyProps.data.length > 0 && plotlyProps.data[0] && (plotlyProps.data[0].type as any) === 'scatter') {
            if (
                plotlyProps.data.length > 1 &&
                plotlyProps.data.every(trace => {
                    const customdata = (trace as any).customdata;
                    return customdata && customdata.length === 1;
                })
            ) {
                return 'multi-line';
            }
            if (
                (plotlyProps.data[0].mode as string).includes('markers') &&
                (plotlyProps.data[0] as any).customdata !== undefined
            ) {
                return 'scatter';
            }
        }
        return 'non-selectable';
    }

    private applySelections(selections: string[]): void {
        const type = this.plotSelectionType(this.props.plotlyProps);
        if (type === 'multi-line') {
            this.applySelectionsToMultiLinePlot(selections);
        } else if (type === 'scatter') {
            this.applySelectionsToScatterPlot(selections);
        }
    }

    private applySelectionsToMultiLinePlot(selections: string[]): void {
        const opacities = this.props.plotlyProps.data.map(trace => {
            if (selections.length === 0) {
                return 1;
            }
            const customdata = (trace as any).customdata;
            return customdata && customdata.length > 0 && selections.indexOf((trace as any).customdata[0]) !== -1
                ? 1
                : 0.3;
        });
        Plotly.restyle(this.guid, 'opacity' as any, opacities);
    }

    private applySelectionsToScatterPlot(selections: string[]): void {
        const selectedPoints =
            selections.length === 0
                ? null
                : this.props.plotlyProps.data.map(trace => {
                      const selectedIndexes: number[] = [];
                      if ((trace as any).customdata) {
                          ((trace as any).customdata as string[]).forEach((id, index) => {
                              if (selections.indexOf(id) !== -1) {
                                  selectedIndexes.push(index);
                              }
                          });
                      }
                      return selectedIndexes;
                  });
        Plotly.restyle(this.guid, 'selectedpoints' as any, selectedPoints as any);
        const newLineWidths =
            selections.length === 0
                ? [0]
                : this.props.plotlyProps.data.map(trace => {
                    if ((trace as any).customdata) {
                        const customData = ((trace as any).customdata as string[]);
                        const newWidths: number[] = new Array(customData.length).fill(0);
                        customData.forEach((id, index) => {
                            if (selections.indexOf(id) !== -1) {
                                newWidths[index] = 2;
                            }
                        });
                        return newWidths
                    }
                    return [0];
                  });
        Plotly.restyle(this.guid, 'marker.line.width' as any, newLineWidths as any);
    }

    private createTableWithPlotlyData(data: Plotly.Data[]): React.ReactNode {
        return (
            <table className="plotly-table hidden">
                <tbody>
                    {data.map((datum, index) => {
                        const xDataLength = datum.x ? datum.x.length : 0;
                        const yDataLength = datum.y ? datum.y.length : 0;
                        const tableWidth = Math.max(xDataLength, yDataLength);

                        const xRowCells = [];
                        const yRowCells = [];
                        for (let i = 0; i < tableWidth; i++) {
                            // Add String() because sometimes data may be Nan
                            xRowCells.push(<td key={i + '.x'}>{datum.x ? formatValue(datum.x[i]) : ''}</td>);
                            yRowCells.push(<td key={i + '.y'}>{datum.y ? formatValue(datum.y[i]) : ''}</td>);
                        }
                        return [<tr key={index + '.x'}>{xRowCells}</tr>, <tr key={index + '.y'}>{yRowCells}</tr>];
                    })}
                </tbody>
            </table>
        );
    }
}
