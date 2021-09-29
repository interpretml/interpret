# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

# NOTE: This module is going to through some major refactoring shortly. Do not use directly.

import plotly.graph_objs as go
import numpy as np
from plotly import subplots
import matplotlib.pyplot as plt
from numbers import Number


import logging

log = logging.getLogger(__name__)

COLORS = ["#1f77b4", "#ff7f0e", "#808080"]


def is_multiclass_global_data_dict(data_dict):
    return isinstance(data_dict["scores"], np.ndarray) and data_dict["scores"].ndim == 2


def is_multiclass_local_data_dict(data_dict):
    return (
        isinstance(data_dict["scores"][0], np.ndarray)
        and len(data_dict["scores"][0]) > 2
    )


def plot_performance_curve(
    data_dict, title="", xtitle="", ytitle="", auc_prefix="", baseline=False
):

    x_values = data_dict["x_values"]
    y_values = data_dict["y_values"]
    auc = data_dict["auc"]
    thresholds = data_dict["threshold"]

    width = 2

    auc_str = "{0} = {1:.4f}".format(auc_prefix, auc)
    auc_trace = go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines",
        text=["Threshold ({0:.3f})".format(x) for x in thresholds],
        hoverinfo="text+x+y",
        line=dict(color="darkorange", width=width),
        name=auc_str,
        showlegend=False,
    )
    data = [auc_trace]
    if baseline:
        baseline_trace = go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="navy", width=width, dash="dash"),
            showlegend=False,
        )
        data.append(baseline_trace)

    layout = go.Layout(
        title=auc_str,
        xaxis=dict(title=xtitle),
        yaxis=dict(title=ytitle),
        showlegend=False,
    )

    main_fig = go.Figure(data=data, layout=layout)
    title += "<br>" + auc_str

    # TODO: Remove this if threshold lines are never used.
    # # Add treshold line
    # figure = _plot_with_line(data_dict, main_fig,
    #                          title=title, xtitle=xtitle, ytitle=ytitle,
    #                          share_xaxis=True, line_name='Threshold')
    # figure['layout']['yaxis2'].update(range=[0.0, 1.0])

    density_fig = plot_density(data_dict["density"])
    figure = _two_plot(main_fig, density_fig, title=title, share_xaxis=False)
    figure["layout"]["xaxis2"].update(title="Absolute Residuals")
    figure["layout"]["yaxis2"].update(title="Density")

    figure["layout"]["xaxis1"].update(title=xtitle)
    figure["layout"]["yaxis1"].update(title=ytitle)
    return figure


def plot_continuous_bar(
    data_dict, multiclass=False, show_error=True, title=None, xtitle="", ytitle=""
):
    if data_dict.get("scores", None) is None:  # pragma: no cover
        return None

    x_vals = data_dict["names"].copy()
    y_vals = data_dict["scores"].copy()
    y_hi = data_dict.get("upper_bounds", None)
    y_lo = data_dict.get("lower_bounds", None)

    # x_min = min(x_vals)
    # x_max = max(x_vals)

    if y_hi is None or multiclass:
        log.warning(
            "Argument show_error is set to true, but there are no bounds in the data."
        )
        show_error = False

    def extend_x_range(x):
        return x

    def extend_y_range(y):
        return np.r_[y, y[np.newaxis, -1]]

    new_x_vals = extend_x_range(x_vals)
    new_y_vals = extend_y_range(y_vals)
    if show_error:
        new_y_hi = extend_y_range(y_hi)
        new_y_lo = extend_y_range(y_lo)

    data = []
    fill = "none"
    if show_error:
        fill = "tonexty"

    if multiclass:
        for i in range(y_vals.shape[1]):
            class_name = (
                "Class {}".format(i)
                if "meta" not in data_dict
                else data_dict["meta"]["label_names"][i]
            )
            class_line = go.Scatter(
                x=new_x_vals,
                y=new_y_vals[:, i],
                line=dict(shape="hv"),
                name=class_name,
                mode="lines",
            )
            data.append(class_line)
    else:
        main_line = go.Scatter(
            x=new_x_vals,
            y=new_y_vals,
            name="Main",
            mode="lines",
            line=dict(color="rgb(31, 119, 180)", shape="hv"),
            fillcolor="rgba(68, 68, 68, 0.15)",
            fill=fill,
        )
        data.append(main_line)

    if show_error:
        upper_bound = go.Scatter(
            name="Upper Bound",
            x=new_x_vals,
            y=new_y_hi,
            mode="lines",
            marker=dict(color="#444"),
            line=dict(width=0, shape="hv"),
            fillcolor="rgba(68, 68, 68, 0.15)",
            fill="tonexty",
        )
        lower_bound = go.Scatter(
            name="Lower Bound",
            x=new_x_vals,
            y=new_y_lo,
            marker=dict(color="#444"),
            line=dict(width=0, shape="hv"),
            mode="lines",
        )
        data = [lower_bound, main_line, upper_bound]

    show_legend = True if multiclass or not show_error else False
    layout = go.Layout(
        title=title,
        showlegend=show_legend,
        xaxis=dict(title=xtitle),
        yaxis=dict(title=ytitle),
    )
    yrange = None
    if data_dict.get("scores_range", None) is not None:
        scores_range = data_dict["scores_range"]
        yrange = scores_range

    main_fig = go.Figure(data=data, layout=layout)

    # Add density
    if data_dict.get("density", None) is not None:
        figure = _plot_with_density(
            data_dict["density"], main_fig, title=title, yrange=yrange, showlegend=show_legend
        )
    else:
        figure = main_fig

    return figure


# Taken from:
# https://stackoverflow.com/questions/579310/formatting-long-numbers-as-strings-in-python
def _human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


# TODO: Clean this up after validation.
# def _pretty_number(x, rounding=2):
def _pretty_number(x):
    if isinstance(x, str):
        return x
    # return round(x, rounding)
    return _human_format(x)


# TODO: Remove this completely once performance graphs are hardened.
# def _plot_with_line(
#     data_dict, main_fig, title="", xtitle="", ytitle="", share_xaxis=False, line_name=""
# ):
#
#     secondary_fig = plot_line(
#         data_dict["line"], title=title, xtitle=xtitle, ytitle=ytitle, name=line_name
#     )
#     figure = _two_plot(main_fig, secondary_fig, title=title, share_xaxis=share_xaxis)
#     return figure
def plot_density(
    data_dict,
    title="",
    xtitle="",
    ytitle="",
    is_categorical=False,
    name="",
    color=COLORS[0],
):
    counts = data_dict["scores"]
    edges = data_dict["names"]
    data = []
    if not is_categorical:
        x_text = []
        for indx in range(len(edges) - 1):
            new_val = "{0} - {1}".format(
                _pretty_number(edges[indx]), _pretty_number(edges[indx + 1])
            )
            x_text.append(new_val)
        x_vals = [np.mean([i, j]) for i, j in zip(edges, edges[1:])]
    else:
        x_vals = edges
        x_text = edges

    data.append(
        go.Bar(
            x=x_vals,
            y=counts,
            hovertemplate='(%{hovertext}): %{y}',
            hovertext=x_text,
            name=name,
            marker=dict(color=color),
        )
    )
    layout = go.Layout(
        title=title,
        showlegend=False,
        xaxis=dict(title=xtitle),
        yaxis=dict(title=ytitle),
        hovermode="closest",
    )
    if not is_categorical:
        layout['xaxis']=dict(
            title=xtitle,
            tickmode = 'array',
            tickvals = x_vals,
            ticktext = x_text,
        )
    bar_fig = go.Figure(data, layout)
    return bar_fig

def _plot_with_density(
    data_dict,
    main_fig,
    title="",
    xtitle="",
    ytitle="",
    yrange=None,
    is_categorical=False,
    density_name="Distribution",
    showlegend=False,
):
    bar_fig = plot_density(
        data_dict, name=density_name, is_categorical=is_categorical, color=COLORS[1]
    )
    figure = _two_plot(main_fig, bar_fig, title=title, share_xaxis=is_categorical, showlegend=showlegend)
    figure["layout"]["yaxis1"].update(title="Score")
    figure["layout"]["yaxis2"].update(title="Density")
    if not is_categorical:
        figure["layout"]["bargap"] = 0
        figure["layout"]["xaxis2"].update(bar_fig.layout.xaxis)
        figure.update_xaxes(matches='x')
    if yrange is not None:
        figure["layout"]["yaxis1"].update(range=yrange)
    return figure


def _two_plot(main_fig, secondary_fig, title="", share_xaxis=True, showlegend=False):
    figure = subplots.make_subplots(
        print_grid=False, shared_xaxes=share_xaxis, rows=2, cols=1
    )
    [figure.append_trace(datum, 1, 1) for datum in main_fig["data"]]
    [figure.append_trace(datum, 2, 1) for datum in secondary_fig["data"]]
    figure["layout"].update(title=title, showlegend=showlegend)
    figure["layout"]["yaxis1"].update(domain=[0.40, 1.0])
    figure["layout"]["yaxis2"].update(domain=[0.0, 0.15])

    return figure


def plot_line(
    data_dict, title=None, xtitle="", ytitle="", name="Main", color=COLORS[0]
):
    if data_dict.get("scores", None) is None:  # pragma: no cover
        return None

    x_vals = data_dict["names"].copy()
    y_vals = data_dict["scores"].copy()
    y_hi = data_dict.get("upper_bounds", None)
    y_lo = data_dict.get("lower_bounds", None)
    background_lines = data_dict.get("background_scores", None)
    error_present = y_hi is not None
    background_present = background_lines is not None

    data = []
    fill = "none"
    if error_present:
        fill = "tonexty"

    main_line = go.Scatter(
        x=x_vals,
        y=y_vals,
        name=name,
        mode="lines",
        line=dict(color=color),
        fillcolor="rgba(68, 68, 68, 0.15)",
        fill=fill,
    )
    data.append(main_line)

    if background_present:
        for i in range(background_lines.shape[0]):
            data.append(
                go.Scatter(
                    x=x_vals,
                    y=background_lines[i, :],
                    mode="lines",
                    opacity=0.3,
                    line=dict(width=1.5),
                    name="Background: " + str(i + 1),
                    connectgaps=True,
                )
            )
    elif error_present:
        upper_bound = go.Scatter(
            name="Upper Bound",
            x=x_vals,
            y=y_hi,
            mode="lines",
            marker=dict(color="#444"),
            line=dict(width=0),
            fillcolor="rgba(68, 68, 68, 0.15)",
            fill="tonexty",
        )
        lower_bound = go.Scatter(
            name="Lower Bound",
            x=x_vals,
            y=y_lo,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode="lines",
        )
        data = [lower_bound, main_line, upper_bound]

    layout = go.Layout(
        title=title,
        showlegend=False,
        xaxis=dict(title=xtitle),
        yaxis=dict(title=ytitle),
    )
    main_fig = go.Figure(data, layout)

    # Add density
    if data_dict.get("density", None) is not None:
        figure = _plot_with_density(data_dict["density"], main_fig, title=title)
    else:
        figure = main_fig

    return figure


def plot_bar(data_dict, title="", xtitle="", ytitle=""):
    if data_dict.get("scores", None) is None:  # pragma: no cover
        return None
    x = data_dict["names"].copy()
    y = data_dict["scores"].copy()
    y_upper_err = data_dict.get("upper_bounds", None)
    if y_upper_err is not None:
        y_err = y_upper_err - y
    else:
        y_err = None
    multiclass = isinstance(y, np.ndarray) and y.ndim == 2
    traces = []
    if multiclass:
        for i in range(y.shape[1]):
            class_name = (
                "Class {}".format(i)
                if "meta" not in data_dict
                else data_dict["meta"]["label_names"][i]
            )
            class_bar = go.Bar(
                x=x,
                y=y[:, i],
                error_y=dict(type="data", array=y_err[:, i], visible=True),
                name=class_name
            )
            traces.append(class_bar)
    else:
        trace = go.Bar(
            x=x,
            y=y,
            error_y=dict(type="data", color="#ff6614", array=y_err, visible=True),
        )
        traces.append(trace)
    layout = go.Layout(
        title=title,
        showlegend=False,
        xaxis=dict(title=xtitle, type="category"),
        yaxis=dict(title=ytitle),
    )
    yrange = None
    if data_dict.get("scores_range", None) is not None:
        scores_range = data_dict["scores_range"]
        yrange = scores_range
    main_fig = go.Figure(data=traces, layout=layout)

    if multiclass:
        main_fig.update_layout(barmode="group")

    # Add density
    if data_dict.get("density", None) is not None:
        figure = _plot_with_density(
            data_dict["density"],
            main_fig,
            title=title,
            is_categorical=True,
            yrange=yrange,
            showlegend=multiclass
        )
    else:
        figure = main_fig
    return figure


def _names_with_values(names, values):
    li = []
    for name, value in zip(names, values):
        if value == "":
            li.append("{0}".format(name))
        elif isinstance(value, Number):
            li.append("{0} ({1:.2f})".format(name, value))
        else:
            li.append("{0} ({1})".format(name, value))

    return li


def plot_horizontal_bar(
    data_dict, multiclass=False, title="", xtitle="", ytitle="", start_zero=False
):
    if data_dict.get("scores", None) is None:  # pragma: no cover
        return None
    scores = data_dict["scores"].copy()
    names = data_dict["names"].copy()
    values = data_dict.get("values", None)
    if values is not None:
        values = data_dict["values"].copy()
        names = _names_with_values(names, values)
    if data_dict.get("perf", None) is not None and title == "":
        title_items = []

        predicted = data_dict["perf"]["predicted"]
        actual = data_dict["perf"]["actual"]
        predicted_score = data_dict["perf"]["predicted_score"]
        actual_score = data_dict["perf"]["actual_score"]

        if (
            "meta" in data_dict and "label_names" in data_dict["meta"]
        ):  # Classification titles   
            label_names = data_dict["meta"]["label_names"]
            predicted = label_names[predicted]

            title_str = f"Predicted Class: {predicted}"
            if not np.isnan(actual):
                actual_class = label_names[actual]
                title_str += f" | Actual Class: {actual_class}"

            title_str += f"<br />Pr(y = {predicted}): {predicted_score:.3f}"

            if not np.isnan(actual) and len(set([predicted, actual])) == 2:
                title_str += f" | Pr(y = {actual}): {actual_score:.3f}"
            title_items.append(title_str)
        else:  # Regression titles
            predicted_score = _pretty_number(predicted_score)
            title_items.append("Predicted: {}".format(predicted_score))

            if not np.isnan(actual):
                actual_score = _pretty_number(actual_score)
                title_items.append("Actual: {}".format(actual_score))

        title = " | ".join(title_items)
    if not multiclass:
        # color by positive/negative:
        color = [COLORS[0] if value <= 0 else COLORS[1] for value in scores]
    else:
        color = []
    extra = data_dict.get("extra", None)
    if extra is not None:
        scores.extend(extra["scores"])
        names.extend(extra["names"])
        if values is not None:
            values.extend(extra["values"])
        color.extend([COLORS[2]] * len(extra["scores"]))
    x = scores
    y = names
    traces = []
    if multiclass:
        for index, cls in enumerate(data_dict["meta"]["label_names"]):
            trace_scores = [x[index] for x in data_dict["scores"]] + [
                data_dict["extra"]["scores"][0][index]
            ]
            trace_names = data_dict["names"] + [data_dict["extra"]["names"]]
            traces.append(
                go.Bar(y=trace_names, x=trace_scores, orientation="h", name=cls)
            )
    else:
        traces.append(go.Bar(x=x, y=y, orientation="h", marker=dict(color=color)))

    if start_zero:
        x_range = [0, np.max(x)]
    else:
        max_abs_x = np.max(np.abs(x))
        if multiclass:
            max_abs_x = np.sum(np.array(x), axis=1)
        x_range = [-max_abs_x, max_abs_x]
    layout = dict(
        title=title,
        yaxis=dict(automargin=True, title=ytitle, dtick=1),
        xaxis=dict(range=x_range, title=xtitle),
    )
    if multiclass:
        layout["barmode"] = "relative"
    figure = go.Figure(data=traces, layout=layout)
    return figure


def mli_plot_horizontal_bar(
    scores,
    names,
    values=None,
    perf=None,
    intercept=None,
    title="",
    xtitle="",
    ytitle="",
    start_zero=False,
):
    if values is not None:
        names = _names_with_values(names, values)

    # title = "🔴 🔵<br>Predicted {0:.2f} | Actual {1:.2f}".format(
    if perf is not None and title == "":
        title_items = []
        title_items.append("Predicted {0:.2f}".format(perf["predicted"]))
        title_items.append("Actual {0:.2f}".format(perf["actual"]))
        title = " | ".join(title_items)

    color = [COLORS[0] if value <= 0 else COLORS[1] for value in scores]

    if intercept is not None:
        scores.append(intercept)
        names.append("Intercept")
        color.append(COLORS[2])

    x = scores
    y = names
    trace = go.Bar(x=x, y=y, orientation="h", marker=dict(color=color))

    if start_zero:
        x_range = [0, max(x)]
    else:
        max_abs_x = max(np.abs(x))
        x_range = [-max_abs_x, max_abs_x]

    layout = dict(
        title=title,
        yaxis=dict(automargin=True, title=ytitle),
        xaxis=dict(range=x_range, title=xtitle),
    )

    figure = go.Figure(data=[trace], layout=layout)

    return figure


def plot_pairwise_heatmap(data_dict, title="", xtitle="", ytitle=""):
    if data_dict.get("scores", None) is None:  # pragma: no cover
        return None

    bin_labels_left = data_dict["left_names"]
    bin_labels_right = data_dict["right_names"]
    bin_vals = data_dict["scores"]

    bin_vals = np.ascontiguousarray(np.transpose(bin_vals, (1, 0)))

    heatmap = go.Heatmap(z=bin_vals, x=bin_labels_left, y=bin_labels_right)
    if data_dict.get("scores_range", None) is not None:
        heatmap["zmin"] = data_dict["scores_range"][0]
        heatmap["zmax"] = data_dict["scores_range"][1]

    layout = go.Layout(title=title, xaxis=dict(title=xtitle), yaxis=dict(title=ytitle))
    figure = go.Figure(data=[heatmap], layout=layout)

    return figure


def sort_take(
    data_dict, sort_fn=None, sort_target="scores", reverse_results=False, top_n=None
):
    if top_n is None:
        top_n = len(data_dict["scores"])

    if sort_fn is not None:
        scored_vals = list(map(sort_fn, data_dict[sort_target]))
        sort_indexes = np.argsort(scored_vals)[:top_n]
    else:
        sort_indexes = np.array(range(top_n))

    data_dict = data_dict.copy()
    for key in data_dict.keys():
        if key in ["names", "scores", "values", "left_names", "right_names"]:
            if reverse_results:
                data_dict[key] = [data_dict[key][i] for i in reversed(sort_indexes)]
            else:
                data_dict[key] = [data_dict[key][i] for i in sort_indexes]

    return data_dict


def get_sort_indexes(data, sort_fn=None, top_n=None):
    if isinstance(data[0], list):
        return get_sort_indexes_2d(data, sort_fn=sort_fn, top_n=top_n)
    else:
        return get_sort_indexes_1d(data, sort_fn=sort_fn, top_n=top_n)


def get_sort_indexes_1d(data, sort_fn=None, top_n=None):
    if top_n is None:
        top_n = len(data)

    if sort_fn is not None:
        scored_vals = list(map(sort_fn, data))
        return np.argsort(scored_vals)[:top_n]
    else:
        return np.arange(top_n)


def get_sort_indexes_2d(data, sort_fn=None, top_n=None):
    if top_n is None:
        top_n = len(data[0])

    if sort_fn is not None:
        out_list = []
        for data_instance in data:
            sorted_vals = list(map(sort_fn, data_instance))
            out_list.append(np.argsort(sorted_vals)[:top_n])
        return out_list
    else:
        return np.arange(top_n)


def mli_sort_take(data, sort_indexes, reverse_results=False):
    if isinstance(data[0], list):
        out_list = []
        for j, data_instance in enumerate(data):
            if reverse_results:
                out_list.append([data_instance[i] for i in reversed(sort_indexes[j])])
            else:
                out_list.append([data_instance[i] for i in sort_indexes[j]])
        return out_list
    if reverse_results:
        return [data[i] for i in reversed(sort_indexes)]
    else:
        return [data[i] for i in sort_indexes]


def get_explanation_index(explanation_list, explanation_type):
    for i, explanation in enumerate(explanation_list):
        if explanation["explanation_type"] == explanation_type:
            return i
    return None


# TODO: Remove from code coverage until skope rules is updated upstream to work with latest scikit-learn.
def rules_to_html(data_dict, title=""):  # pragma: no cover
    multi_html_template = r"""
        <style>
        .container {{
            display: flex;
            justify-content: center;
            flex-direction: column;
            text-align: center;
            align-items: center;
        }}
        .row {{
            width: 50%;
            flex: none;
        }}
        .dotted-hr {{
            border: none;
            border-top: 1px dotted black;
        }}
        </style>
        <div class='container'>
        <div class='row'>
            <div>
                <p>
                    <h1>{title}</h1>
                </p>
            </div>
            <hr>
            {rules}
        </div>
        </div>
    """
    rule_template = r"""
        <p>
            <h3>{rule}</h3>
            <h4>
            Prediction: {outcome},
            Precision: {prec:.3f},
            Recall: {recall:.3f}
            </h4>
        </p>
        <hr class='dotted-hr'/>
    """

    template_list = []
    for idx, rule in enumerate(data_dict["rule"]):
        rule_str = rule.replace("and ", "<br/>")
        outcome = data_dict["outcome"][idx]
        precision = data_dict["precision"][idx]
        recall = data_dict["recall"][idx]
        template_item = rule_template.format(
            rule=rule_str, outcome=outcome, prec=precision, recall=recall
        )
        template_list.append(template_item)
    if len(template_list) != 0:
        rule_final = " ".join(template_list)
    else:  # pragma: no cover
        rule_final = "<h2>No rules found.</h2>"

    html_str = multi_html_template.format(title=title, rules=rule_final)
    return html_str


def plot_ebm_multiple_booleans(feat_names, ebm_global, mpl_style=False, figname=None):  # pragma: no cover
    """
        Helper function to plot the effect sizes of many Boolean features on the same figure.
        Args:
            feat_names: list of feature names to be plotted.
            ebm_global: Explanation Object for which the effect sizes should be plotted.
            mpl_style: Boolean, if True the Figure is plotted in matplotlib style.
                        If False, the Figure is plotted as an interactive Plotly figure.
            figname: str name of figure to be saved. If none, the figure is not saved.
        Returns:
            None
    """
    names = []
    upper_bounds = []
    impacts = []
    lower_bounds = []
    densities = []
    counter = 0
    for i, feat_name in enumerate(ebm_global.feature_names):
        if feat_name in feat_names:
            my_data = ebm_global.data(i)
            if len(my_data['scores']) == 2:
                my_name = "{} ({})".format(feat_name, my_data['density']['scores'][1])
                names.append(my_name)
                impacts.append(my_data['scores'][1]-my_data['scores'][0])
                upper_bounds.append(my_data['upper_bounds'][1] - my_data['lower_bounds'][0])
                lower_bounds.append(my_data['lower_bounds'][1] - my_data['upper_bounds'][0])
                densities.append(my_data['density']['scores'][1])
                counter += 1
            else:
                print("Feature: {} is not observed as a Boolean variable.".format(feat_name))
    if mpl_style:
        _ = plt.figure(figsize=(12, 12))
        sorted_i = np.argsort(impacts)
        for counter, i in enumerate(sorted_i):
            plt.bar(counter, impacts[i], width=0.5, color='blue', edgecolor='black',
                   yerr=upper_bounds[i]-impacts[i]) # Assume symmetric error.
        plt.xticks(range(len(names)), np.array(names)[sorted_i], rotation=90, fontsize=24)
        plt.ylabel("Addition to Score", fontsize=32)
        plt.yticks(fontsize=26)
        if figname is not None:
            plt.savefig(figname, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    else:
        sorted_i = np.argsort(impacts)
        names = np.array(names)[sorted_i]
        impacts = np.array(impacts)[sorted_i]
        upper_bounds = np.array(upper_bounds)[sorted_i]
        lower_bounds = np.array(lower_bounds)[sorted_i]
        densities_dict = {'names': names,
                          'scores': np.array(densities)[sorted_i]}
        data_dict = {'type': 'univariate',
            'names': names,
            'scores': impacts,
            'scores_range': (np.min(lower_bounds), np.max(upper_bounds)),
            'upper_bounds': upper_bounds,
            'lower_bounds': lower_bounds,
            'density': densities_dict,
            }
        plot_bar(data_dict)
