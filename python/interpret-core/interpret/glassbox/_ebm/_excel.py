# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import logging
from io import BytesIO
from typing import Any, Dict, Optional

import dotsi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from xlsxwriter.utility import xl_range_abs, xl_rowcol_to_cell
from xlsxwriter.workbook import Workbook

DEBUGFORMATTER = "%(filename)s:%(name)s:%(funcName)s:%(lineno)d: %(message)s"
"""Debug file formatter."""

INFOFORMATTER = "%(message)s"
"""Log file and stream output formatter."""

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter(INFOFORMATTER))
log.addHandler(_ch)

_global_config = {
    "tabs": {
        # Main tab: model summary
        "Overview": {
            "tab": {
                "color": "#03045E",
                "label": "Overview",
            },
            "sheet": {"zoom": 100},
            "titlebar": {
                "row1_height": 38,
                "row2_height": 30,
                "main_title": "Model overview",
                "bg_color": "#03045E",
                "fg_color": "#FFFFFF",
                "author": "",
                "datestamp": "",
            },
            "description": {
                "field": {
                    "bg_color": "#D9D9D9",
                    "color": "#000000",
                    "font_size": 14,
                    "align": "left",
                    "indent": 1,
                    "valign": "top",
                    "text_wrap": True,
                },
                "row_height": 100,
                "surrounding_rows_height": 9,
            },
            "toc": {
                "label": "Table of contents",
                "title": {
                    "color": "#03045E",
                    "font_size": 22,
                    "indent": 1,
                    "bold": True,
                    "align": "left",
                    "valign": "center",
                },
                "category": {
                    "label": {"font_size": 14, "bold": True, "indent": 4},
                    "description": {"font_size": 14},
                },
                "subcategory": {
                    "label": {"font_size": 12, "bold": False, "indent": 7},
                    "description": {"font_size": 12, "indent": 7},
                },
            },
        },
        # Variables tabs: entrypoint & per group
        "Variables": {
            "tab": {
                "color": "#0077B6",
                "label": "Variables ➡",
            },
            "sheet": {"zoom": 100},
            "titlebar": {
                "row1_height": 38,
                "row2_height": 30,
                "main_title": "Model variables",
                "bg_color": "#0077B6",
                "fg_color": "#FFFFFF",
            },
            "table": {"style": "Table Style Light 2"},
            "toc_label": "Variables",
            "toc_description": (
                "Input variables of the model along some of their characteristics."
            ),
            "toc_group_description": (
                "Details of the variables belonging to the {} category."
            ),
            "interaction": {"max_bins": 10},
            "charts": {"display_distribution": True},
        },
        # Variables data tab: compute & plot coordinates
        "VariablesData": {
            "tab": {"color": "#CAF0F8", "label": "Variables data", "hidden": True},
            "sheet": {"zoom": 60},
        },
        "ShapePlots": {
            "tab": {"color": "#0077B6", "label": "Shape Plots ➡"},
            "sheet": {"zoom": 100},
            "toc_label": "Shape Plots",
            "toc_description": ("Shape plots of the variables."),
        },
        # Evaluation tab: ability to evaluate model on one point
        "Evaluation": {
            "enable": True,
            "tab": {
                "color": "#00B4D8",
                "label": "Evaluation",
            },
            "sheet": {"zoom": 100},
            "titlebar": {
                "row1_height": 38,
                "row2_height": 30,
                "main_title": "Model evaluation (prediction on one case)",
                "bg_color": "#00B4D8",
                "fg_color": "#FFFFFF",
            },
            "table": {"style": "Table Style Light 6"},
            "waterfall": {
                "default_color": "#00B4D8",
                "intercept_and_final_color": "#C0504D",
            },
            "toc_label": "Evaluation",
            "toc_description": "Model evaluation of a case, live in Excel.",
            "display_probability": True,
        },
    }
}

options = dotsi.fy(_global_config)


class Format:
    """
    A class to represent an workbook format.
    Simply stores it as a dict.

    Attributes:
    ----------
    desc : dict
        Format as a dict, in the Xlsxwriter format.
    """

    def __init__(self, desc: dict):
        """Construct the Format class."""
        self.desc = desc

    def __add__(self, fmt_enrichment):
        """Merge two Format objects."""
        new_desc = {**self.desc, **fmt_enrichment.desc}
        return Format(desc=new_desc)

    def register(self, workbook: Workbook):
        """Register a Format object within an XlsxWriter workbook."""
        log.debug("Registering format with attributes")
        log.debug(self.desc)
        return workbook.add_format(self.desc)


BasicFormats = dotsi.fy(
    {
        # Alignment
        "align_center": Format({"align": "center"}),
        "align_left": Format({"align": "left"}),
        "align_right": Format({"align": "right"}),
        "valign_center": Format({"valign": "vcenter"}),
        "valign_top": Format({"valign": "top"}),
        "valign_bottom": Format({"valign": "bottom"}),
        # Font size
        "font_14": Format({"font_size": 14}),
        "font_16": Format({"font_size": 16}),
        "font_18": Format({"font_size": 18}),
        "font_28": Format({"font_size": 28}),
        # Font face
        "bold": Format({"bold": True}),
        "no_bold": Format({"bold": False}),
        "underline": Format({"underline": True}),
        "no_underline": Format({"underline": False}),
        # Indent
        "indent_1": Format({"indent": 1}),
        # Content formatting
        "number_two_digits": Format({"num_format": "0.00"}),
        "percent": Format({"num_format": "0%"}),
        # Borders
        "left_border": Format({"left": 1}),
        "right_border": Format({"right": 1}),
        "top_border": Format({"top": 1}),
        "bottom_border": Format({"bottom": 1}),
    }
)

Formats = dotsi.fy({})


class Worksheet:
    """
    A class to represent a worksheet within the export.
    Allows to simplify worksheet configuration on top of Xlsxwriter.

    Attributes:
    ----------
    writer : pd.io.excel._xlsxwriter.XlsxWriter
        Reference to the writer object
    workbook : xlsxwriter.workbook.Workbook
        Reference to the Xlsxwriter workbook object
    worksheet : xlsxwriter.workbook.worksheet
        Reference to the actual worksheet object
    tab_name : str
        Tab name
    tab_color : str
        Tab color
    """

    def __init__(
        self,
        writer: pd.io.excel._xlsxwriter.XlsxWriter,
        tab_name: str,
        tab_color: str,
        default_zoom: int = 100,
        hidden: bool = False,
    ):
        """
        Construct all the necessary attributes for the worksheet wrapper.

        Parameters:
        ----------
            writer : pandas.io.excel._xlsxwriter.XlsxWriter
                XlsxWriter the sheet is associated to
            tab_name : str
                Tab name
            tab_color : str
                Tab color, HTML-style (e.g. #FFFFFF)
            default_zoom : int
                Default sheet zoom (optional, 100% by default)
            hidden : bool
                Tab hidden (default = False)
        """
        self.tab_name = tab_name
        self.tab_color = tab_color
        self.default_zoom = default_zoom
        self.hidden = hidden

        self.writer = writer
        self.workbook, self.worksheet = self.__add_tab()

        if hidden:
            self.worksheet.hide()

        self.register_formats()

        # Columns
        for c in self.sheet_columns():
            if len(c) == 3:
                self.worksheet.set_column(first_col=c[0], last_col=c[1], width=c[2])
            if len(c) == 4:
                self.worksheet.set_column(
                    first_col=c[0], last_col=c[1], width=c[2], cell_format=c[3]
                )

    def provide_data(self, *args, **kwargs):
        """Provide specific information required for this worksheet."""
        pass

    def create_worksheet(self):
        """Create worksheet content."""
        self.sheet_creation_handler()

    def __add_tab(self):
        workbook = self.writer.book
        worksheet = workbook.add_worksheet(self.tab_name)
        self.writer.sheets[self.tab_name] = worksheet

        worksheet.set_tab_color(self.tab_color)
        worksheet.hide_gridlines(2)
        worksheet.set_zoom(self.default_zoom)
        return workbook, worksheet

    def sheet_columns(self):
        """Declare columns settings for the worksheet."""
        return []

    def register_formats(self):
        """Register formats required by this worksheet."""
        pass

    def sheet_creation_handler(self):
        """
        Implement any behavior at sheet creation time.
        Note this is called after formats are registered, allowing to use them.
        """
        pass

    def sheet_save_handler(self):
        """Implement any behavior at workbook save time."""
        pass


class OverviewWorksheet(Worksheet):
    """A class to build the Overview sheet."""

    def register_formats(self):
        """Register the formats required for this worksheet."""
        # TITLE BAR
        # -------------------------------
        titlebar_colors = Format(
            {
                "bg_color": options.tabs.Overview.titlebar.bg_color,
                "color": options.tabs.Overview.titlebar.fg_color,
            }
        )
        Formats.centered = BasicFormats.align_center.register(self.workbook)

        # Title bar background
        Formats.overview_titlebar_bg = titlebar_colors.register(self.workbook)
        # Title bar – title
        Formats.overview_titlebar_title = (
            titlebar_colors
            + BasicFormats.align_left
            + BasicFormats.valign_center
            + BasicFormats.font_28
            + BasicFormats.no_bold
            + BasicFormats.indent_1
        ).register(self.workbook)
        # Title bar author
        Formats.overview_titlebar_author = (
            titlebar_colors
            + BasicFormats.align_right
            + BasicFormats.font_18
            + BasicFormats.no_bold
            + BasicFormats.valign_bottom
            + BasicFormats.indent_1
        ).register(self.workbook)
        # Title bar date stamp
        Formats.overview_titlebar_datestamp = (
            titlebar_colors
            + BasicFormats.align_right
            + BasicFormats.font_14
            + BasicFormats.no_bold
            + BasicFormats.valign_center
            + BasicFormats.indent_1
        ).register(self.workbook)

        # Links
        Formats.links = (
            BasicFormats.no_underline + BasicFormats.align_center + BasicFormats.bold
        ).register(self.workbook)

        # Model description
        Formats.overview_model_description_field = Format(
            options.tabs.Overview.description.field
        ).register(self.workbook)

        # Table of contents
        Formats.overview_toc_title = Format(options.tabs.Overview.toc.title).register(
            self.workbook
        )
        Formats.overview_toc_category_label = Format(
            options.tabs.Overview.toc.category.label
        ).register(self.workbook)
        Formats.overview_toc_category_description = Format(
            options.tabs.Overview.toc.category.description
        ).register(self.workbook)
        Formats.overview_toc_subcategory_label = Format(
            options.tabs.Overview.toc.subcategory.label
        ).register(self.workbook)
        Formats.overview_toc_subcategory_description = Format(
            options.tabs.Overview.toc.subcategory.description
        ).register(self.workbook)

    def provide_data(self, *args, **kwargs):
        """Provide data specific to this worksheet."""
        self.variables = kwargs.get("variables", None)
        self.model_description = kwargs.get("model_description", "")

    def sheet_creation_handler(self):
        """Create sheet layout elements."""
        self.__title_bar()
        if self.model_description:
            self.__model_description()
        self.__table_of_contents()

    def __title_bar(self):
        self.worksheet.set_row(
            1,
            options.tabs.Overview.titlebar.row1_height,
            cell_format=Formats.overview_titlebar_bg,
        )
        self.worksheet.set_row(
            2,
            options.tabs.Overview.titlebar.row2_height,
            cell_format=Formats.overview_titlebar_bg,
        )

        self.worksheet.merge_range("B2:D3", "")
        self.worksheet.merge_range("A2:A3", "")

        self.worksheet.write_string(
            "B2",
            options.tabs.Overview.titlebar.main_title,
            Formats.overview_titlebar_title,
        )

        if options.tabs.Overview.titlebar.author:
            self.worksheet.write_string(
                "G2",
                options.tabs.Overview.titlebar.author,
                cell_format=Formats.overview_titlebar_author,
            )

        if options.tabs.Overview.titlebar.datestamp:
            self.worksheet.write_string(
                "G3",
                options.tabs.Overview.titlebar.datestamp,
                cell_format=Formats.overview_titlebar_datestamp,
            )

    def __model_description(self):
        # Text field
        self.worksheet.merge_range("B6:F6", "")
        self.worksheet.set_row(
            5,
            options.tabs.Overview.description.row_height,
        )
        self.worksheet.write_string(
            "B6",
            self.model_description,
            cell_format=Formats.overview_model_description_field,
        )
        # Surrounding rows (top & bottom)
        self.worksheet.merge_range("B5:F5", "")
        self.worksheet.set_row(
            4,
            options.tabs.Overview.description.surrounding_rows_height,
        )
        self.worksheet.write_string(
            "B5",
            "",
            cell_format=Formats.overview_model_description_field,
        )
        self.worksheet.merge_range("B7:F7", "")
        self.worksheet.set_row(
            6,
            options.tabs.Overview.description.surrounding_rows_height,
        )
        self.worksheet.write_string(
            "B7",
            "",
            cell_format=Formats.overview_model_description_field,
        )

    def __table_of_contents(self):
        self.worksheet.merge_range("C9:C10", "")

        self.worksheet.write_string(
            "C9",
            options.tabs.Overview.toc.label,
            cell_format=Formats.overview_toc_title,
        )

        current_row = 10
        self.worksheet.write_url(
            current_row,
            2,
            "internal:'" + options.tabs.Variables.tab.label + "'!A1",
            cell_format=Formats.overview_toc_category_label,
            string=options.tabs.Variables.toc_label,
        )
        self.worksheet.write_string(
            current_row,
            3,
            options.tabs.Variables.toc_description,
            cell_format=Formats.overview_toc_category_description,
        )
        current_row += 1
        self.worksheet.write_url(
            current_row,
            2,
            "internal:'" + options.tabs.ShapePlots.tab.label + "'!A1",
            cell_format=Formats.overview_toc_category_label,
            string=options.tabs.ShapePlots.toc_label,
        )
        self.worksheet.write_string(
            current_row,
            3,
            options.tabs.ShapePlots.toc_description,
            cell_format=Formats.overview_toc_category_description,
        )
        current_row += 1

        if options.tabs.Evaluation.enable:
            self.worksheet.write_url(
                current_row,
                2,
                "internal:'" + options.tabs.Evaluation.tab.label + "'!A1",
                cell_format=Formats.overview_toc_category_label,
                string=options.tabs.Evaluation.toc_label,
            )
            self.worksheet.write_string(
                current_row,
                3,
                options.tabs.Evaluation.toc_description,
                cell_format=Formats.overview_toc_category_description,
            )
            current_row += 1

    def sheet_columns(self):
        """Declare columns settings for the worksheet."""
        return [
            (0, 0, 3),
            (1, 1, 3),
            (2, 2, 59),
            (3, 3, 60),
            (4, 4, 20),
            (5, 5, 22),
            (6, 6, 3),
            (7, 7, 2.3),
        ]


class VariablesWorksheet(Worksheet):
    """A class to build the Variables sheets."""

    def register_formats(self):
        """Register the formats required for this worksheet."""
        # TITLE BAR
        # -------------------------------
        titlebar_colors = Format(
            {
                "bg_color": options.tabs.Variables.titlebar.bg_color,
                "color": options.tabs.Variables.titlebar.fg_color,
            }
        )
        # Title bar background
        Formats.variables_titlebar_bg = titlebar_colors.register(self.workbook)
        # Title bar – title
        Formats.variables_titlebar_title = (
            titlebar_colors
            + BasicFormats.align_left
            + BasicFormats.valign_center
            + BasicFormats.font_28
            + BasicFormats.no_bold
            + BasicFormats.indent_1
        ).register(self.workbook)

        Formats.centered = BasicFormats.align_center.register(self.workbook)
        Formats.round_centered = (
            BasicFormats.align_center + BasicFormats.number_two_digits
        ).register(self.workbook)

    def provide_data(self, *args, **kwargs):
        """Provide data specific to this worksheet."""
        self.variables = kwargs.get("variables", None)
        self.ebm_model = kwargs.get("ebm_model", None)

    def sheet_creation_handler(self):
        """Create sheet layout elements."""
        self.__title_bar()
        self.__prepare_amplitude_table()
        self.__variables_table()
        self.__variables_amplitude_chart()

        self.variables_data_worksheet = self.__generate_variables_data_tab()
        self.variables_data_worksheet.provide_data(
            ebm_model=self.ebm_model, variables=self.variables
        )
        self.variables_data_worksheet.create_worksheet()

        self.data_reference = self.variables_data_worksheet.data_reference
        self.__generate_shape_plots_tab()

    def __title_bar(self):
        self.worksheet.set_row(
            1,
            options.tabs.Variables.titlebar.row1_height,
            cell_format=Formats.variables_titlebar_bg,
        )
        self.worksheet.set_row(
            2,
            options.tabs.Variables.titlebar.row2_height,
            cell_format=Formats.variables_titlebar_bg,
        )

        self.worksheet.merge_range("B2:D3", "")
        self.worksheet.merge_range("A2:A3", "")

        self.worksheet.write_string(
            "B2",
            options.tabs.Variables.titlebar.main_title,
            Formats.variables_titlebar_title,
        )

    def __prepare_amplitude_table(self):
        self.amplitude_table = self.variables.reset_index(drop=False)[
            ["#", "Variable", "Type", "Min", "Max", "Abs max"]
        ]

    def __variables_table(self):
        self.amplitude_table.to_excel(
            self.writer,
            sheet_name=options.tabs.Variables.tab.label,
            startrow=4,
            startcol=1,
            index=False,
            header=True,
        )
        header = [{"header": c} for c in self.amplitude_table.columns]
        self.worksheet.add_table(
            xl_range_abs(4, 1, 4 + len(self.amplitude_table), 6),
            {
                "autofilter": 0,
                "header_row": 1,
                "columns": header,
                "style": options.tabs.Variables.table.style,
            },
        )

        for index, row in enumerate(range(5, 5 + len(self.amplitude_table))):
            link_reference = f"internal:VariablePlot{index}"
            self.worksheet.write_url(row, 1, link_reference, Formats.links, str(index))
        self.worksheet.ignore_errors(
            {
                "number_stored_as_text": xl_range_abs(
                    5, 1, 5 + len(self.amplitude_table), 1
                )
            }
        )

    def __variables_amplitude_chart(self):
        chart = self.workbook.add_chart({"type": "bar", "subtype": "stacked"})
        chart.add_series(
            {
                "name": "Min",
                "categories": "='Variables ➡'!"
                + xl_range_abs(5, 2, 5 + len(self.amplitude_table) - 1, 2),
                "values": "='Variables ➡'!"
                + xl_range_abs(5, 4, 5 + len(self.amplitude_table) - 1, 4),
            }
        )
        chart.add_series(
            {
                "name": "Max",
                "categories": "='Variables ➡'!"
                + xl_range_abs(5, 2, 5 + len(self.amplitude_table) - 1, 2),
                "values": "='Variables ➡'!"
                + xl_range_abs(5, 5, 5 + len(self.amplitude_table) - 1, 5),
            }
        )
        chart.set_title({"name": "Variables contribution ranges"})
        chart.set_x_axis(
            {
                "major_gridlines": {
                    "visible": True,
                    "line": {"width": 1, "dash_type": "dash", "color": "#DDDDDD"},
                }
            }
        )
        chart.set_y_axis(
            {
                "label_position": "low",
                "reverse": True,
            }
        )
        chart.set_size({"x_scale": 2, "y_scale": 2})
        chart.set_legend({"none": True})
        chart.set_chartarea({"border": {"none": True}})
        self.worksheet.insert_chart(
            xl_rowcol_to_cell(9 + len(self.amplitude_table), 1),
            chart,
            {"x_offset": -40, "y_offset": -40},
        )

    def __generate_variables_data_tab(self):
        self.variables_data_worksheet = VariablesDataWorksheet(
            self.writer,
            tab_name=options.tabs.VariablesData.tab.label,
            tab_color=options.tabs.VariablesData.tab.color,
            default_zoom=options.tabs.VariablesData.sheet.zoom,
            hidden=options.tabs.VariablesData.tab.hidden,
        )
        return self.variables_data_worksheet

    def __generate_shape_plots_tab(self):
        shape_plots_worksheet = ShapePlotsWorksheet(
            self.writer,
            tab_name=options.tabs.ShapePlots.tab.label,
            tab_color=options.tabs.Variables.tab.color,
            default_zoom=options.tabs.Variables.sheet.zoom,
        )
        shape_plots_worksheet.provide_data(
            variables=self.variables,
            data_reference=self.data_reference,
            ebm_model=self.ebm_model,
        )
        shape_plots_worksheet.create_worksheet()

    def sheet_columns(self):
        """Declare columns settings for the worksheet."""
        return [
            (0, 0, 8),
            (1, 1, 3, Formats.centered),
            (2, 2, 30, Formats.centered),
            (3, 3, 40, Formats.centered),
            (4, 4, 15, Formats.centered),
            (5, 5, 10, Formats.centered),
            (6, 8, 8, Formats.round_centered),
        ]


class VariablesDataWorksheet(Worksheet):
    """A class to build the hidden sheet storing Features data."""

    def provide_data(self, *args, **kwargs):
        """Provide data specific to this worksheet."""
        self.variables = kwargs.get("variables", None)
        self.ebm_model = kwargs.get("ebm_model", None)

    def sheet_creation_handler(self):
        """Create sheet layout elements."""
        self.data_reference = []
        for _, var in self.variables.iterrows():
            start_row_feature = 6 * var["#"]
            if var.Type != "interaction":
                self.__generate_feature_data(start_row_feature, var)
            else:
                self.__generate_interaction_data(start_row_feature, var)

    def __generate_feature_data(self, start_row, var):
        feature_index = var["#"]
        feature_name = var.name
        feature_type = var.Type
        row_label = f"{feature_index}_{feature_name}"

        if feature_type == "continuous":
            # Continuous
            x_compute = np.concatenate(
                [
                    [np.nan, -1e12],
                    self.ebm_model.bins_[feature_index][0],
                ]
            )
            y_compute = self.ebm_model.term_scores_[feature_index][:-1]

            x_plot = [self.ebm_model.feature_bounds_[feature_index][0]]
            y_plot = [y_compute[1]]
            y_prev = y_compute[1]
            for (
                x_i,
                y_i,
            ) in zip(x_compute[2:], y_compute[2:]):
                x_plot.extend([x_i, x_i])
                y_plot.extend([y_prev, y_i])
                y_prev = y_i
            x_plot.append(self.ebm_model.feature_bounds_[feature_index][1])
            y_plot.append(y_compute[-1])

            plot_data = (
                pd.DataFrame([x_plot, y_plot])
                .T.rename(
                    columns={
                        0: f"{row_label}_plot_x",
                        1: f"{row_label}_plot_y",
                    }
                )
                .T
            )
            plot_data.to_excel(
                self.writer,
                sheet_name=self.tab_name,
                startrow=start_row + 2,
                startcol=0,
                index=True,
                header=False,
                na_rep="<missing>",
            )

            if options.tabs.Variables.charts.display_distribution:
                distribution_x = self.ebm_model.bins_[feature_index][0]
                distribution_y = self.ebm_model.bin_weights_[feature_index][1:-1]
                d_x_plot = [
                    self.ebm_model.feature_bounds_[feature_index][0],
                    self.ebm_model.feature_bounds_[feature_index][0],
                ]
                d_y_plot = [0, distribution_y[0]]
                d_y_prev = distribution_y[0]
                for (
                    d_x_i,
                    d_y_i,
                ) in zip(distribution_x, distribution_y[1:]):
                    d_x_plot.extend([d_x_i, d_x_i, d_x_i])
                    d_y_plot.extend([d_y_prev, 0, d_y_i])
                    d_y_prev = d_y_i
                d_x_plot.extend(
                    [
                        self.ebm_model.feature_bounds_[feature_index][1],
                        self.ebm_model.feature_bounds_[feature_index][1],
                    ]
                )
                d_y_plot.extend([d_y_prev, 0])
                distribution_data = (
                    pd.DataFrame([d_x_plot, d_y_plot])
                    .T.rename(
                        columns={
                            0: f"{row_label}_plot_distribution_x",
                            1: f"{row_label}_plot_distribution_y",
                        }
                    )
                    .T
                )

        # If the feature is not continuous, it should be either nominal or ordinal
        else:
            categories_mapping = self.ebm_model.bins_[feature_index][0]
            x_compute = list({**{"<missing>": 0}, **categories_mapping}.keys())
            y_compute = self.ebm_model.term_scores_[feature_index][:-1]
            plot_data = None
            if options.tabs.Variables.charts.display_distribution:
                d_y_plot = self.ebm_model.bin_weights_[feature_index][:-1]
                d_x_plot = None
                distribution_data = (
                    pd.DataFrame([d_y_plot])
                    .T.rename(
                        columns={
                            0: f"{row_label}_plot_distribution",
                        }
                    )
                    .T
                )

        compute_data = (
            pd.DataFrame([x_compute, y_compute])
            .T.rename(
                columns={
                    0: f"{row_label}_compute_values",
                    1: f"{row_label}_compute_contribution",
                }
            )
            .T
        )
        compute_data.to_excel(
            self.writer,
            sheet_name=self.tab_name,
            startrow=start_row,
            startcol=0,
            index=True,
            header=False,
            na_rep="<missing>",
        )
        if options.tabs.Variables.charts.display_distribution:
            distribution_data.to_excel(
                self.writer,
                sheet_name=self.tab_name,
                startrow=start_row + 4,
                startcol=0,
                index=True,
                header=False,
                na_rep="<missing>",
            )

        self.data_reference.append(
            {
                "compute_values_row": start_row,
                "compute_contributions_row": start_row + 1,
                "compute_values_length": len(x_compute),
                "plot_values_row": (start_row + 2) if plot_data is not None else None,
                "plot_contributions_row": (start_row + 3)
                if plot_data is not None
                else None,
                "plot_values_length": len(x_plot) if plot_data is not None else None,
                "plot_distribution_x_row": (start_row + 4)
                if options.tabs.Variables.charts.display_distribution
                and d_x_plot is not None
                else None,
                "plot_distribution_y_row": (start_row + 5)
                if options.tabs.Variables.charts.display_distribution
                else None,
                "plot_distribution_length": (len(d_x_plot))
                if options.tabs.Variables.charts.display_distribution
                and d_x_plot is not None
                else None,
            }
        )

    def __generate_interaction_data(self, start_row, var):
        feature_index = var["#"]
        feature_name = var.name
        row_label = f"{feature_index}_{feature_name}"
        term_feature = self.ebm_model.term_features_[feature_index]
        plot_values = {}
        bin_lens = {}
        term_interaction_names = feature_name.split(" & ")
        for index in [0, 1]:
            term_interaction = term_feature[index]
            if self.ebm_model.feature_types_in_[term_interaction] == "continuous":
                x_compute = np.concatenate(
                    [
                        [np.nan, -1e12],
                        self.ebm_model.bins_[term_interaction][-1],
                    ]
                )
                plot_values[index] = (
                    [
                        (
                            self.ebm_model.feature_bounds_[term_interaction][0],
                            self.ebm_model.bins_[term_interaction][-1][0],
                        )
                    ]
                    + [
                        *zip(
                            self.ebm_model.bins_[term_interaction][-1][:-1],
                            self.ebm_model.bins_[term_interaction][-1][1:],
                        )
                    ]
                    + [
                        (
                            self.ebm_model.bins_[term_interaction][-1][-1],
                            self.ebm_model.feature_bounds_[term_interaction][1],
                        )
                    ]
                )
            else:
                x_compute = list(
                    {
                        **{"<missing>": 0},
                        **self.ebm_model.bins_[term_interaction][-1],
                    }.keys()
                )
                plot_values[index] = self.ebm_model.bins_[term_interaction][-1]

            compute_data = pd.DataFrame([x_compute]).T.rename(
                columns={
                    0: f"{row_label}_compute_values_{index}",
                }
            )
            bin_lens[index] = len(compute_data)
            compute_data.T.to_excel(
                self.writer,
                sheet_name=self.tab_name,
                startrow=start_row + index,
                startcol=0,
                index=True,
                header=False,
                na_rep="<missing>",
            )
        y_compute = np.array(
            [row[:-1] for row in self.ebm_model.term_scores_[feature_index][:-1]]
        ).reshape(-1)
        compute_data = pd.DataFrame([y_compute]).T.rename(
            columns={
                0: f"{row_label}_compute_contribution",
            }
        )
        compute_data.T.to_excel(
            self.writer,
            sheet_name=self.tab_name,
            startrow=start_row + 2,
            startcol=0,
            index=True,
            header=False,
            na_rep="<missing>",
        )

        self.data_reference.append(
            {
                "compute_values_row_0": start_row,
                "compute_values_row_1": start_row + 1,
                "compute_contributions_row": start_row + 2,
                "plot_values": plot_values,
                "term_interaction_names": term_interaction_names,
                "plot_contributions": np.array(
                    [
                        row[1:-1]
                        for row in self.ebm_model.term_scores_[feature_index][1:-1]
                    ]
                ),
                "bin_lengths": bin_lens,
            }
        )

    def sheet_columns(self):
        """Declare columns settings for the worksheet."""
        return [(0, 0, 70)]


class ShapePlotsWorksheet(Worksheet):
    """A class to build shape plots of each variable."""

    def register_formats(self):
        """Register the formats required for this worksheet."""
        # TITLE BAR
        # -------------------------------
        titlebar_colors = Format(
            {
                "bg_color": options.tabs.Variables.titlebar.bg_color,
                "color": options.tabs.Variables.titlebar.fg_color,
            }
        )
        # Title bar background
        Formats.variables_titlebar_bg = titlebar_colors.register(self.workbook)
        # Title bar – title
        Formats.variables_titlebar_title = (
            titlebar_colors
            + BasicFormats.align_left
            + BasicFormats.valign_center
            + BasicFormats.font_16
            + BasicFormats.no_bold
            + BasicFormats.indent_1
        ).register(self.workbook)

        Formats.centered = BasicFormats.align_center.register(self.workbook)

    def provide_data(self, *args, **kwargs):
        """Provide data specific to this worksheet."""
        self.variables = kwargs.get("variables", None)
        self.data_reference = kwargs.get("data_reference", None)
        self.ebm_model = kwargs.get("ebm_model", None)

    def sheet_creation_handler(self):
        """Create sheet layout elements."""
        self.__shape_plots_doc()

    def __shape_plots_doc(self):
        start_row = 1
        for _, var in self.variables.iterrows():
            next_row = self.__shape_plot_doc(var, start_row)
            # Next start row
            start_row = next_row

    def __shape_plot_doc(self, var, start_row):
        feature_name = var.name
        feature_index = var["#"]
        feature_type = var["Type"]
        # Feature title (separator)
        self.worksheet.set_row(start_row, 25, cell_format=Formats.variables_titlebar_bg)
        self.worksheet.write_string(
            xl_rowcol_to_cell(start_row, 0),
            f"#{feature_index} - ({feature_name})",
            cell_format=Formats.variables_titlebar_title,
        )

        self.workbook.define_name(
            f"VariablePlot{feature_index}",
            "='"
            + self.tab_name
            + "'!"
            + xl_range_abs(start_row, 0, start_row + 19, 15),
        )

        # Feature plot(s)
        if feature_type != "interaction":
            compute_values_row = self.data_reference[feature_index][
                "compute_values_row"
            ]
            compute_contributions_row = self.data_reference[feature_index][
                "compute_contributions_row"
            ]
            compute_values_length = self.data_reference[feature_index][
                "compute_values_length"
            ]
            if options.tabs.Variables.charts.display_distribution:
                plot_distribution_y_row = self.data_reference[feature_index][
                    "plot_distribution_y_row"
                ]
            if feature_type == "continuous":
                plot_values_row = self.data_reference[feature_index]["plot_values_row"]
                plot_contributions_row = self.data_reference[feature_index][
                    "plot_contributions_row"
                ]
                plot_values_length = self.data_reference[feature_index][
                    "plot_values_length"
                ]
                compute_contributions_row = self.data_reference[feature_index][
                    "compute_contributions_row"
                ]
                if options.tabs.Variables.charts.display_distribution:
                    plot_distribution_x_row = self.data_reference[feature_index][
                        "plot_distribution_x_row"
                    ]
                    plot_distribution_length = self.data_reference[feature_index][
                        "plot_distribution_length"
                    ]
        else:
            plot_values = self.data_reference[feature_index]["plot_values"]
            term_interaction_names = self.data_reference[feature_index][
                "term_interaction_names"
            ]
            plot_contributions = self.data_reference[feature_index][
                "plot_contributions"
            ]

        if feature_type == "continuous":
            chart = self.workbook.add_chart({"type": "scatter", "subtype": "straight"})
            chart.add_series(
                {
                    "name": feature_name,
                    "categories": "='"
                    + options.tabs.VariablesData.tab.label
                    + "'!"
                    + xl_range_abs(
                        plot_values_row, 1, plot_values_row, plot_values_length
                    ),
                    "values": "='"
                    + options.tabs.VariablesData.tab.label
                    + "'!"
                    + xl_range_abs(
                        plot_contributions_row,
                        1,
                        plot_contributions_row,
                        plot_values_length,
                    ),
                }
            )
            chart.add_series(
                {
                    "name": "<missing>",
                    "categories": "{0}",
                    "values": "='"
                    + options.tabs.VariablesData.tab.label
                    + "'!"
                    + xl_rowcol_to_cell(compute_contributions_row, 1),
                    "marker": {
                        "type": "diamond",
                        "size": 10,
                        "fill": {"color": "#BE4B48"},
                    },
                    "line": {"none": True},
                }
            )
            if options.tabs.Variables.charts.display_distribution:
                chart.add_series(
                    {
                        "name": "Distribution",
                        "categories": "='"
                        + options.tabs.VariablesData.tab.label
                        + "'!"
                        + xl_range_abs(
                            plot_distribution_x_row,
                            1,
                            plot_distribution_x_row,
                            plot_distribution_length,
                        ),
                        "values": "='"
                        + options.tabs.VariablesData.tab.label
                        + "'!"
                        + xl_range_abs(
                            plot_distribution_y_row,
                            1,
                            plot_distribution_y_row,
                            plot_distribution_length,
                        ),
                        "line": {"color": "#FFD580", "transparency": 50},
                        "marker": {"type": "none"},
                        "y2_axis": True,
                    }
                )
                chart.set_y2_axis(
                    {
                        "num_font": {"color": "#FFD580"},
                        "line": {"none": True},
                    }
                )
        elif feature_type in ["nominal", "ordinal"]:
            chart = self.workbook.add_chart({"type": "column"})
            chart.add_series(
                {
                    "name": feature_name,
                    "categories": "='"
                    + options.tabs.VariablesData.tab.label
                    + "'!"
                    + xl_range_abs(
                        compute_values_row,
                        1,
                        compute_values_row,
                        compute_values_length,
                    ),
                    "values": "='"
                    + options.tabs.VariablesData.tab.label
                    + "'!"
                    + xl_range_abs(
                        compute_contributions_row,
                        1,
                        compute_contributions_row,
                        compute_values_length,
                    ),
                }
            )
            if options.tabs.Variables.charts.display_distribution:
                area_chart = self.workbook.add_chart({"type": "area"})
                area_chart.add_series(
                    {
                        "name": "Distribution",
                        "categories": "='"
                        + options.tabs.VariablesData.tab.label
                        + "'!"
                        + xl_range_abs(
                            compute_values_row,
                            1,
                            compute_values_row,
                            compute_values_length,
                        ),
                        "values": "='"
                        + options.tabs.VariablesData.tab.label
                        + "'!"
                        + xl_range_abs(
                            plot_distribution_y_row,
                            1,
                            plot_distribution_y_row,
                            compute_values_length,
                        ),
                        "fill": {"color": "#D9D9D9"},
                        "border": {"color": "#D9D9D9"},
                        "y2_axis": True,
                    }
                )
                area_chart.set_y2_axis(
                    {
                        "num_font": {"color": "#D9D9D9"},
                        "line": {"none": True},
                    }
                )
                chart.combine(area_chart)
        else:
            imgdata = BytesIO()
            heatmap = sns.heatmap(
                plot_contributions.T,
                xticklabels=plot_values[0],
                yticklabels=plot_values[1],
                cmap="RdBu_r",
            )
            heatmap.set_xlabel(term_interaction_names[0])
            heatmap.set_ylabel(term_interaction_names[1])
            heatmap.invert_yaxis()
            plt.locator_params(nbins=options.tabs.Variables.interaction.max_bins)
            fig = heatmap.get_figure()
            plt.close(fig)
            fig.savefig(imgdata, format="png", bbox_inches="tight")
            self.worksheet.insert_image(
                start_row + 7,
                1,
                "",
                {"image_data": imgdata, "x_offset": -40, "y_offset": -110},
            )
            next_start_row = start_row + 25
            return next_start_row

        # Chart style
        default_config = {
            "title": {"name": feature_name},
            "x_axis": {},
            "y_axis": {
                "major_gridlines": {
                    "visible": True,
                    "line": {"width": 1, "dash_type": "dash", "color": "#DDDDDD"},
                }
            },
            "size": {
                "x_scale": 2,
                "y_scale": 1.2,
            },
            "legend": {
                "position": "bottom",
            },
            "chartarea": {"border": {"none": True}},
        }

        for config in default_config:
            getattr(chart, "set_" + config)(default_config[config])

        self.worksheet.insert_chart(
            xl_rowcol_to_cell(start_row + 7, 1),
            chart,
            {"x_offset": -40, "y_offset": -100},
        )

        next_start_row = start_row + 20

        # end for each chart

        log.debug(f"Generated plots for variable {feature_name}…")
        return next_start_row


class EvaluationWorksheet(Worksheet):
    """A class to build the Evaluation sheet."""

    def register_formats(self):
        """Register the formats required for this worksheet."""
        titlebar_colors = Format(
            {
                "bg_color": options.tabs.Evaluation.titlebar.bg_color,
                "color": options.tabs.Evaluation.titlebar.fg_color,
            }
        )
        Formats.centered = BasicFormats.align_center.register(self.workbook)

        Formats.evaluation_titlebar_bg = titlebar_colors.register(self.workbook)
        Formats.evaluation_titlebar_title = (
            titlebar_colors
            + BasicFormats.align_left
            + BasicFormats.valign_center
            + BasicFormats.font_28
            + BasicFormats.no_bold
            + BasicFormats.indent_1
        ).register(self.workbook)

        Formats.links = (
            BasicFormats.no_underline + BasicFormats.align_center + BasicFormats.bold
        ).register(self.workbook)

        Formats.centered_2digits = (
            BasicFormats.align_center + BasicFormats.number_two_digits
        ).register(self.workbook)

        Formats.bold = (
            BasicFormats.bold
            + BasicFormats.align_center
            + BasicFormats.number_two_digits
        ).register(self.workbook)

        white_color = Format(
            {
                "color": options.tabs.Evaluation.titlebar.fg_color,
            }
        )
        Formats.white = white_color.register(self.workbook)

    def provide_data(self, *args, **kwargs):
        """Provide data specific to this worksheet."""
        self.variables = kwargs.get("variables", None)
        self.ebm_model = kwargs.get("ebm_model", None)
        self.data_reference = kwargs.get("data_reference", None)
        self.default_evaluation = kwargs.get("default_evaluation", None)

    def sheet_creation_handler(self):
        """Create sheet layout elements."""
        self.__title_bar()
        self.__table()
        self.__prediction()
        self.__summation()
        self.__waterfall()

    def __title_bar(self):
        self.worksheet.set_row(
            1,
            options.tabs.Evaluation.titlebar.row1_height,
            cell_format=Formats.evaluation_titlebar_bg,
        )
        self.worksheet.set_row(
            2,
            options.tabs.Evaluation.titlebar.row2_height,
            cell_format=Formats.evaluation_titlebar_bg,
        )

        self.worksheet.merge_range("B2:D3", "")
        self.worksheet.merge_range("A2:A3", "")

        self.worksheet.write_string(
            "B2",
            options.tabs.Evaluation.titlebar.main_title,
            Formats.evaluation_titlebar_title,
        )

    def __table(self):
        display_variables = self.variables.reset_index(drop=False)[
            ["#", "Variable", "Description", "Type"]
        ]
        display_variables["Value"] = ""
        display_variables["Score"] = ""

        # Add EBM intercept to the table
        display_variables = pd.concat(
            [
                display_variables,
                pd.DataFrame(
                    {
                        "#": [""],
                        "Variable": ["Baseline score"],
                        "Description": [
                            "Baseline value for the score, should all contributions be zero."  # noqa: E501
                        ],
                        "Type": ["constant"],
                        "Value": [""],
                        "Score": [self.ebm_model.intercept_[0]],
                    }
                ),
            ],
            axis=0,
            ignore_index=True,
        )

        header = [{"header": c} for c in display_variables.columns]
        self.worksheet.add_table(
            xl_range_abs(4, 1, 4 + len(display_variables) + 2, 6),
            {
                "autofilter": 0,
                "header_row": 1,
                "columns": header,
                "data": display_variables.values.tolist(),
                "style": options.tabs.Evaluation.table.style,
            },
        )

        for index, row in enumerate(range(5, 5 + len(display_variables) - 1)):
            link_reference = f"internal:VariablePlot{index}"
            self.worksheet.write_url(row, 1, link_reference, Formats.links, str(index))

        self.worksheet.ignore_errors(
            {"number_stored_as_text": xl_range_abs(5, 1, 5 + len(display_variables), 1)}
        )

    def __prediction(self):
        model_data_tab = options.tabs.VariablesData.tab.label
        for _, var in self.variables.iterrows():
            feature_index = var["#"]
            feature_name = var.name
            feature_type = var.Type
            row_within_table = 5 + feature_index
            if feature_type != "interaction":
                row_values = (
                    self.data_reference[feature_index]["compute_values_row"] + 1
                )
                row_contributions = (
                    self.data_reference[feature_index]["compute_contributions_row"] + 1
                )
                if self.default_evaluation is not None:
                    default_evaluation_value = self.default_evaluation[feature_name]
                    if isinstance(default_evaluation_value, str):
                        self.worksheet.write_string(
                            row_within_table, 5, default_evaluation_value
                        )
                    else:
                        self.worksheet.write_number(
                            row_within_table, 5, default_evaluation_value
                        )

                self.worksheet.write_formula(
                    row_within_table,
                    7,
                    "=IF(ISBLANK({}),".format(xl_rowcol_to_cell(row_within_table, 5))
                    + '"<missing>",'
                    + "IFERROR({},".format(xl_rowcol_to_cell(row_within_table, 5))
                    + '"<missing>"))',
                    Formats.white,
                )

                if feature_type == "continuous":
                    formula = (
                        "=HLOOKUP("
                        # Lookup value
                        + "{}, ".format(xl_rowcol_to_cell(row_within_table, 7))
                        # Excel rows for values & contributions
                        + f"'{model_data_tab}'!{row_values}:{row_contributions}, "
                        # Move to second row from the table to get the contribution value
                        + "2, "
                        # Approximate search (we don't look up the exact value)
                        + "TRUE"
                        + ")"
                    )
                else:
                    formula = (
                        "=IFERROR(HLOOKUP("
                        # Lookup value
                        + "{}, ".format(xl_rowcol_to_cell(row_within_table, 7))
                        # Excel rows for values & contributions
                        + f"'{model_data_tab}'!{row_values}:{row_contributions}, "
                        # Move to second row from the table to get the contribution value
                        + "2, "
                        # exact search
                        + "FALSE"
                        + "),0)"
                    )
            else:
                term_feature = self.ebm_model.term_features_[feature_index]
                self.worksheet.write_string(row_within_table, 5, "-")
                row_contributions = (
                    self.data_reference[feature_index]["compute_contributions_row"] + 1
                )
                row_values_0 = (
                    self.data_reference[feature_index]["compute_values_row_0"] + 1
                )
                row_values_1 = (
                    self.data_reference[feature_index]["compute_values_row_1"] + 1
                )
                row_feature_0 = term_feature[0] + 5
                row_feature_1 = term_feature[1] + 5
                bins_len = self.data_reference[feature_index]["bin_lengths"][1]
                formula = (
                    "=INDEX("
                    + f"'{model_data_tab}'!{row_contributions}:{row_contributions}, "
                    + f"{bins_len}*(MATCH("
                    + "{},".format(xl_rowcol_to_cell(row_feature_0, 7))
                    + f"'{model_data_tab}'!{row_values_0}:{row_values_0},1)-2)"
                    + "+MATCH("
                    + "{},".format(xl_rowcol_to_cell(row_feature_1, 7))
                    + f"'{model_data_tab}'!{row_values_1}:{row_values_1},1)"
                    + ")"
                )
            self.worksheet.write_formula(row_within_table, 6, formula)

    def __summation(self):
        # Model score (actual summation made via table total row)
        model_scroe_row = len(self.variables) + 6
        self.worksheet.write_string(model_scroe_row, 2, "Model score", Formats.bold)
        self.worksheet.write_string(
            model_scroe_row,
            3,
            "Model score: sum of baseline + variables scores",
            Formats.bold,
        )
        self.worksheet.write_formula(
            model_scroe_row,
            6,
            "=SUM({})".format(xl_range_abs(5, 6, model_scroe_row - 1, 6)),
            Formats.bold,
        )

        # Model probability
        if options.tabs.Evaluation.display_probability:
            self.worksheet.write_string(
                model_scroe_row + 1, 2, "Model probability", Formats.bold
            )
            self.worksheet.write_string(
                model_scroe_row + 1,
                3,
                "Output probability (sigmoid applied to score)",
                Formats.bold,
            )
            model_score_cell = xl_rowcol_to_cell(model_scroe_row, 6)
            self.worksheet.write_formula(
                model_scroe_row + 1,
                6,
                f"=EXP({model_score_cell})/(1+EXP({model_score_cell}))",
                Formats.bold,
            )

    def __waterfall(self):
        model_scroe_row = len(self.variables) + 6
        baseline_row = model_scroe_row + 7
        score_row = 2 * len(self.variables) + 14

        self.worksheet.write_string(baseline_row, 2, "Baseline score", Formats.white)
        self.worksheet.write_string(score_row, 2, "Model score", Formats.white)
        self.worksheet.write_number(score_row + 1, 2, 1, Formats.white)
        self.worksheet.write_number(
            score_row + 2,
            2,
            len(self.variables) + 2,
            Formats.white,
        )
        self.worksheet.write_formula(
            baseline_row,
            3,
            "={}".format(xl_rowcol_to_cell(model_scroe_row - 1, 6)),
            Formats.white,
        )
        self.worksheet.write_formula(
            baseline_row,
            4,
            "={}".format(xl_rowcol_to_cell(baseline_row, 3)),
            Formats.white,
        )
        self.worksheet.write_formula(
            score_row + 1,
            3,
            "={}".format(xl_rowcol_to_cell(model_scroe_row - 1, 6)),
            Formats.white,
        )
        self.worksheet.write_formula(
            score_row + 2,
            3,
            "={}".format(xl_rowcol_to_cell(model_scroe_row - 1, 6)),
            Formats.white,
        )
        self.worksheet.write_formula(
            score_row + 1,
            4,
            "={}".format(xl_rowcol_to_cell(model_scroe_row, 6)),
            Formats.white,
        )
        self.worksheet.write_formula(
            score_row + 2,
            4,
            "={}".format(xl_rowcol_to_cell(model_scroe_row, 6)),
            Formats.white,
        )
        self.worksheet.write_formula(
            baseline_row,
            9,
            "={}".format(xl_rowcol_to_cell(model_scroe_row - 1, 6)),
            Formats.white,
        )

        # Columns for waterfall
        for feature_index in self.variables["#"]:
            feature_row = baseline_row + feature_index + 1
            self.worksheet.write_formula(
                feature_row,
                2,
                (
                    "={}".format(xl_rowcol_to_cell(5 + feature_index, 2))
                    + '&" = "&'
                    + "{}".format(xl_rowcol_to_cell(5 + feature_index, 5))
                ),
                Formats.white,
            )
            self.worksheet.write_formula(
                feature_row,
                3,
                "={}".format(xl_rowcol_to_cell(5 + feature_index, 6)),
                Formats.white,
            )
            # Cumulative column
            self.worksheet.write_formula(
                feature_row,
                4,
                "=SUM({})".format(xl_range_abs(baseline_row, 3, feature_row, 3)),
                Formats.white,
            )
            # Hidden column
            self.worksheet.write_formula(
                feature_row,
                5,
                (
                    "=IF({}".format(xl_rowcol_to_cell(feature_row - 1, 4))
                    + "*{}".format(xl_rowcol_to_cell(feature_row, 4))
                    + "<0,0,"
                    + "IF({}".format(xl_rowcol_to_cell(feature_row - 1, 4))
                    + "*{}".format(xl_rowcol_to_cell(feature_row, 3))
                    + ">=0,"
                    + "{},".format(xl_rowcol_to_cell(feature_row - 1, 4))
                    + "{}))".format(xl_rowcol_to_cell(feature_row, 4))
                ),
                Formats.white,
            )
            # Display column
            self.worksheet.write_formula(
                feature_row,
                6,
                (
                    "=IF({}".format(xl_rowcol_to_cell(feature_row - 1, 4))
                    + "*{}".format(xl_rowcol_to_cell(feature_row, 4))
                    + "<0,0,"
                    + "IF({}".format(xl_rowcol_to_cell(feature_row - 1, 4))
                    + "*{}".format(xl_rowcol_to_cell(feature_row, 3))
                    + ">=0,"
                    + "{},".format(xl_rowcol_to_cell(feature_row, 3))
                    + "-{}))".format(xl_rowcol_to_cell(feature_row, 3))
                ),
                Formats.white,
            )
            # Inverse negative column
            self.worksheet.write_formula(
                feature_row,
                7,
                (
                    "=IF({}".format(xl_rowcol_to_cell(feature_row - 1, 4))
                    + "*{}".format(xl_rowcol_to_cell(feature_row, 4))
                    + "<0,{},".format(xl_rowcol_to_cell(feature_row - 1, 4))
                    + "0)"
                ),
                Formats.white,
            )
            # Inverse positive column
            self.worksheet.write_formula(
                feature_row,
                8,
                (
                    "=IF({}".format(xl_rowcol_to_cell(feature_row - 1, 4))
                    + "*{}".format(xl_rowcol_to_cell(feature_row, 4))
                    + "<0,{},".format(xl_rowcol_to_cell(feature_row, 4))
                    + "0)"
                ),
                Formats.white,
            )

        column_chart = self.workbook.add_chart({"type": "column", "subtype": "stacked"})
        categories = (
            "='"
            + options.tabs.Evaluation.tab.label
            + "'!"
            + "{}".format(xl_range_abs(baseline_row, 2, score_row, 2))
        )
        column_chart.add_series(
            {
                "categories": categories,
                "values": (
                    "='"
                    + options.tabs.Evaluation.tab.label
                    + "'!"
                    + "{}".format(xl_range_abs(baseline_row, 5, score_row, 5))
                ),
                "fill": {"none": True},
            }
        )
        for col_index in [6, 7, 8]:
            column_chart.add_series(
                {
                    "categories": categories,
                    "values": (
                        "='"
                        + options.tabs.Evaluation.tab.label
                        + "'!"
                        + "{}".format(
                            xl_range_abs(baseline_row, col_index, score_row, col_index)
                        )
                    ),
                    "fill": {"color": options.tabs.Evaluation.waterfall.default_color},
                }
            )
        column_chart.add_series(
            {
                "categories": categories,
                "values": (
                    "='"
                    + options.tabs.Evaluation.tab.label
                    + "'!"
                    + "{}".format(xl_range_abs(baseline_row, 9, score_row, 9))
                ),
                "fill": {
                    "color": options.tabs.Evaluation.waterfall.intercept_and_final_color
                },
            }
        )
        scatter_chart = self.workbook.add_chart({"type": "scatter"})
        scatter_chart.add_series(
            {
                "categories": categories,
                "values": (
                    "='"
                    + options.tabs.Evaluation.tab.label
                    + "'!"
                    + "{}".format(xl_range_abs(baseline_row, 4, score_row, 4))
                ),
                "marker": {"type": "none"},
                "x_error_bars": {
                    "type": "fixed",
                    "value": 1,
                    "end_style": 0,
                    "direction": "plus",
                    "line": {"color": "#D9D9D9"},
                },
            }
        )
        scatter_chart.add_series(
            {
                "categories": (
                    "='"
                    + options.tabs.Evaluation.tab.label
                    + "'!"
                    + "{}".format(xl_range_abs(score_row + 1, 2, score_row + 2, 2))
                ),
                "values": (
                    "='"
                    + options.tabs.Evaluation.tab.label
                    + "'!"
                    + "{}".format(xl_range_abs(score_row + 1, 3, score_row + 2, 3))
                ),
                "marker": {"type": "none"},
                "line": {
                    "color": options.tabs.Evaluation.waterfall.intercept_and_final_color,
                    "dash_type": "long_dash",
                    "width": 2,
                    "transparency": 80,
                },
                "data_labels": {
                    "value": True,
                    "custom": [{"value": "Baseline score"}, {"delete": True}],
                    "font": {
                        "color": options.tabs.Evaluation.waterfall.intercept_and_final_color
                    },
                    "position": "below",
                },
            }
        )
        scatter_chart.add_series(
            {
                "categories": (
                    "='"
                    + options.tabs.Evaluation.tab.label
                    + "'!"
                    + "{}".format(xl_range_abs(score_row + 1, 2, score_row + 2, 2))
                ),
                "values": (
                    "='"
                    + options.tabs.Evaluation.tab.label
                    + "'!"
                    + "{}".format(xl_range_abs(score_row + 1, 4, score_row + 2, 4))
                ),
                "marker": {"type": "none"},
                "line": {
                    "color": options.tabs.Evaluation.waterfall.intercept_and_final_color,
                    "dash_type": "long_dash",
                    "width": 1.5,
                },
                "data_labels": {
                    "value": True,
                    "custom": [{"delete": True}, {"value": "Model score"}],
                    "font": {
                        "color": options.tabs.Evaluation.waterfall.intercept_and_final_color
                    },
                    "position": "above",
                },
            }
        )
        column_chart.combine(scatter_chart)
        column_chart.set_legend({"none": True})
        column_chart.set_y_axis(
            {
                "major_gridlines": {"visible": True, "line": {"color": "#F2F2F2"}},
                "line": {"none": True},
                "num_font": {"color": "#595959"},
            }
        )
        column_chart.set_x_axis(
            {
                "line": {"color": "#BFBFBF", "width": 2},
                "major_tick_mark": "none",
                "num_font": {
                    "color": "#595959",
                    "size": 8,
                },
                "label_position": "low",
            }
        )
        column_chart.set_title(
            {
                "name": "Prediction Waterfall",
                "name_font": {
                    "name": "Calibri",
                    "color": "#595959",
                    "size": 14,
                    "bold": False,
                },
            }
        )
        position_shift = 4 if options.tabs.Evaluation.display_probability else 5
        self.worksheet.insert_chart(
            "B{}".format(baseline_row - position_shift),
            column_chart,
            {"x_scale": 2, "y_scale": 2},
        )

    def sheet_columns(self):
        """Declare columns settings for the worksheet."""
        return [
            (0, 0, 3),
            (1, 1, 3, Formats.centered),
            (2, 2, 32, Formats.centered),
            (3, 3, 51, Formats.centered),
            (4, 4, 12, Formats.centered),
            (5, 5, 15, Formats.centered),
            (6, 6, 15, Formats.centered_2digits),
            (7, 7, 2.3),
        ]


class ExportableEBMModel:
    """
    A class to represent an exportable EBM model.

    Attributes:
    ----------
    ebm_model : ExplainableBoostingClassifier
        EBM model to explain
    output_file : str
        path to the Excel file to generate
    writer_excel : pandas.io.excel._xlsxwriter.XlsxWriter
        Excel writer placeholder
    workbook : xlsxwriter.workbook.Workbook
        Excel workbook within the Xlsxwriter
    variables : pd.DataFrame
        Data structure holding model input variables information

    Methods:
    -------
    info(additional=""):
        Prints the person's name and age.
    """

    def __init__(
        self,
        ebm_model,
        output_file: str,
        author: str = "",
        date_stamp: str = "",
    ):
        """
        Construct all the necessary attributes for the exportable model.

        Parameters:
        ----------
            ebm_model : ExplainableBoostingClassifier
                EBM model to explain
            output_file : str
                path to the Excel file to generate
        """
        self.ebm_model = ebm_model
        self.output_file = output_file

        # Excel related variables
        self.excel_writer = None
        self.workbook: Optional[Workbook] = None
        self.variables: Optional[pd.DataFrame] = None
        self.model_description: str = ""
        self.default_evaluation: Optional[Dict[str, Any]] = None

        # Other initializations
        self.__extract_variables()

    def generate(self):
        """Generate the Excel spreadsheet."""
        self.__initialize_excel_writer()

        # Overview worksheet
        self.sheet_overview = generate_overview(self.excel_writer)
        self.sheet_overview.provide_data(
            variables=self.variables,
            model_description=self.model_description,
        )
        self.sheet_overview.create_worksheet()
        log.info("Created Overview tab…")

        # Variables worksheets
        self.sheets_variables = generate_variables(self.excel_writer)
        self.sheets_variables.provide_data(
            variables=self.variables, ebm_model=self.ebm_model
        )
        self.sheets_variables.create_worksheet()
        self.data_reference = self.sheets_variables.data_reference
        log.info("Created Variables tab…")

        # Evaluation worksheet
        if options.tabs.Evaluation.enable:
            self.sheet_evaluation = generate_evaluation(self.excel_writer)
            self.sheet_evaluation.provide_data(
                variables=self.variables,
                ebm_model=self.ebm_model,
                data_reference=self.data_reference,
                default_evaluation=self.default_evaluation,
            )
            self.sheet_evaluation.create_worksheet()
            log.info("Created Evaluation tab…")
        else:
            log.info("Skipped Evaluation tab as requested within options.")

    def register_model_description(self, model_description: str):
        """Register a model description for the Overview tab."""
        self.model_description = model_description

    def register_variables(self, data):
        """Enrich variables data."""
        if data:
            if isinstance(data, dict):
                data = pd.DataFrame([data]).T
                data.columns = ["Description"]
            self.variables = self.variables.merge(
                right=data, how="left", left_index=True, right_index=True
            )
            self.variables["Description"] = self.variables["Description"].fillna("")
        else:
            self.variables["Description"] = ""

    def register_default_evaluation(self, data):
        """Enrich evaluation case with default values."""
        self.default_evaluation = data

    def save(self):
        """Export the model report into the Excel workbook."""
        self.excel_writer.close()
        log.info("File saved.")

        log.info("Successfully imported EBM model…")

    def __extract_variables(self):
        self.variables = (
            pd.DataFrame(
                data=[
                    [
                        self.ebm_model.feature_types_in_[feature_index],
                        self.ebm_model.term_names_[feature_index],
                        feature_index,
                    ]
                    if len(feature_term) == 1
                    else [
                        "interaction",
                        self.ebm_model.term_names_[feature_index],
                        feature_index,
                    ]
                    for feature_index, feature_term in enumerate(
                        self.ebm_model.term_features_
                    )
                ]
            )
            .rename(columns={0: "Type", 1: "Variable", 2: "#"})
            .set_index("Variable")
        )

        log.info(f"Model has {len(self.variables)} input variables, with types:")
        for t in self.variables.groupby("Type"):
            log.info(f" {t[0]} \t {len(t[1])}")

        def min_max_abs(x):
            return pd.Series(
                index=["Min", "Max", "Abs max"],
                data=[x.min(), x.max(), max(abs(x.min()), abs(x.max()))],
            )

        df_amplitude = pd.concat(
            [
                pd.DataFrame([t.reshape(-1) for t in self.ebm_model.term_scores_])
                .T.apply(min_max_abs)
                .T[["Min", "Max", "Abs max"]],
                pd.DataFrame(self.ebm_model.term_names_, columns=["Variable"]),
            ],
            axis=1,
        ).set_index("Variable")

        self.variables = pd.concat([self.variables, df_amplitude], axis=1)

        log.debug("Generated variables amplitude informations")

    def __initialize_excel_writer(self):
        self.excel_writer = pd.ExcelWriter(self.output_file, engine="xlsxwriter")
        self.workbook = self.excel_writer.book

        log.debug("Initialized Excel workbook writer…")


def generate_overview(writer: pd.io.excel._xlsxwriter.XlsxWriter):
    """Generate the overview worksheet."""
    return OverviewWorksheet(
        writer,
        tab_name=options.tabs.Overview.tab.label,
        tab_color=options.tabs.Overview.tab.color,
        default_zoom=options.tabs.Overview.sheet.zoom,
    )


def generate_variables(writer: pd.io.excel._xlsxwriter.XlsxWriter):
    """Generate the variables worksheets."""
    return VariablesWorksheet(
        writer,
        tab_name=options.tabs.Variables.tab.label,
        tab_color=options.tabs.Variables.tab.color,
        default_zoom=options.tabs.Variables.sheet.zoom,
    )


def generate_evaluation(writer: pd.io.excel._xlsxwriter.XlsxWriter):
    """Generate the Evaluation worksheet."""
    return EvaluationWorksheet(
        writer,
        tab_name=options.tabs.Evaluation.tab.label,
        tab_color=options.tabs.Evaluation.tab.color,
        default_zoom=options.tabs.Evaluation.sheet.zoom,
    )


def UNTESTED_to_excel_exportable(
    ebm,
    file,
    model_description=None,
    variables_description=None,
    default_evaluation=None,
):
    """Generates an Excel representation of the EBM model.

    Args:

    Returns:
        An xlsxwriter.Workbook object with an Excel representation of the model.
        This Workbook can be modified and then exported as any xlsxwriter object
        for advanced usages when custom export is required.
    """

    workbook = ExportableEBMModel(ebm, file)

    if model_description:
        workbook.register_model_description(model_description)
    if default_evaluation:
        workbook.register_default_evaluation(default_evaluation)
    workbook.register_variables(variables_description)

    workbook.generate()

    return workbook.excel_writer
