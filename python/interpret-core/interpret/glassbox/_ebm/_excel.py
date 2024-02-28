# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from typing import Any, Dict, Optional
from xlsxwriter.utility import xl_range_abs, xl_rowcol_to_cell
from xlsxwriter.workbook import Workbook
import pandas as pd
import numpy as np
import dotsi

import logging

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
            "toc_description": (
                "Shape plots of the variables."
            ),
        },
        # Evaluation tab: ability to evaluate model on one point
        "Evaluation": {
            "enable": False,
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
        Formats.overview_toc_title = Format(
            options.tabs.Overview.toc.title
        ).register(self.workbook)
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
        for feature_index, feature_name in enumerate(self.ebm_model.feature_names_in_):
            start_row_feature = 6 * feature_index
            self.__generate_feature_data(start_row_feature, feature_index, feature_name)

    def __generate_feature_data(self, start_row, feature_index, feature_name):
        row_label = f"{feature_index}_{feature_name}"
        
        if self.ebm_model.feature_types_in_[feature_index] == "continuous":
            # Continuous
            x_compute = np.concatenate(
                [
                    [np.NaN, -1e12],
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
                distribution_x = self.ebm_model.bins_[
                    feature_index
                ][0]
                distribution_y = self.ebm_model.bin_weights_[
                    feature_index
                ][1:-1]
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
                        self.ebm_model.feature_bounds_[feature_index][1]
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

        elif self.ebm_model.feature_types_in_[feature_index] == "nominal":
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
        # TODO: add interaction features

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
        for feature_index, feature_name in enumerate(self.ebm_model.feature_names_in_):
            feature_type = self.ebm_model.feature_types_in_[feature_index]
            next_row = self.__shape_plot_doc(feature_index, feature_name, feature_type, start_row)
            # Next start row
            start_row = next_row

    def __shape_plot_doc(self, feature_index, feature_name, feature_type, start_row):
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

        if feature_type == "continuous":
            chart = self.workbook.add_chart(
                {"type": "scatter", "subtype": "straight"}
            )
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
        elif feature_type == "nominal":
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
        #self.default_evaluation: Optional[Dict[str, Any]] = None

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

        # Evaluation worksheet: TODO
        if options.tabs.Evaluation.enable:
            self.sheet_evaluation = evaluation.generate_evaluation(self.excel_writer)
            self.sheet_evaluation.provide_data(
                variables=self.variables,
                ebm_model=self.ebm_model,
                data_reference=self.data_reference,
                default_evaluation=self.default_evaluation,
                group_parameters=self.group_parameters,
            )
            self.sheet_evaluation.create_worksheet()
            log.info("Created Evaluation tab…")
        else:
            log.info("Skipped Evaluation tab as requested within options.")

    def register_model_description(self, model_description: str):
        """Register a model description for the Overview tab."""
        self.model_description = model_description

    def save(self):
        """Export the model report into the Excel workbook."""
        self.excel_writer.close()
        log.info("File saved.")

        log.info("Successfully imported EBM model…")

    def __extract_variables(self):
        self.variables = (
            pd.DataFrame(
                data=[
                    self.ebm_model.feature_types_in_,
                    self.ebm_model.feature_names_in_,
                    [f"{i}" for i in range(len(self.ebm_model.feature_names_in_))],
                ]
            )
            .T.rename(columns={0: "Type", 1: "Variable", 2: "#"})
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
                pd.DataFrame(self.ebm_model.feature_names_in_, columns=["Variable"]),
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

def UNTESTED_to_excel_exportable(ebm, file, model_description=None):
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

    workbook.generate()

    return workbook.excel_writer
