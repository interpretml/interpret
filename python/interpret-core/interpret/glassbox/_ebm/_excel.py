# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import xlsxwriter


def UNTESTED_to_excel_exportable(ebm, file):
    """Generates an Excel representation of the EBM model.

    Args:

    Returns:
        An xlsxwriter.Workbook object with an Excel representation of the model.
        This Workbook can be modified and then exported as any xlsxwriter object
        for advanced usages when custom export is required.
    """

    workbook = xlsxwriter.Workbook(file)
    worksheet = workbook.add_worksheet()

    worksheet.write(0, 0, "Hello EBM")

    return workbook
