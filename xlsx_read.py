# coding: utf-8
import xlrd
def result(result):
    data = xlrd.open_workbook('fenlei.xlsx')
    sheet = data.sheets()[0]
    cols = sheet.col_values(0) #第一列内容
    length = len(cols)
    for i in range(length):
        if cols[i] == int(result):
            return sheet.row_values(i)#该行内容