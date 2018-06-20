
import sys
import pandas


def xls2csv (inputfilename):
    # converts xlsx file to csv file
    outputfilename = str(inputfilename).replace('xlsx', 'csv')
    xls_file = pandas.read_excel(inputfilename)
    xls_file.to_csv(outputfilename, index=False)

xls2csv('Prospects.xlsx')
xls2csv('Sales.xlsx')

print(sys.path)




