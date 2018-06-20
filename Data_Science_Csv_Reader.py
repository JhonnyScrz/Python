import csv

datelist = []
valueslist = []
mydict = {}
inputfilename = 'TrendData.csv'
outputfilename = 'output' + inputfilename


def output_trend_data():
    with open(inputfilename) as csvfileRead:
        reader = csv.DictReader(csvfileRead)
        for row in reader:
            datelist.append(row['Date'])
            valueslist.append(row['Value'])

    with open(outputfilename, 'w') as outputfile:
        with open(inputfilename) as inputfile:
            reader = csv.DictReader(inputfile)
            fieldnames = ['StartDate', 'EndDate', 'StartValue', 'EndValue']
            writer = csv.DictWriter(outputfile, fieldnames=fieldnames)
            writer.writeheader()
            i = 0

            for row in reader:
                datelist.append(row['Date'])
                valueslist.append(row['Value'])
                startDate = row['Date']
                startValue = row['Value']
                # unused trend data
                # trend = float(startValue) - float(valueslist[i+1])
                mydict[startDate] = startValue
                writer.writerow({'StartDate': startDate, 'EndDate': datelist[i + 1], 'StartValue': startValue,
                                 'EndValue': valueslist[i + 1]})
                i += 1

            print("number of dates found: " + str(len(mydict)))


output_trend_data()

