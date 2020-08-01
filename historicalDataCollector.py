import urllib.request
def main():
    # takes the csv file and put the companies symbol to a list
    file = open("constituents_csv.csv", "r")
    f1 = file.readlines()
    companies = [""]
    for i in f1:
        # need to filter the second word and put it in a list?
        symbol = i.partition(",")
        companies.insert(0, symbol[0])
    for i in companies:
        print(i);
        url = "https://query1.finance.yahoo.com/v7/finance/download/" + i + "?period1=1437955200&period2=1595808000&interval=1d&events=history"
        urllib.request.urlretrieve(url,'result/' + i + ".csv")
    f.close()
    file.close()
if  __name__  ==  '__main__':
    main()
