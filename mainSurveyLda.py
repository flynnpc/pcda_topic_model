import csv, os
import tokenLists as tkl
import wordCloud as wc
import researchLda as rlda

research = os.path.join(os.path.dirname(__file__), "mx_survey.csv")

def main(inputCsv, columnName):
    with open(inputCsv, newline="", encoding='utf-8') as csvFile:
        reader = csv.DictReader(csvFile)
        surveyResponse = []
        for row in reader:
            if any(field.strip() for field in row[columnName]):
                surveyResponse.append(row[columnName])
    return surveyResponse

if __name__ == "__main__":

    data = main(research,'responseCombined')

    wc.mxWordCloud(tkl.cloudPrep(data))
    rlda.topicModel(tkl.tokenLda(data))
