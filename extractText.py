from PyPDF2 import PdfReader
import re
from icecream import ic
import csv
text = []
with open("contract.pdf", "rb") as pdfFileObj:
    pdfReader = PdfReader(pdfFileObj)
    for pageNum in range(16, 134):
        pageObj = pdfReader.pages[pageNum]
        text.append(pageObj.extract_text())

sections = []

for t in text:
    sections += t.split("Section")

# Removing empty strings
sections = [s for s in sections if s]


# Creating a list of lists to store the data
data = []

# Iterating through each section
for section in sections:
    lines = section.split("\n")
    title = lines[0].strip()
    content = "\n".join(lines[1:]).strip()
    data.append([title, content])

# Writing the data to a CSV file
with open('output.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['title', 'content'])
    for row in data:
        # to find empty list with '' contnent eg:- ['3','']
        if not row[1]=='':
            print('found a empty row',row)
            writer.writerow(row)

