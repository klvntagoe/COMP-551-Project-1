from collections import Counter
counter = Counter()

def preProcessing(string):
    del processd_text[:]
    processed = ""
    processed = string.lower()
    print processed
    split = processed.split()
    for word in split:
        processed_text.append(word)

def splitText(text):
    lowerCaseText =  text.lower()
    splitText = lowerCaseText.split()
    return splitText

#DATA CONSTRUCTION
def construct_dataset(d):
    for p in d:
        temp = []
        for info_name, info_value in p.items():
            if (info_name == "popularity_score"):
                y.append(info_value)
            if (info_name == "is_root"):
                if (info_value == "true"):
                    y.append(1)
                else:
                    y.append(0)
            else:
                preProcessing(info_value)
                counter = counter + Counter(processed_text)
                temp.append(processed_text)
        print(".")
        X.append(temp)
