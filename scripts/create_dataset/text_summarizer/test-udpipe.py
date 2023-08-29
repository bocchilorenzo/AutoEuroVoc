from requests import post

class UDPipe:
    def __init__(self, url):
        self.url = url
    
    def tokenize(self, text):
        response = post(self.url, data={
            "tokenizer": "",
            "output": "horizontal",
            "data": text
        })
        if response.status_code == 200:
            return response.json()["result"].split("\n")
        else:
            raise Exception("Cannot tokenize text")

test = UDPipe("http://127.0.0.1:30101/process")
c = test.tokenize("Tuesday 14 July 2009\nMINUTES OF THE SITTING OF 14 JULY 2009\n2010/C 47 E/01\nContents\nPROCEEDINGS OF THE SITTING\n1.\nOpening of the sitting (first sitting of the newly elected Parliament)\n2.\nComposition of Parliament\n3.\nComposition of political groups\n4.\nOrder of business\n5.\nFormation of political groups\n6.\nVerification of credentials\n7.\nElection of the President of the European Parliament\n8.\nElection of Vice-Presidents (deadline for submitting nominations)\n9.\nElection of Vice-Presidents (first, second and third ballots)")

print(c)
