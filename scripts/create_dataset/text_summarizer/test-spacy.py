import spacy

text = "Tuesday 14 July 2009. This is a new day. Or a new sentence.\nMINUTES OF THE SITTING OF 14 JULY 2009\n2010/C 47 E/01\nContents\nPROCEEDINGS OF THE SITTING\n1.\nOpening of the sitting (first sitting of the newly elected Parliament)\n2.\nComposition of Parliament\n3.\nComposition of political groups\n4.\nOrder of business\n5.\nFormation of political groups\n6.\nVerification of credentials\n7.\nElection of the President of the European Parliament\n8.\nElection of Vice-Presidents (deadline for submitting nominations)\n9.\nElection of Vice-Presidents (first, second and third ballots)"

nlp = spacy.load("en_core_web_lg")
nlp.max_length = 1000000

parts = text.split("\n")
sent_id = 0
for p in parts:
    p = p.strip()
    doc = nlp(p)
    for sent in doc.sents:
        text = sent.text.strip()
        if text:
            sent_id += 1
            print(sent_id, "=>", text)

