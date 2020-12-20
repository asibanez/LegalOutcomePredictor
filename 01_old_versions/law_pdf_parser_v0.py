# Converts pdf documents to txt files

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import os

input_path = 'C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\01_Local\\05_Liberty_Mutual\\Research\\Legal_stuff\\convention_eng.pdf'
output_dir = 'C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\01_Local\\10_Takeda\\OneDrive_2020-07-21\\TAEG018 MIT SLR\\parsed_to_text'
output_file = os.path.basename(input_path).split('.')[0] + '.txt'
output_path = os.path.join(output_dir, output_file)

rsrcmgr = PDFResourceManager()
retstr = StringIO()
codec = 'utf-8'
laparams = LAParams()
device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
fp = open(input_path, 'rb')
interpreter = PDFPageInterpreter(rsrcmgr, device)
password = ""
maxpages = 0
caching = True
pagenos=set()

for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,
                              caching=caching, check_extractable=True):
    interpreter.process_page(page)

text = retstr.getvalue()
fp.close()
device.close()
retstr.close()

    txt_file.write(extracted_text.encode("utf-8"))"""