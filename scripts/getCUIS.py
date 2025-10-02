
import pandas as pd
import glob
import os

loc = "/raid/SUIT/Archive/ML_Substance_Misuse/Model_Notes_NonN_24/"

all_files = glob.glob(os.path.join(loc,"*.txt"))

df = pd.DataFrame({"NAME":all_files,"CUIS":all_files})

df["CUIS"] = df["CUIS"].apply(lambda x: open(x,"r").read())

def con(x):
    x = x.split()
    return len(x)

df["LEN"] = df["CUIS"].apply(lambda x: con(x))

print(df["LEN"].describe())

