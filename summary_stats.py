#gets summary statistics for review lengths

import pandas as pd
import IPython

#load the reviews    
x_file = open("x_unsplit_balanced.txt", "r", encoding = "utf-8")
x_unsplit_balanced = x_file.readlines()

x_usb_lengths = pd.Series([len(line) for line in x_unsplit_balanced])

IPython.embed()