import os
import numpy as np
import pandas as pd

labels = pd.read_csv('metadata.csv')[['label', 'context']]
print(labels)


