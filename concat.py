# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:06:47 2024

@author: eungi
"""

import os
import pandas as pd

##concat results from different regions -------------------------------------------------

dirc = "./csv/az_csv/multi"
csvnms = [file for file in os.listdir(dirc) if file.endswith(".csv")]

dfs = []

for i in range(len(csvnms)):
    dfs.append(pd.read_csv(f"{dirc}/{csvnms[i]}", low_memory=False))

az = pd.concat(dfs, ignore_index=True)

az.to_csv("./res_tmp/az_concated.csv", index=False)
print("./res_tmp/az_concated.csv exported")


##concat mrgds ------------------------------------------------
# flat = pd.read_csv('./csv/mrgd/mrgd_flat23.csv', low_memory=False)
# officetel = pd.read_csv('./csv/mrgd/mrgd_officetel23.csv', low_memory=False)
# multi = pd.read_csv('./csv/mrgd/mrgd_multi23.csv', low_memory=False)

# az = pd.concat([flat,officetel,multi], ignore_index=True)
# az.memory_usage(deep=True).sum() /(1024**2)

# az.to_csv('./res_tmp/az_fom.csv', index=False)
# print('./res_tmp/az_fom.csv exported')
