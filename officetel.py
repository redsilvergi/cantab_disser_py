# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 01:36:43 2024

@author: eungi
"""

import os
import pandas as pd
import sys


## AUXILIARY ------------------------------------------------
def switch_reg(v):
    if v == "sol":
        return 11
    elif v == "bus":
        return 26
    elif v == "dag":
        return 27
    elif v == "inc":
        return 28
    elif v == "gwj":
        return 29
    elif v == "daj":
        return 30
    elif v == "uls":
        return 31
    elif v == "sej":
        return 36
    elif v == "gyg":
        return 41
    elif v == "gnw":
        return 51
    elif v == "ncc":
        return 43
    elif v == "scc":
        return 44
    elif v == "njl":
        return 52
    elif v == "sjl":
        return 46
    elif v == "ngs":
        return 47
    elif v == "sgs":
        return 48
    elif v == "jju":
        return 50


##import csv -------------------------------------------------
dirc = "./csv/2023/sale/officetel"
csvnms = [file for file in os.listdir(dirc) if file.endswith(".csv")]
csvnms3 = list(map(lambda x: x[:3], csvnms))

##import csv -------------------------------------------------
# if len(sys.argv) < 2:
#     print("Please provide parameter")
#     sys.exit(1)

# param1 = sys.argv[1]

try:
    for i in range(len(csvnms3)):
        data = pd.read_csv(f"./csv/2023/sale/officetel/{csvnms3[i]}.csv", encoding='ansi', skiprows=15, low_memory=False)
        datar = pd.read_csv(f"./csv/2023/rent/officetel/{csvnms3[i]}.csv", encoding='ansi', skiprows=15, low_memory=False)

        ##delete unnecessary columns -------------------------------------------------

        # data = pd.read_csv(
        #     "./csv/2023/sale/officetel/sej.csv",
        #     encoding="ansi",
        #     skiprows=15,
        #     low_memory=False,
        # )
        # datar = pd.read_csv(
        #     "./csv/2023/rent/officetel/sej.csv",
        #     encoding="ansi",
        #     skiprows=15,
        #     low_memory=False,
        # )

        sale_del_col = [
            "NO",
            "시군구",
            "번지",
            "본번",
            "부번",
            "계약년월",
            "계약일",
            "매수",
            "매도",
            "해제사유발생일",
            "거래유형",
            "중개사소재지",
        ]
        rent_del_col = [
            "NO",
            "시군구",
            "번지",
            "본번",
            "부번",
            "전월세구분",
            "계약년월",
            "계약일",
            "계약기간",
            "계약구분",
            "갱신요구권 사용",
            "종전계약 보증금(만원)",
            "종전계약 월세(만원)",
        ]

        sale = data.drop(columns=sale_del_col)
        rent = datar.drop(columns=rent_del_col)

        sale_col = [
            "cmplx",
            "area",
            "price",
            "floor",
            "built",
            "address"
        ]

        sale.columns = sale_col
        sale.dtypes
        sale["price"] = sale["price"].astype(str).str.strip().str.replace(",", "")
        sale["price"] = pd.to_numeric(sale["price"], errors="coerce")
        sale.dtypes

        ##mean sale -------------------------------------------------
        sf = (
            sale.groupby(["address", "area", "cmplx", "built"])
            .agg({"price": "mean", "floor": "mean"})
            .reset_index()
        )
        ##rent columns refactor -------------------------------------------------
        print(rent.dtypes)
        print(rent.columns.tolist())
        rent_col = [
            "cmplx",
            "area",
            "dpst",
            "rent",
            "floor",
            "built",
            "address"
        ]
        rent.columns = rent_col
        print(rent.dtypes)
        # rnt = rent[rent["rntchrtr"] == "월세"]
        rnt = rent

        ## dpst, rent as numeric --------------------------------------------------
        print(rnt.dtypes)
        rnt["dpst"] = rnt["dpst"].astype(str).str.strip().str.replace(",", "")
        rnt["dpst"] = pd.to_numeric(rnt["dpst"], errors="coerce")
        rnt["rent"] = rnt["rent"].astype(str).str.strip().str.replace(",", "")
        rnt["rent"] = pd.to_numeric(rnt["rent"], errors="coerce")
        print(rnt[rnt["dpst"].isna()])
        print(rnt[rnt["rent"].isna()])
        print(rnt.dtypes)

        ##mean rent ------------------------------------------------------------------
        rf = (
            rnt.groupby(["address", "area", "cmplx", "built"])
            .agg(
                {
                    "dpst": "mean",
                    "rent": "mean",
                    "floor": "mean",
                }
            )
            .reset_index()
        )

        ##concat sf, rf -------------------------------------------------------
        print(sf.dtypes)
        print(rf.dtypes)
        az = sf.merge(
            rf,
            on=["address", "area", "cmplx", "built"],
            how="left",
            suffixes=("_sf", "_rf"),
        )
        az = az[az["rent"].notna()]
        az = az[az["address"].str.len() > 1].reset_index(drop=True)
        print(az.dtypes)

        az["floor"] = (az["floor_sf"] + az["floor_rf"]) / 2
        az["floor"] = round(az["floor"])
        az["floorDff"] = az["floor_sf"] - az["floor_rf"]

        az = az[~((az["floor_sf"] < 5) & (abs(az["floorDff"]) > 10))]
        az = az[~((az["floor_rf"] < 5) & (abs(az["floorDff"]) > 10))]

        az = az.drop(columns=["floor_sf", "floor_rf", "floorDff"])
        az["type"] = 2
        az["reg"] = switch_reg(csvnms3[i])
        az["yr"] = 23

        ##export az-------------------------------------------------------
        az.to_csv(f"./res_tmp/az_{csvnms3[i]}_officetel.csv", index=False)
        print(f"./res_tmp/az_{csvnms3[i]}_officetel.csv exported")
except FileNotFoundError:
    print("File not found")
