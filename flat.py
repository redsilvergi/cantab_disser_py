# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 04:33:33 2024

@author: eungi
"""
import os
import pandas as pd
import sys

# import numpy as np
# import matplotlib.pyplot as plt
# import math
# from sklearn.cluster import KMeans
# scikit-learn


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
dirc = "./csv/sale"
csvnms = [file for file in os.listdir(dirc) if file.endswith(".csv")]
csvnms3 = list(map(lambda x: x[:3], csvnms))

# datatypes1 = {
#     "NO": "int64",
#     "시군구": "object",
#     "번지": "object",
#     "본번": "int64",
#     "부번": "int64",
#     "단지명": "object",
#     "전용면적(㎡)": "float64",
#     "계약년월": "int64",
#     "계약일": "int64",
#     "거래금액(만원)": "object",
#     "동": "object",
#     "층": "int64",
#     "매수자": "object",
#     "매도자": "object",
#     "건축년도": "int64",
#     "도로명": "object",
#     "해제사유발생일": "object",
#     "거래유형": "object",
#     "중개사소재지": "object",
#     "등기일자": "object",
# }

# datatypes2 = {
#     "NO": "int64",
#     "시군구": "object",
#     "번지": "object",
#     "본번": "int64",
#     "부번": "int64",
#     "단지명": "object",
#     "전월세구분": "object",
#     "전용면적(㎡)": "float64",
#     "계약년월": "int64",
#     "계약일": "int64",
#     "보증금(만원)": "object",
#     "월세금(만원)": "int64",
#     "층": "int64",
#     "건축년도": "int64",
#     "도로명": "object",
#     "계약기간": "object",
#     "계약구분": "object",
#     "갱신요구권 사용": "object",
#     "종전계약 보증금(만원)": "object",
#     "종전계약 월세(만원)": "float64",
#     "주택유형": "object",
# }
##import csv -------------------------------------------------
# if len(sys.argv) < 2:
#     print("Please provide parameter")
#     sys.exit(1)

# param1 = sys.argv[1]

try:
    for i in range(len(csvnms3)):
        data = pd.read_csv(f"./csv/sale/{csvnms3[i]}23.csv", low_memory=False)
        datar = pd.read_csv(f"./csv/rent/{csvnms3[i]}23r.csv", low_memory=False)

        # data = pd.read_csv(f"./csv/{param1}.csv")
        # datar = pd.read_csv(f"./csv/{param1}r.csv")

        ##delete unnecessary columns -------------------------------------------------
        sale = data.drop(columns=["NO", "시군구", "번지", "본번", "부번"])
        rent = datar.drop(columns=["NO", "시군구", "번지", "본번", "부번"])
        # sale = data.rename(columns={'단지명':'cmplx'})
        sale.rename(
            columns={
                "단지명": "cmplx",
                "전용면적(㎡)": "area",
                "계약년월": "cntrct_ym",
                "계약일": "cntrct_d",
                "거래금액(만원)": "price",
                "동": "dong",
                "층": "floor",
                "매수자": "buyer",
                "매도자": "seller",
                "건축년도": "built",
                "도로명": "address",
                "해제사유발생일": "cancel_at",
                "거래유형": "trns_type",
                "중개사소재지": "brokerat",
                "등기일자": "reg_at",
            },
            inplace=True,
        )
        sale.dtypes
        sale["price"] = sale["price"].astype(str).str.strip().str.replace(",", "")
        sale["price"] = pd.to_numeric(sale["price"], errors="coerce")
        sale.dtypes
        ##check data for sale & rent integration -------------------------------------------------
        # f_rows = sale[(sale['address'] == '한림말길 50') & (sale['area'] == 114.78) & (sale['cancel_at']=='-')]
        # print(sale.dtypes)
        # print((sale['dong'] == '-').sum())
        # f_rows2 = rent[(rent['도로명'] == '한림말길 50') & (rent['전용면적(㎡)'] == 114.78) & (rent['전월세구분']=='월세')]
        # print(f_rows2.dtypes)
        # f_rows2.loc[:,'보증금(만원)'] = pd.to_numeric(f_rows2['보증금(만원)'].str.replace(',',''))
        # f_rows2.loc[:,'월세금(만원)'] = pd.to_numeric(f_rows2['월세금(만원)'].str.replace(',',''))
        # print(f_rows2['보증금(만원)'].mean(), f_rows2['월세금(만원)'].mean())
        # print(f_rows2.loc[164])

        ##mean sale -------------------------------------------------
        sf = (
            sale.groupby(["address", "area", "cmplx", "built"])
            .agg({"price": "mean", "floor": "mean"})
            .reset_index()
        )
        # sf2 = sale.groupby(['address','area']).agg({'price':'mean','floor':'mean'}).reset_index()
        # sf3 = sale.groupby(['address','area','cmplx','built']).agg({'price':'mean','floor':'mean'}).reset_index()
        # sf_tst = sale.groupby(['address','area']).agg({'price':'mean','floor':'mean'}).reset_index()
        # sf_tst2 = sale.groupby(['address', 'area']).agg({'cmplx': 'nunique'}).reset_index()
        # sf_tst3 = sf_tst2[sf_tst2['cmplx'] > 1]

        ##unique addresses -------------------------------------------------
        # addresses = sf['address'].unique()
        # print(addresses)
        # addf = pd.DataFrame({'address': addresses})
        # addf2 = addf.loc[0].notnull()

        ##rent columns refactor -------------------------------------------------
        print(rent.dtypes)
        print(rent.columns.tolist())
        new_col = [
            "cmplx",
            "rntchrtr",
            "area",
            "cntrct_ym",
            "cntrct_d",
            "dpst",
            "rent",
            "floor",
            "built",
            "address",
            "cntdur",
            "cntnew",
            "upright",
            "dpstold",
            "rentold",
            "prptyp",
        ]
        rent.columns = new_col
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

        ###dpstold, rentold null 91507 -> so igonre ---------------------------------
        # print((rnt['dpstold'].notnull()).sum())
        # print((rnt['rentold'].notnull()).sum())

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
        # rf2 = rnt.groupby(['address','area']).agg({'dpst':'mean', 'rent':'mean', 'floor':'mean',}).reset_index()

        ###rntduration avail 77883 out of 112779 -----------------------------------
        ### 64180: 2yrs 13703: not 2yrs so ignore -----------------------------------
        # print((rnt['cntdur'] != '-').sum())
        # rntmp = rnt[rnt['cntdur'] != '-']
        # leftdur = rntmp['cntdur'].str[:6].astype(int)
        # rightdur = rntmp['cntdur'].str[-6:].astype(int)
        # tmpres = rightdur - leftdur
        # print((tmpres == 200).sum())

        ##concat sf, rf -------------------------------------------------------
        print(sf.dtypes)
        print(rf.dtypes)
        az = sf.merge(
            rf,
            on=["address", "area", "cmplx", "built"],
            how="left",
            suffixes=("_sf", "_rf"),
        )
        # print(az['rent'].notna().sum()) #6352
        # print(az['rent'].isna().sum()) #3972
        az = az[az["rent"].notna()]
        az = az[az["address"].str.len() > 1].reset_index(drop=True)
        print(az.dtypes)

        az["floor"] = (az["floor_sf"] + az["floor_rf"]) / 2
        az["floor"] = round(az["floor"])
        az["floorDff"] = az["floor_sf"] - az["floor_rf"]

        # print(az['floorDff'].max())
        # print(az['floorDff'].min())
        # print(az['floorDff'].mean())

        # check = az[((az['floor_sf'] < 5 )&(abs(az['floorDff']) > 15))]
        # check2 = az[((az['floor_rf'] < 5 )&(abs(az['floorDff']) > 15))]

        az = az[~((az["floor_sf"] < 5) & (abs(az["floorDff"]) > 10))]
        az = az[~((az["floor_rf"] < 5) & (abs(az["floorDff"]) > 10))]

        az = az.drop(columns=["floor_sf", "floor_rf", "floorDff"])
        az["type"] = 1
        az["reg"] = switch_reg(csvnms3[i])
        az["yr"] = 23

        ##export az-------------------------------------------------------
        az.to_csv(f"./res_tmp/az_{csvnms3[i]}23.csv", index=False)
        print(f"./res_tmp/az_{csvnms3[i]}23.csv exported")
except FileNotFoundError:
    print("File not found")


# ##############################################################################
# az['pdratio'] = (az['price']*10000) / (az['dpst']*10000*0.035 + az['rent']*10000*12)
# az['dpratio'] = (az['dpst']*10000*0.035 + az['rent']*10000*12) / (az['price']*10000)

# az.memory_usage(deep=True).sum() / (1024 * 1024)

# az['g4'] = (0.04*az['pdratio']-1) / (az['pdratio']+1)
# az['r-g4'] = 0.04 - ((0.04*az['pdratio']-1) / (az['pdratio']+1))

# #az['g5'] = (0.05*az['pdratio']-1) / (az['pdratio']+1)
# #az['g6'] = (0.06*az['pdratio']-1) / (az['pdratio']+1)
# #az['g7'] = (0.07*az['pdratio']-1) / (az['pdratio']+1)
# #az['r-g5'] = 0.05 - ((0.05*az['pdratio']-1) / (az['pdratio']+1))
# #az['r-g6'] = 0.06 - ((0.06*az['pdratio']-1) / (az['pdratio']+1))
# # az['r-g7'] = 0.07 - ((0.07*az['pdratio']-1) / (az['pdratio']+1))
# #az['r-g8'] = 0.08 - ((0.08*az['pdratio']-1) / (az['pdratio']+1))
# #az['r-g9'] = 0.09 - ((0.09*az['pdratio']-1) / (az['pdratio']+1))
# #az['r-g10'] = 0.1 - ((0.1*az['pdratio']-1) / (az['pdratio']+1))
# #az['r-g15'] = 0.15 - ((0.15*az['pdratio']-1) / (az['pdratio']+1))
# #az['r-g20'] = 0.2 - ((0.2*az['pdratio']-1) / (az['pdratio']+1))
# #az['r-g30'] = 0.3 - ((0.3*az['pdratio']-1) / (az['pdratio']+1))
# #az['r-g40'] = 0.4 - ((0.4*az['pdratio']-1) / (az['pdratio']+1))
# #az['r-g50'] = 0.5 - ((0.5*az['pdratio']-1) / (az['pdratio']+1))
# #az['r-g100'] = 1 - ((1*az['pdratio']-1) / (az['pdratio']+1))

# vals2 = []
# for i in range (4, 16):
#     rg = 0.01*i - ((0.01*i*az['pdratio']-1) / (az['pdratio']+1))
#     vals2.append(rg)
# az['rg415m'] = sum(vals2) / len(vals2)

# az.memory_usage(deep=True).sum() / (1024 * 1024)

# plt.hist(az['rg415m'])


# ###############################################
# plt.plot(az['price'], az['rg415m'])
# plt.xlabel('price')
# plt.ylabel('rg415m')
# plt.title('kw')

# corrkw = az['price'].corr(az['rg415m'])
# print(corrkw)

# ##############################################
# inputdata= az[['price','rg415m','built','dpst','rent','floor','area']]
# kmeans = KMeans(n_clusters=4)
# kmeans.fit(inputdata)

# centroids = kmeans.cluster_centers_
# labels = kmeans.labels_

# print("Centroids:", centroids)
# print("Labels:", labels)

# plt.scatter(az['rg415m'], az['price'], c=labels, cmap='viridis')
# plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red', label='Centroids')
# plt.xlabel('prc')
# plt.ylabel('rg')
# plt.title('K-means Clustering')
# plt.legend()
# plt.show()
