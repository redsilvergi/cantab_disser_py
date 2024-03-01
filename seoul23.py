# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 01:51:02 2024

@author: eungi
"""

import pandas as pd
import numpy as np

##import csv -------------------------------------------------
seoul23 = pd.read_csv('seoul23.csv')
seoul23r = pd.read_csv('seoul23r.csv')

##delete unnecessary columns -------------------------------------------------
sale = seoul23.drop(columns=['NO','시군구','번지','본번','부번'])
rent = seoul23r.drop(columns=['NO','시군구','번지','본번','부번'])
#sale = seoul23.rename(columns={'단지명':'cmplx_nm'})
sale.rename(columns={'단지명':'cmplx_nm','전용면적(㎡)':'area', '계약년월': 'cntrct_ym','계약일':'cntrct_d','거래금액(만원)':'price','동':'dong','층':'floor','매수자':'buyer','매도자':'seller','건축년도':'built_y','도로명':'address','해제사유발생일':'cancel_at','거래유형':'trns_type','중개사소재지':'brokerat','등기일자':'reg_at'}, inplace=True)

##check data for sale & rent integration -------------------------------------------------
f_rows = sale[(sale['address'] == '한림말길 50') & (sale['area'] == 114.78) & (sale['cancel_at']=='-')]
print(sale.dtypes)
print((sale['dong'] == '-').sum())
f_rows2 = rent[(rent['도로명'] == '한림말길 50') & (rent['전용면적(㎡)'] == 114.78) & (rent['전월세구분']=='월세')]
print(f_rows2.dtypes)
f_rows2.loc[:,'보증금(만원)'] = pd.to_numeric(f_rows2['보증금(만원)'].str.replace(',',''))
f_rows2.loc[:,'월세금(만원)'] = pd.to_numeric(f_rows2['월세금(만원)'].str.replace(',',''))
print(f_rows2['보증금(만원)'].mean(), f_rows2['월세금(만원)'].mean())
#print(f_rows2.loc[164])

##mean sale -------------------------------------------------
sf = sale.groupby(['address','area','cmplx_nm','built_y']).agg({'price':'mean','floor':'mean'}).reset_index()
sf2 = sale.groupby(['address','area']).agg({'price':'mean','floor':'mean'}).reset_index()
sf3 = sale.groupby(['address','area','cmplx_nm','built_y']).agg({'price':'mean','floor':'mean'}).reset_index()
#sf_tst = sale.groupby(['address','area']).agg({'price':'mean','floor':'mean'}).reset_index()
#sf_tst2 = sale.groupby(['address', 'area']).agg({'cmplx_nm': 'nunique'}).reset_index()
#sf_tst3 = sf_tst2[sf_tst2['cmplx_nm'] > 1]

##unique addresses -------------------------------------------------
addresses = sf['address'].unique()
print(addresses)
addf = pd.DataFrame({'address': addresses})
addf.loc[0].notnull()

##rent columns refactor -------------------------------------------------
print(rent.dtypes)
print(rent.columns.tolist())
new_col = ['cmplx_nm', 'rntchrtr', 'area', 'cntrct_ym', 'cntrct_d','dpst','rent','floor','built_y', 'address', 'cntdur','cntnew','upright','dpstold','rentold','prptyp'] 
rent.columns = new_col
print(rent.dtypes)
rnt = rent[rent['rntchrtr'] =='월세']

## dpst, rent as numeric --------------------------------------------------
print(rnt.dtypes)
rnt.loc[:,'dpst'] = rnt['dpst'].astype(str).str.strip()
rnt.loc[:,'dpst'] = rnt['dpst'].str.replace(',', '')
rnt.loc[:,'dpst'] = pd.to_numeric(rnt['dpst'])
rnt.loc[:,'rent'] = rnt['rent'].astype(str).str.strip()
rnt.loc[:,'rent'] = rnt['rent'].str.replace(',', '')
rnt.loc[:,'rent'] = pd.to_numeric(rnt['rent'])
print(rnt.dtypes)

###dpstold, rentold null 91507 -> so igonre ---------------------------------
#print((rnt['dpstold'].notnull()).sum())
#print((rnt['rentold'].notnull()).sum())

##mean rent ------------------------------------------------------------------
rf = rnt.groupby(['address','area','cmplx_nm','built_y']).agg({'dpst':'mean', 'rent':'mean', 'floor':'mean',}).reset_index()
rf2 = rnt.groupby(['address','area']).agg({'dpst':'mean', 'rent':'mean', 'floor':'mean',}).reset_index()

###rntduration avail 77883 out of 112779 -----------------------------------
### 64180: 2yrs 13703: not 2yrs so ignore -----------------------------------
#print((rnt['cntdur'] != '-').sum())
#rntmp = rnt[rnt['cntdur'] != '-']
#leftdur = rntmp['cntdur'].str[:6].astype(int)
#rightdur = rntmp['cntdur'].str[-6:].astype(int)
#tmpres = rightdur - leftdur 
#print((tmpres == 200).sum())


##concat sf, rf -------------------------------------------------------
print(sf.dtypes)
print(rf.dtypes)
mrgd = sf.merge(rf, on=['address', 'area'], how='left', suffixes=('_sf', '_rf'))
#print(mrgd['rent'].notna().sum()) #6352
#print(mrgd['rent'].isna().sum()) #3972
mrgd2 = mrgd[mrgd['rent'].notna()]
mrgd3 = mrgd2[mrgd2['address'].str.len() > 1]

##export mrgd3-------------------------------------------------------
#mrgd3.to_csv('mrgd3.csv', index=False)
