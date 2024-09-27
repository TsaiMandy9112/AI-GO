#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 19:25:37 2023

@author: 
"""

import pandas as pd
import pyproj 

# load dataset (raw)
df=pd.read_csv('training_data.csv')

#-------------------------------------------------------------------------------'
# 「主要用途」欄位整理(精簡並歸納)
# Before: 
#    住家用      8230
#    集合住宅     2660
#    其他        471
#    商業用       263
#    一般事務所      59
#    國民住宅       29
#    住商用        11
#    工業用        11
#    辦公室        11
#    住工用         3
#    店鋪          2
#    廠房          1
# After:
#    住家用
#    集合住宅:  集合住宅,國民住宅
#    商業用:  商業用,一般事務所,住商用,辦公室,店鋪
#    工業用:  工業用,住工用,廠房
#    其他  
#-------------------------------------------------------------------------------'
dct = {'住家用':'住家用',
       '集合住宅':'集合住宅',
       '其他':'其他',
       '商業用':'商業用',
       '一般事務所':'商業用',
       '國民住宅':'集合住宅',
       '住商用':'商業用',
       '工業用':'工業用', 
       '辦公室':'商業用', 
       '住工用':'工業用',    
       '店鋪':'商業用',    
       '廠房':'工業用'      
       }   # use value_counts() to check unique value
df['main_purpose']=df['主要用途'].map(dct) 

#-------------------------------------------------------------------------------'
# 「主要建材」欄位整理(精簡並歸納)
# Before: 
#    鋼筋混凝土造       10923
#    鋼骨造            419
#    加強磚造           251
#    其他              145
#    鋼筋混凝土加強磚    12
#    磚造               1
# After:
#    加強磚造: 磚造,加強磚造
#    鋼筋混凝土造:鋼筋混凝土造,鋼筋混凝土加強磚
#    鋼骨造 
#    其他
#-------------------------------------------------------------------------------'
dct = {'鋼筋混凝土造':'鋼筋混凝土造',
       '鋼骨造':'鋼骨造',
       '加強磚造':'加強磚造',
       '其他':'其他',
       '鋼筋混凝土加強磚':'鋼筋混凝土造',
       '磚造':'加強磚造'     
       }   # use value_counts() to check unique value
df['building_material']=df['主要建材'].map(dct) 

#-------------------------------------------------------------------------------'
# 「建物型態」欄位整理(精簡並歸納)
# Before: 
#    住宅大樓(11層含以上有電梯)    7148
#    公寓(5樓含以下無電梯)       2437
#    華廈(10層含以下有電梯)      2158
#    透天厝                   8
# After 
#    公寓
#    華廈
#    住宅大樓
#    透天厝
#-------------------------------------------------------------------------------'
dct = {'住宅大樓(11層含以上有電梯)':'住宅大樓',
       '公寓(5樓含以下無電梯)':'公寓',
       '華廈(10層含以下有電梯)':'華廈',
       '透天厝':'透天厝'
       }   # use value_counts() to check unique value
df['building_type']=df['建物型態'].map(dct)

#--------------------------------------------------------------------------
# 類別變數編碼：main_purpose, building_material, building_type
#--------------------------------------------------------------------------
df1=df.copy()
# 對 main_purpose 進行One-Hot Encoding
dummy=pd.get_dummies(df1['main_purpose'])
df1=pd.concat((df1,dummy),axis=1) 
df1.drop(['其他'],axis=1, inplace=True)  # drop one colume to avoid dummy variable trap

# 對 building_material 進行One-Hot Encoding
dummy=pd.get_dummies(df1['building_material'])
df1=pd.concat((df1,dummy),axis=1) 
df1.drop(['其他'],axis=1, inplace=True)  # drop one colume to avoid dummy variable trap

# 對 building_type 進行One-Hot Encoding
dummy=pd.get_dummies(df1['building_type'])
df1=pd.concat((df1,dummy),axis=1) 
df1.drop(['透天厝'],axis=1, inplace=True)  # drop one colume to avoid dummy variable trap

#-------------------------------------------------------------------------------'
# 經緯度座標轉換 : 二度分帶座標(TWD97) -> 經緯度座標(WGS84)
# Example:
#    input  "橫坐標, 縱坐標" = "305266, 2768378" (TWD97)
#    output "經度, 緯度"    = "121.5476,25.0225" (WGS84)
#
#-------------------------------------------------------------------------------
def tran_coordination(x):
    x1, y1  = x['橫坐標'], x['縱坐標']
    proj = pyproj.Transformer.from_crs(3826, 4326, always_xy=True) #EPSG:3826(TWD97/121分帶)
    x2, y2 = proj.transform(x1, y1)  # 轉換成 lon, lat
    return x2, y2

df1[['經度','緯度']]=df1.apply(tran_coordination,axis=1,result_type='expand')


#--------------------------------------------------------------------------
# 取出想要的欄位，並重新安排欄位順序
#--------------------------------------------------------------------------
new_cols = ['經度','緯度',    # 房屋地點：經度(lon)/ 緯度(lat)
            '屋齡',
            '住家用','集合住宅','商業用','工業用',  # 主要用途
            '公寓','華廈','住宅大樓',              # 建物型態            
            '加強磚造','鋼筋混凝土造','鋼骨造',     # 主要建材
            '土地面積','建物面積',
            '主建物面積','陽台面積','附屬建物面積',
            '移轉層次','總樓層數',                #出售標的物所在樓層/總樓層
            '車位面積','車位個數',
            '單價']
df2=df1[new_cols]


#--------------------------------------------------------------------------
# Rename column name 
#--------------------------------------------------------------------------
renamed_cols={'經度':'lon','緯度':'lat',
              '屋齡':'house_age', 
              '住家用':'residence_housing','集合住宅':'congregate_housing','商業用':'commercial_use','工業用':'industrial_use',
              '公寓':'apartment','華廈':'building_low','住宅大樓':'building_high',
              '加強磚造':'RB','鋼筋混凝土造':'RC','鋼骨造':'SC',
              '土地面積': 'land_area','建物面積': 'building_area',
              '主建物面積': 'main_building_area','陽台面積': 'balcony_area','附屬建物面積':'auxiliary_area',
              '移轉層次': 'floor','總樓層數': 'total_floor',
              '車位面積': 'parking_area', '車位個數': 'parking_number',
              '單價': 'unit_price'
             }
df3=df2.rename(columns=renamed_cols)

#------------------------------------------------------------------------
# save file
#------------------------------------------------------------------------
df3.to_csv('clean_dataset.csv', index=False)  


