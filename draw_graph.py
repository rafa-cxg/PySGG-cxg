import random

import numpy as np
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
predicate_new_order = [10, 42, 43, 34, 28, 17, 19, 7, 29, 33, 18, 35, 32, 27, 50, 22, 44, 45, 25, 2, 9, 5, 15, 26, 23, 37, 48, 41, 6, 4, 1, 38, 21, 46, 30, 36, 47, 14, 49, 11, 16, 39, 13, 31, 40, 20, 24, 3, 12, 8]
predicate_new_order_name = ['on', 'has', 'wearing', 'of', 'in', 'near', 'behind', 'with', 'holding', 'above', 'sitting on', 'wears', 'under', 'riding', 'in front of', 'standing on', 'at', 'carrying', 'attached to', 'walking on', 'over', 'for', 'looking at', 'watching', 'hanging from', 'laying on', 'eating', 'and', 'belonging to', 'parked on', 'using', 'covering', 'between', 'along', 'covered in', 'part of', 'lying on', 'on back of', 'to', 'walking in', 'mounted on', 'across', 'against', 'from', 'growing on', 'painted on', 'playing', 'made of', 'says', 'flying in']
order=[]
for i in predicate_new_order:
    order.append(i-1)
# file1 = open('transformer_human.txt', 'r')# ours
# file2 = open('transformer.txt', 'r')# baseline
# file1 = open('transformer_human_sgcls.txt', 'r')# ours
# file2 = open('transformer_sgcls.txt', 'r')# baseline
# file1 = open('transformer_human_sgdet.txt', 'r')# ours
# file2 = open('transformer_sgdet.txt', 'r')# baseline
# file1 = open('bgnn_base_human_sgdet.txt', 'r')# ours
# file2 = open('bgnn_base_sgdet.txt', 'r')# baseline
# file1 = open('motif_bls_human.txt', 'r')# ours
# file2 = open('motif_bls_tde.txt', 'r')# baseline
# file1 = open('bgnn_bls_human.txt', 'r')# ours
file2 = open('bgnn_bls.txt', 'r')# baseline
file1 = open('bgnn_bls_human.txt', 'r')# ours
# file2 = open('bgnn_bls_rtpb.txt', 'r')# baseline
data1 = file1.read()
data2 = file2.read()
dict1={}
dict2={}
for num_str in  re.findall('\((.*?)\)',data1):
    name,value=num_str.split(':')
    dict1[name]=float(value)
d1=np.array(list(dict1.values()))

for num_str in  re.findall('\((.*?)\)',data2):
    name,value=num_str.split(':')
    dict2[name]=float(value)
d2=np.array(list(dict2.values()))

width = 0.3  # the width of the bars
xlabels = list(dict1.keys())
new_xlabels=[]
for i in predicate_new_order_name:
    new_xlabels.append(xlabels.index(i))
d1=d1[new_xlabels]
d2=d2[new_xlabels]
x = np.arange(len(new_xlabels))
plt.figure(figsize=(8, 2))  # 设置画布的尺寸
# plt.title('Examples of Histogram', fontsize=20)  # 标题，并设定字号大小
# plt.xlabel(u'relation', fontsize=8)  # 设置x轴，并设定字号大小
plt.ylabel(u'possbility', fontsize=8)  # 设置y轴，并设定字号大小

# alpha：透明度；width：柱子的宽度；facecolor：柱子填充色；edgecolor：柱子轮廓色；lw：柱子轮廓的宽度；label：图例；
plt.rcParams.update({'font.size': 6})
draw_relative=True
if draw_relative:
    plt.ylabel(u'improvement', fontsize=8)  # 设置y轴，并设定字号大小
    relative=d1-d2
    colors=['darkturquoise' for i in range(0,len(relative))]
    idx=0
    for i1,i2,color in (zip(d1,d2,colors)):
        if i1<=0.000:
            colors[idx]='k'
        if i2<=0.0002 :
            colors[idx]='r'
        idx=idx+1
    plt.bar(x, relative, width, color=colors,label='BGNN+BLS')
    plt.xticks(x, predicate_new_order_name, rotation=90, fontsize=7)
    # plt.legend(loc=1)  # 图例展示位置，数字代表第几象限
    plt.tight_layout()
    plt.savefig("relative_predicate.pdf", dpi=500)
    plt.show()  # 显示图像
else:
    plt.bar(x - width/2, d1,width, label='LANDMARK')
    plt.bar(x + width/2, d2,width, label='BGNN')
    plt.xticks(x,predicate_new_order_name,rotation=90,fontsize=7)
    plt.legend(loc=1)  # 图例展示位置，数字代表第几象限
    plt.tight_layout()
    plt.savefig("pred_compare.pdf",dpi=500)
    plt.show()  # 显示图像
#############################################
# plt.figure(figsize=(15,5))
# n = [0.1,0.2,0.3,0.5,0.6,0.7,0.8,1.0]
# imp_factor=[0.1,0.2,0.3,0.5,0.7,1.0]
# tran_factor=[0.1,0.3,0.5,0.7,1.0]
# bgnn_bivl = [16.09, 16.99,17.50,16.9,16.83,16.93,17.11,17.02]
# imp=[8.29,8.33,6.91,7.79,8.45,7.92]#7.92是union而不是fusion
# transformer=[10.24,10.68,10.47,10.01,10.26]
# SHA_factor=[0.1,0.3,0.5,0.7,1.0]
# SHA_GCL=[18.91,19.50,18.51,18.15,18.20]
# motif_factor=[0.3,0.7,1]
# motif=[8.83,8.19,]
# plt.plot(n, bgnn_bivl,marker = '.',markersize = 7,label='BGNN+BLS')
# plt.plot(imp_factor, imp,marker = 'v',markersize = 5,label='IMP')
# plt.plot(tran_factor, transformer,marker = '*',markersize = 5,label='Transformer')
# plt.plot(SHA_factor, SHA_GCL,marker = 'o',markersize = 5,label='SHA+GCL')
# plt.xlabel('factor')
# plt.ylabel('mR@100')
# plt.legend()
# plt.savefig("factor.pdf",dpi=500,bbox_inches='tight')
# plt.show(bbox_inches='tight')


################for TDE RESULT ELLUSTRATION##################
# bgnn_m=[32.76,17.83,15.56]
# bgnn_tde_m=[34.89,19.28,14.32]
# bgnn_r=[56.8,36.94,27.56]
# bgnn_tde_r=[33.33,21.43,24.11]
# m_improve=[]
# r_improve=[]
# #计算差值
# for b_m,b_t_m in zip(bgnn_m,bgnn_tde_m):
#     m_improve.append((b_t_m-b_m)/b_m)
# for b_r, b_t_r in zip(bgnn_r, bgnn_tde_r):
#     r_improve.append((b_t_r - b_r)/b_r)
# width = 0.25
# plt.figure(figsize=(3,3))
# x = np.arange(3)
# xlabels=['Predcls','SgCls','SGGen']
# plt.ylabel('relative improvement:(%)', fontsize=10)
# plt.rcParams.update({'font.size': 8})
# d1=[1,2,5]
# d2=[4,4,4]
# plt.bar(x - width/2, m_improve,width, label='mR@100')
# plt.bar(x + width/2, r_improve,width, label='R@100')
# plt.tick_params(labelsize=7)#改变刻度字号
# plt.xticks(x,xlabels,rotation=0,fontsize=10)
# plt.legend(loc=1)  # 图例展示位置，数字代表第几象限
# plt.tight_layout()
# plt.savefig("bgnn.pdf",dpi=500,bbox_inches='tight')
# plt.show(bbox_inches='tight')
################baseline model mean recall######################
# fig, ax = plt.subplots()
# model_name=['IMP','Transformer','G-rcnn','Motifs','BGNN','SHA','DTrans','KERN','VCTree']
# mean_recall=[7.3,9.68,6.24,8.83,8.82,7.46,10.8,7.3,8.6]
# year=[2017,2017,2018,2018,2021,2022,2022,2019,2019]
# year_no_overlap=[2017,2018,2019,2021,2022]
# # mark=random.sample(range(5,20),9)
# mark=["2","o","^","3","P","*",9,"X","D"]
# plt.xticks(np.arange(len(year_no_overlap)),year_no_overlap,rotation=0,fontsize=10)
# for x,i in enumerate(mean_recall):
#     y=year[x]
#     idx=year_no_overlap.index(y)
#     scatter = ax.scatter(idx,i,marker=mark[x],s=50)
#
#     plt.text(idx,i, model_name[x],size=10)
# plt.xlabel('Year')
# plt.ylabel('mR@100(%)')
# plt.tight_layout()
# plt.savefig("modelrecall.pdf",dpi=500,bbox_inches='tight')
# plt.show(bbox_inches='tight')