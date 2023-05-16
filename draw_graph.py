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
#
# file2 = open('bgnn_bls.txt', 'r')# baseline
# file1 = open('bgnn_bls_human.txt', 'r')# ours
#
# data1 = file1.read()
# data2 = file2.read()
# dict1={}
# dict2={}
# for num_str in  re.findall('\((.*?)\)',data1):
#     name,value=num_str.split(':')
#     dict1[name]=float(value)
# d1=np.array(list(dict1.values()))
#
# for num_str in  re.findall('\((.*?)\)',data2):
#     name,value=num_str.split(':')
#     dict2[name]=float(value)
# d2=np.array(list(dict2.values()))
#
# width = 0.5  # the width of the bars
# xlabels = list(dict1.keys())
# new_xlabels=[]
# for i in predicate_new_order_name:
#     new_xlabels.append(xlabels.index(i))
# d1=d1[new_xlabels]
# d2=d2[new_xlabels]
# x = np.arange(len(new_xlabels))
#
# fig, ax =plt.subplots(figsize=(10, 2))  # 设置画布的尺寸
# # ax.spines['right'].set_visible(False)
# # ax.spines['top'].set_visible(False)
# # plt.axis('off')
# # plt.title('Examples of Histogram', fontsize=20)  # 标题，并设定字号大小
# # plt.xlabel(u'relation', fontsize=8)  # 设置x轴，并设定字号大小
# plt.ylabel(u'possbility', fontsize=8)  # 设置y轴，并设定字号大小
#
# # alpha：透明度；width：柱子的宽度；facecolor：柱子填充色；edgecolor：柱子轮廓色；lw：柱子轮廓的宽度；label：图例；
# plt.rcParams.update({'font.size': 6})
# draw_relative=True
# if draw_relative:
#     plt.ylabel(u'improvement', fontsize=8)  # 设置y轴，并设定字号大小
#     relative=d1-d2
#     colors=['darkturquoise' for i in range(0,len(relative))]
#     idx=0
#     for i1,i2,color in (zip(d1,d2,colors)):
#         if i1<=0.000:
#             colors[idx]='k'
#         if i2<=0.0002 :
#             colors[idx]='r'
#         idx=idx+1
#     plt.bar(x, relative, width, color=colors,label='BGNN+BLS')
#     plt.xlim(-0.5,49.5)
#     plt.xticks(x, predicate_new_order_name, rotation=45, fontsize=5)
#     # plt.legend(loc=1)  # 图例展示位置，数字代表第几象限
#     plt.tight_layout()
#
#     plt.savefig("relative_predicate.pdf", dpi=500)
#     plt.show()  # 显示图像
# else:
#     plt.bar(x - width/2, d1,width, label='LANDMARK')
#     plt.bar(x + width/2, d2,width, label='BGNN')
#     plt.xticks(x,predicate_new_order_name,rotation=90,fontsize=7)
#     plt.legend(loc=1)  # 图例展示位置，数字代表第几象限
#     plt.tight_layout()
#     plt.savefig("pred_compare.pdf",dpi=500)
#     plt.show()  # 显示图像
############### evaluation eem factor ##############################

plt.figure(figsize=(8,6),dpi=500)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,dpi=500)
fig.subplots_adjust(hspace=0.15)  # adjust space between axes
n = [0.1,0.2,0.3,0.5,0.6,0.7,0.8,1.0]
imp_factor=[0.1,0.2,0.3,0.5,0.7,1.0]
tran_factor=[0.1,0.3,0.5,0.7,1.0]
bgnn_bivl = [16.09, 16.99,17.50,16.9,16.83,16.93,17.11,17.02]
# imp=[8.29,8.33,6.91,7.79,8.45,7.92]#7.92是union而不是fusion
imp=[8.29,8.33,7.11,7.79,8.45,8]
# transformer=[10.24,10.68,10.47,10.01,10.26]
transformer=[10.24,10.38,10.47,10.68,10.43]
# SHA_factor=[0.1,0.3,0.5,0.7,1.0]
# SHA_GCL=[18.91,19.50,18.51,18.15,18.20]
motif_factor=[0.1, 0.3,0.7,1]
motif=[9.3,9.03,10.76,10.56]
# plt.yscale('log')
ax1.plot(motif_factor, motif,marker = 10,markersize = 8,label='Motifs')
ax1.plot(n, bgnn_bivl,marker = '.',markersize = 8,label='BGNN+BLS')
ax1.plot(imp_factor, imp,marker = 'v',markersize = 8,label='IMP')
ax1.plot(tran_factor, transformer,marker = '*',markersize = 8,label='Transformer')
# plt.plot(SHA_factor, SHA_GCL,marker = 'o',markersize = 5,label='SHA+GCL')
ax2.plot(motif_factor, motif,marker = 10,markersize = 8,label='Motifs')
ax2.plot(n, bgnn_bivl,marker = '.',markersize = 8,label='BGNN+BLS')
ax2.plot(imp_factor, imp,marker = 'v',markersize = 8,label='IMP')
ax2.plot(tran_factor, transformer,marker = '*',markersize = 8,label='Transformer')
ax1.set_ylim(16, 18)  # outliers only
ax2.set_ylim(6, 11)  # most of the data
# hide the   spines between ax and ax2
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()#刻度是朝上还是朝下
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()
plt.xlabel('factor',fontsize=15)
plt.ylabel('mR@100 (%)',fontsize=15)
plt.rcParams.update({'font.size':10})
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 2.35),ncol=4)
plt.savefig("factor.pdf",dpi=500)
# plt.tight_layout()
d = .5  # proportion of vertical to horizontal extent of the slanted line 其实就是斜杠的斜率
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
plt.show()
############################################################################

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
#----------------------------------------------------
#画不同层数transformer变化

# bgnn_1_layer=[349.017,16.4]
# bgnn_2_layer=[352.16,16.7]
# bgnn_3_layer=[355.32,17.3]
# bgnn_4_layer=[358.73,18.34]
# bgnn_6_layer=[364.77,18.5]
# bgnn_size=[349,355,358,364]
# bgnn_recall=[16.7,18.0,18.34,18.45]
# bgnn_base_recall=[7.5,8.5,8.97,9.05]
# motif_size=[380,387,389,396]
# motif_recall=[9.8,10.25,10.56,10.5]
# motif_bil_recall=[]
# trans_size=[329,342,348,361]
# trans_recall = [10,10.35,10.43,10.5]
# imp_size = [347,353,356,362]
# imp_recall=[7.5,7.8,8,8.02]
# agcnn_size=[379,385,386,394]
# agcnn_recall=[6.2,6.7,7.2,7.21]
# plt.figure(figsize=(8,6),dpi=500)
# # ax1.plot(bgnn_size,bgnn_recall,linestyle= '--', marker = '.',markersize = 7,label='BGNN+BLS')
# plt.plot(bgnn_size,bgnn_base_recall,linestyle= (0, (3, 1, 1, 1)), marker = '4',markersize = 10,label='BGNN')
# plt.plot(motif_size,motif_recall,linestyle= '-', marker = '^',markersize = 10,label='Motifs')
# plt.plot(trans_size,trans_recall,linestyle= '-.', marker = 'v',markersize = 10,label='Transformer')
# plt.plot(imp_size,imp_recall,linestyle= ':', marker = 'o',markersize = 8,label='IMP')
# plt.plot(agcnn_size,agcnn_recall,linestyle= (0, (3, 10, 1, 10)), marker = '2',markersize = 10,label='G-RCNN')
# plt.xlabel('# Parameters (M)',fontsize= 15)
# plt.ylabel('mR@100 (%)',fontsize=15)
# plt.rcParams.update(({'font.size': 9}))
# plt.legend(loc='lower left')
#
# # plt.show()
# plt.savefig("trans_layer.pdf",bbox_inches='tight')

###################画factor中的断轴所用的函数#####################

class BrokenAxisPlot:
    def __init__(self, which="x",d=0.015, **kwargs):
        args = (1, 2) if which == "x" else (2, 1)
        self.is_xaxis = which == "x"
        self.fig, (self.ax1, self.ax2) = plt.subplots(*args, **kwargs)
        self._set_broken_style(d)

    def _set_broken_style(self, d=0.015):
        ax1, ax2 = self.ax1, self.ax2
        if self.is_xaxis:
            ax2.yaxis.tick_right()
            ax1_spine, ax2_spine = ("right", "left")
            ax1_diagonal = [[[1 - d, 1 + d], [-d, d]], [[1 - d, 1 + d], [1 - d, 1 + d]]]
            ax2_diagonal = [[[-d, d], [-d, d]], [[-d, d], [1 - d, 1 + d]]]

        else:
            ax1.xaxis.tick_top()
            ax1_spine, ax2_spine = ("bottom", "top")
            ax1_diagonal = [[[-d, d], [-d, d]], [[1 - d, 1 + d], [-d, d]]]
            ax2_diagonal = [
                [
                    [-d, d],
                    [1 - d, 1 + d],
                ],
                [[1 - d, 1 + d], [1 - d, 1 + d]],
            ]
        ax1.spines[ax1_spine].set_visible(False)
        ax2.spines[ax2_spine].set_visible(False)

        kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
        ax1.plot(*ax1_diagonal[0], **kwargs)
        ax1.plot(*ax1_diagonal[1], **kwargs)
        kwargs["transform"] = ax2.transAxes
        ax2.plot(*ax2_diagonal[0], **kwargs)
        ax2.plot(*ax2_diagonal[1], **kwargs)

    def set_lims(self, ax1_lim, ax2_lim):
        if self.is_xaxis:
            self.ax1.set_xlim(*ax1_lim)
            self.ax2.set_xlim(*ax2_lim)
        else:
            self.ax1.set_ylim(*ax1_lim)
            self.ax2.set_ylim(*ax2_lim)

    def plot(self, x, y,**kwargs):
        self.ax1.plot(x, y,**kwargs)
        self.ax2.plot(x, y,**kwargs)