import matplotlib.pyplot as plt
from matplotlib import pylab as pl
from pylab import *
import numpy as np
import os
from matplotlib.font_manager import _rebuild


if __name__ == '__main__':
    fon = 18
    # if not os.path.exists('/home/shenyi/picture/total'):
    #     os.makedirs('/home/shenyi/picture/total')
    # NYC HR
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    plt.grid(ls='--')
    plt.rc('font', family='Times New Roman')
    fig, ax = plt.subplots(2,3)
    names = [i for i in range(1, 7)]
    x = names
    NY_hr1 = [0.5189, 0.5189, 0.5125, 0.5199, 0.5162, 0.5005]
    NY_hr3 = [0.6667, 0.6759, 0.6750, 0.6676, 0.6667, 0.6787]
    NY_hr5 = [0.7175, 0.7211, 0.7239, 0.7101, 0.7119, 0.7147]
    NY_hr10 = [0.7886, 0.7959, 0.7849, 0.7738, 0.7756, 0.7793]
    # plt.plot(x, y, 'ro-')
    # plt.plot(x, y1, 'bo-')
    # pl.xlim(-1, 11)  # 限定横轴的范围
    ax[0][0].set_ylim(0.45, 0.85)  # 限定纵轴的范围
    ax[0][0].plot(x, NY_hr1, marker='o', markersize=8,label=u'HR@1')
    ax[0][0].plot(x, NY_hr3, marker='p',  markersize=8, label=u'HR@3')
    ax[0][0].plot(x, NY_hr5, marker='s',  markersize=8, label=u'HR@5')
    ax[0][0].plot(x, NY_hr10, marker='v',  markersize=8, label=u'HR@10')
    ax[0][0].grid(ls='--')
    ax[0][0].set_xticks(names)
    ax[0][0].set_xticklabels(names,fontsize=14)
    ax[0][0].yaxis.set_tick_params(labelsize=14)
    ax[0][0].margins(0.03)
    ax[0][0].set_xlabel(u"sequence length",fontsize=fon) #X轴标签
    ax[0][0].set_ylabel("HR", fontsize=fon)  # Y轴标签
    ax[0][0].set_title("Foursquare-NY",fontsize=fon) #标题
    # NYC NDCG
    NY_ndcg1 = [0.5189, 0.5189, 0.5125, 0.5199, 0.5162, 0.5005]
    NY_ndcg3 = [0.6062, 0.6118, 0.6083, 0.6068, 0.6042, 0.6056]
    NY_ndcg5 = [0.6284, 0.6292, 0.6283, 0.6245, 0.6226, 0.6203]
    NY_ndcg10 = [0.6502, 0.6545, 0.6479, 0.6450, 0.6431, 0.6411]
    ax[1][0].set_ylim(0.45, 0.70)  # 限定纵轴的范围
    ax[1][0].plot(x, NY_ndcg1, marker='o', markersize=8, label=u'NDCG@1')
    ax[1][0].plot(x, NY_ndcg3, marker='p',  markersize=8, label=u'NDCG@3')
    ax[1][0].plot(x, NY_ndcg5, marker='s',  markersize=8, label=u'NDCG@5')
    ax[1][0].plot(x, NY_ndcg10, marker='v',  markersize=8, label=u'NDCG@10')
    ax[1][0].grid(ls='--')
    ax[1][0].set_xticks(names)
    ax[1][0].set_xticklabels(names,fontsize=14)
    ax[1][0].yaxis.set_tick_params(labelsize=14)
    ax[1][0].margins(0.03)
    ax[1][0].set_xlabel(u"sequence length",fontsize=fon)  # X轴标签
    ax[1][0].set_ylabel("NDCG",fontsize=fon) #Y轴标签
    ax[1][0].set_title("Foursquare-NY",fontsize=fon)  # 标题
    # TKY HR
    x = names
    TKY_hr1 = [0.4915, 0.5185, 0.5107, 0.5028, 0.5076, 0.4963]
    TKY_hr3 = [0.7100, 0.7253, 0.7143, 0.7091, 0.7157, 0.7122]
    TKY_hr5 = [0.7780, 0.7915, 0.7924, 0.7828, 0.7885, 0.7902]
    TKY_hr10 = [0.8565, 0.8709, 0.8666, 0.8639, 0.8596, 0.8622]
    # plt.plot(x, y, 'ro-')
    # plt.plot(x, y1, 'bo-')
    # pl.xlim(-1, 11)  # 限定横轴的范围
    ax[0][1].set_ylim(0.45, 0.9)  # 限定纵轴的范围
    ax[0][1].plot(x, TKY_hr1, marker='o',  markersize=8,label=u'HR@1')
    ax[0][1].plot(x, TKY_hr3, marker='p',  markersize=8, label=u'HR@3')
    ax[0][1].plot(x, TKY_hr5, marker='s',  markersize=8, label=u'HR@5')
    ax[0][1].plot(x, TKY_hr10, marker='v',  markersize=8, label=u'HR@10')
    ax[0][1].grid(ls='--')
    ax[0][1].set_xticks(names)
    ax[0][1].set_xticklabels(names,fontsize=14)
    ax[0][1].yaxis.set_tick_params(labelsize=14)
    ax[0][1].margins(0.03)
    ax[0][1].set_xlabel(u"sequence length",fontsize=fon) #X轴标签
    ax[0][1].set_ylabel("HR",fontsize=fon) #Y轴标签
    ax[0][1].set_title("Foursquare-TKY",fontsize=fon) #标题
    # TKY NDCG
    TKY_ndcg1 = [0.4915, 0.5185, 0.5107, 0.5028, 0.5076, 0.4963]
    TKY_ndcg3 = [0.6326, 0.6401, 0.6307, 0.6232, 0.6305, 0.6236]
    TKY_ndcg5 = [0.6607, 0.6674, 0.6629, 0.6537, 0.6606, 0.6558]
    TKY_ndcg10 = [0.6862, 0.6929, 0.6871, 0.6801, 0.6836, 0.6792]
    ax[1][1].set_ylim(0.45, 0.75)  # 限定纵轴的范围
    l1 = ax[1][1].plot(x, TKY_ndcg1, marker='o',  markersize=8, label=u'TOP@1')
    l2 = ax[1][1].plot(x, TKY_ndcg3, marker='p',  markersize=8, label=u'TOP@3')
    l3 = ax[1][1].plot(x, TKY_ndcg5, marker='s',  markersize=8, label=u'TOP@5')
    l4 = ax[1][1].plot(x, TKY_ndcg10, marker='v',  markersize=8, label=u'TOP@10')
    ax[1][1].grid(ls='--')
    ax[1][1].set_xticks(names)
    ax[1][1].set_xticklabels(names,fontsize=14)
    ax[1][1].yaxis.set_tick_params(labelsize=14)
    ax[1][1].margins(0.03)
    ax[1][1].set_xlabel(u"sequence length",fontsize=fon)  # X轴标签
    ax[1][1].set_ylabel("NDCG",fontsize=fon) #Y轴标签
    ax[1][1].set_title("Foursquare-TKY",fontsize=fon)  # 标题
    # Gowalla HR
    # plt.subplot(2,2,3)
    x = names
    Gowalla_hr1 = [0.3802, 0.4216, 0.4157, 0.4151, 0.4156, 0.4121]
    Gowalla_hr3 = [0.6202, 0.6709, 0.6603, 0.6670, 0.6609, 0.6619]
    Gowalla_hr5 = [0.7032, 0.7639, 0.7614, 0.7599, 0.7602, 0.7612]
    Gowalla_hr10 = [0.8374, 0.8937, 0.8911, 0.8893, 0.8885, 0.8928]
    # plt.plot(x, y, 'ro-')
    # plt.plot(x, y1, 'bo-')
    # pl.xlim(-1, 11)  # 限定横轴的范围
    ax[0][2].set_ylim(0.35, 0.95)  # 限定纵轴的范围
    ax[0][2].plot(x, Gowalla_hr1, marker='o', markersize=8, label=u'HR@1')
    ax[0][2].plot(x, Gowalla_hr3, marker='p',  markersize=8, label=u'HR@3')
    ax[0][2].plot(x, Gowalla_hr5, marker='s',  markersize=8, label=u'HR@5')
    ax[0][2].plot(x, Gowalla_hr10, marker='v',  markersize=8, label=u'HR@10')
    ax[0][2].grid(ls='--')
    ax[0][2].set_xticks(names)
    ax[0][2].set_xticklabels(names,fontsize=14)
    ax[0][2].yaxis.set_tick_params(labelsize=14)
    ax[0][2].margins(0.03)
    ax[0][2].set_xlabel(u"sequence length", fontsize=fon)  # X轴标签
    ax[0][2].set_ylabel("HR", fontsize=fon)  # Y轴标签
    ax[0][2].set_title("Gowalla", fontsize=fon)  # 标题
    # Gowalla NDCG
    # plt.subplot(2,2,4)
    Gowalla_ndcg1 = [0.3802, 0.4216, 0.4157, 0.4151, 0.4156, 0.4121]
    Gowalla_ndcg3 = [0.5346, 0.5736, 0.5677, 0.5730, 0.5701, 0.5703]
    Gowalla_ndcg5 = [0.5672, 0.6167, 0.6091, 0.6122, 0.6067, 0.6056]
    Gowalla_ndcg10 = [0.6120, 0.6536, 0.6468, 0.6513, 0.6501, 0.6481]
    ax[1][2].set_ylim(0.35, 0.7)  # 限定纵轴的范围
    ax[1][2].plot(x, Gowalla_ndcg1, marker='o',  markersize=8, label=u'TOP@1')
    ax[1][2].plot(x, Gowalla_ndcg3, marker='p',  markersize=8, label=u'TOP@3')
    ax[1][2].plot(x, Gowalla_ndcg5, marker='s',  markersize=8, label=u'TOP@5')
    ax[1][2].plot(x, Gowalla_ndcg10, marker='v',  markersize=8, label=u'TOP@10')
    ax[1][2].grid(ls='--')
    ax[1][2].set_xticks(names)
    ax[1][2].set_xticklabels(names,fontsize=14)
    ax[1][2].yaxis.set_tick_params(labelsize=14)
    ax[1][2].margins(0.03)
    ax[1][2].set_xlabel(u"sequence length", fontsize=fon)  # X轴标签
    ax[1][2].set_ylabel("NDCG", fontsize=fon)  # Y轴标签
    ax[1][2].set_title("Gowalla", fontsize=fon)  # 标题
    # label
    fig.subplots_adjust(bottom=0.3, wspace=0.33)
    labels = ["TOP@1", "TOP@3","TOP@5","TOP@10"]
    fig.legend([l1, l2, l3, l4], bbox_to_anchor=(0.5, 0.99), labels=labels,loc="upper center", ncol=4
                    ,fancybox=False, shadow=False,borderaxespad=0.1,fontsize=fon)
    plt.tight_layout()
    # plt.savefig('/home/shenyi/picture/total/total_length.svg')
    plt.show()