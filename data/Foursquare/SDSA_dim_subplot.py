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
    print(matplotlib.matplotlib_fname())
    # NYC HR
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    names = [8,16,32,64,128,256]
    plt.grid(ls='--')
    plt.rc('font', family='Times New Roman')
    fig, ax = plt.subplots(2,3)
    x = names
    # plt.subplot(2,2,1)
    NY_hr1 = [0.5088,0.5319,0.5189,0.5199,0.5171,0.5124]
    NY_hr3 = [0.6500,0.6731,0.6759,0.6750,0.6685,0.6629]
    NY_hr5 = [0.7110,0.7267,0.7175,0.7175,0.7101,0.7156]
    NY_hr10 = [0.7775,0.7959,0.7895,0.7886,0.7886,0.7756]
    # plt.plot(x, y, 'ro-')
    # plt.plot(x, y1, 'bo-')
    # pl.xlim(-1, 11)  # 限定横轴的范围
    ax[0][0].set_ylim(0.45, 0.85)  # 限定纵轴的范围
    ax[0][0].plot(x, NY_hr1, marker='o', markersize=8,label=u'HR@1')
    ax[0][0].plot(x, NY_hr3, marker='p', markersize=8, label=u'HR@3')
    ax[0][0].plot(x, NY_hr5, marker='s', markersize=8, label=u'HR@5')
    ax[0][0].plot(x, NY_hr10, marker='v', markersize=8, label=u'HR@10')
    ax[0][0].grid(ls='--')
    ax[0][0].set_xscale('log')
    ax[0][0].set_xticks(names)
    ax[0][0].set_xticklabels(names,fontsize=14)
    ax[0][0].yaxis.set_tick_params(labelsize=14)
    ax[0][0].margins(0.03)
    ax[0][0].set_xlabel(u"latent vector dimension",fontsize=fon) #X轴标签
    ax[0][0].set_ylabel(u"HR",fontsize=fon) #X轴标签
    # plt.ylabel("HR") #Y轴标签
    ax[0][0].set_title("Foursquare-NY",fontsize=fon) #标题
    # NYC NDCG
    # plt.subplot(2,2,2)
    NY_ndcg1 = [0.5088,0.5319,0.5189,0.5199,0.5171,0.5124]
    NY_ndcg3 = [0.5921,0.6160,0.6118,0.6111,0.6062,0.6040]
    NY_ndcg5 = [0.6173,0.6382,0.6252,0.6286,0.6234,0.6257]
    NY_ndcg10 = [0.6387,0.6586,0.6499,0.6517,0.6490,0.6455]
    ax[1][0].set_ylim(0.45, 0.70)  # 限定纵轴的范围
    ax[1][0].plot(x, NY_ndcg1, marker='o', markersize=8, label=u'NDCG@1')
    ax[1][0].plot(x, NY_ndcg3, marker='p', markersize=8, label=u'NDCG@3')
    ax[1][0].plot(x, NY_ndcg5, marker='s',  markersize=8, label=u'NDCG@5')
    ax[1][0].plot(x, NY_ndcg10, marker='v',  markersize=8, label=u'NDCG@10')
    ax[1][0].grid(ls='--')
    ax[1][0].set_xscale('log')
    ax[1][0].set_xticks(names)
    ax[1][0].set_xticklabels(names,fontsize=14)
    ax[1][0].yaxis.set_tick_params(labelsize=14)
    ax[1][0].margins(0.03)
    ax[1][0].set_xlabel(u"latent vector dimension",fontsize=fon)  # X轴标签
    ax[1][0].set_ylabel("NDCG",fontsize=fon) #Y轴标签
    ax[1][0].set_title("Foursquare-NY",fontsize=fon)  # 标题
    # TKY HR
    # plt.subplot(2,2,3)
    x = names
    TKY_hr1 = [0.4902,0.4954,0.5107,0.5081,0.4993,0.4941]
    TKY_hr3 = [0.7082,0.7039,0.7143,0.7170,0.7117,0.7113]
    TKY_hr5 = [0.7802,0.7828,0.7924,0.7933,0.7863,0.7867]
    TKY_hr10 = [0.8574,0.8578,0.8666,0.8556,0.8539,0.8565]
    # plt.plot(x, y, 'ro-')
    # plt.plot(x, y1, 'bo-')
    # pl.xlim(-1, 11)  # 限定横轴的范围
    ax[0][1].set_ylim(0.45, 0.9)  # 限定纵轴的范围
    ax[0][1].plot(x, TKY_hr1, marker='o',  markersize=8,label=u'HR@1')
    ax[0][1].plot(x, TKY_hr3, marker='p',  markersize=8, label=u'HR@3')
    ax[0][1].plot(x, TKY_hr5, marker='s',  markersize=8, label=u'HR@5')
    ax[0][1].plot(x, TKY_hr10, marker='v',  markersize=8, label=u'HR@10')
    ax[0][1].grid(ls='--')
    ax[0][1].set_xscale('log')
    ax[0][1].set_xticks(names)
    ax[0][1].set_xticklabels(names,fontsize=14)
    ax[0][1].yaxis.set_tick_params(labelsize=14)
    ax[0][1].margins(0.03)
    ax[0][1].set_xlabel(u"latent vector dimension",fontsize=fon) #X轴标签
    ax[0][1].set_ylabel("HR",fontsize=fon) #Y轴标签
    ax[0][1].set_title("Foursquare-TKY",fontsize=fon) #标题
    # TKY NDCG
    # plt.subplot(2,2,4)
    TKY_ndcg1 = [0.4902,0.4954,0.5107,0.5081,0.4993,0.4941]
    TKY_ndcg3 = [0.6196,0.6178,0.6307,0.6311,0.6191,0.6213]
    TKY_ndcg5 = [0.6492,0.6504,0.6629,0.6627,0.6501,0.6526]
    TKY_ndcg10 = [0.6744,0.6750,0.6871,0.6830,0.6754,0.6751]
    ax[1][1].set_ylim(0.45, 0.7)  # 限定纵轴的范围
    ax[1][1].plot(x, TKY_ndcg1, marker='o',  markersize=8, label=u'TOP@1')
    ax[1][1].plot(x, TKY_ndcg3, marker='p',  markersize=8, label=u'TOP@3')
    ax[1][1].plot(x, TKY_ndcg5, marker='s',  markersize=8, label=u'TOP@5')
    ax[1][1].plot(x, TKY_ndcg10, marker='v',  markersize=8, label=u'TOP@10')
    ax[1][1].grid(ls='--')
    ax[1][1].set_xscale('log')
    ax[1][1].set_xticks(names)
    ax[1][1].set_xticklabels(names,fontsize=14)
    ax[1][1].yaxis.set_tick_params(labelsize=14)
    ax[1][1].margins(0.03)
    ax[1][1].set_xlabel(u"latent vector dimension",fontsize=fon)  # X轴标签
    ax[1][1].set_ylabel("NDCG",fontsize=fon) #Y轴标签
    ax[1][1].set_title("Foursquare-TKY",fontsize=fon)  # 标题
    # Gowalla HR
    # plt.subplot(2,2,3)
    x = names
    Gowalla_hr1 = [0.3802, 0.3916, 0.3957, 0.4051, 0.4216, 0.4121]
    Gowalla_hr3 = [0.6202, 0.6359, 0.6403, 0.6570, 0.6709, 0.6559]
    Gowalla_hr5 = [0.7032, 0.7289, 0.7354, 0.7533, 0.7639, 0.7595]
    Gowalla_hr10 = [0.8274, 0.8437, 0.8531, 0.8656, 0.8937, 0.8811]
    # plt.plot(x, y, 'ro-')
    # plt.plot(x, y1, 'bo-')
    # pl.xlim(-1, 11)  # 限定横轴的范围
    ax[0][2].set_ylim(0.35, 0.93)  # 限定纵轴的范围
    ax[0][2].plot(x, Gowalla_hr1, marker='o', markersize=8, label=u'HR@1')
    ax[0][2].plot(x, Gowalla_hr3, marker='p',  markersize=8, label=u'HR@3')
    ax[0][2].plot(x, Gowalla_hr5, marker='s',  markersize=8, label=u'HR@5')
    ax[0][2].plot(x, Gowalla_hr10, marker='v',  markersize=8, label=u'HR@10')
    ax[0][2].grid(ls='--')
    ax[0][2].set_xscale('log')
    ax[0][2].set_xticks(names)
    ax[0][2].set_xticklabels(names,fontsize=14)
    ax[0][2].yaxis.set_tick_params(labelsize=14)
    ax[0][2].margins(0.03)
    ax[0][2].set_xlabel(u"latent vector dimension", fontsize=fon)  # X轴标签
    ax[0][2].set_ylabel("HR", fontsize=fon)  # Y轴标签
    ax[0][2].set_title("Gowalla", fontsize=fon)  # 标题
    # Gowalla NDCG
    # plt.subplot(2,2,4)
    Gowalla_ndcg1 = [0.3802, 0.3916, 0.3957, 0.4051, 0.4216, 0.4121]
    Gowalla_ndcg3 = [0.5346, 0.5496, 0.5557, 0.5611, 0.5736, 0.5673]
    Gowalla_ndcg5 = [0.5672, 0.5769, 0.5891, 0.6052, 0.6167, 0.6056]
    Gowalla_ndcg10 = [0.6120, 0.6226, 0.6368, 0.6493, 0.6536, 0.6451]
    ax[1][2].set_ylim(0.35, 0.7)  # 限定纵轴的范围
    l1 = ax[1][2].plot(x, Gowalla_ndcg1, marker='o',  markersize=8, label=u'TOP@1')
    l2 = ax[1][2].plot(x, Gowalla_ndcg3, marker='p',  markersize=8, label=u'TOP@3')
    l3 = ax[1][2].plot(x, Gowalla_ndcg5, marker='s',  markersize=8, label=u'TOP@5')
    l4 = ax[1][2].plot(x, Gowalla_ndcg10, marker='v',  markersize=8, label=u'TOP@10')
    ax[1][2].grid(ls='--')
    ax[1][2].set_xscale('log')
    ax[1][2].set_xticks(names)
    ax[1][2].set_xticklabels(names,fontsize=14)
    ax[1][2].yaxis.set_tick_params(labelsize=14)
    ax[1][2].margins(0.03)
    ax[1][2].set_xlabel(u"latent vector dimension", fontsize=fon)  # X轴标签
    ax[1][2].set_ylabel("NDCG", fontsize=fon)  # Y轴标签
    ax[1][2].set_title("Gowalla", fontsize=fon)  # 标题

    # 图表标题
    fig.subplots_adjust(bottom=0.3, wspace=0.33)
    labels = ["TOP@1", "TOP@3","TOP@5","TOP@10"]
    fig.legend([l1, l2, l3, l4], bbox_to_anchor=(0.5, 0.99), labels=labels,loc="upper center", ncol=4
                    ,fancybox=False, shadow=False,borderaxespad=0.1,fontsize=fon)
    plt.tight_layout()
    # plt.savefig('/home/shenyi/picture/total/total_dim.svg')
    plt.show()