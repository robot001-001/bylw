import pandas as pd
pd.set_option('display.max_columns',None)       #打印列数无限制
pd.set_option('display.max_rows',None)          #打印行数无限制
pd.set_option('display.float_format',lambda x : '%.3f' % x)    #禁用科学计数法，保留3位小数


import numpy as np
np.set_printoptions(formatter={'all':lambda x: str(x)},threshold=100)     #禁用科学计数法


import warnings
warnings.filterwarnings("ignore") #过滤掉警告的意思


import time
import datetime

from collections import Counter # 统计列表中各元素出现次数


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
plt.rcParams['font.sans-serif']= ['Heiti TC'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


import os

import logging
logging.getLogger('matplotlib.font_manager').disabled = True


def color_print(spell, color):
    '''
    red, yellow, brown, green, ching, blue, purple
    '''
    color_dict = {
        'red': 91,
        'yellow': 33,
        'brown': 93,
        'green': 92,
        'ching': 96,
        'blue': 94,
        'purple': 95,
    }
    return f'\033[{color_dict[color]}m' + str(spell) + '\033[0m'


class plot_base:
    '''
    功能: 
        * 传入基础参数，这些参数为所有画图函数共有
        * check_dim: 检查数据的维数是否正确
        * InitPlot: 初始化绘图区域，plot_base只允许一张图
    '''
    def __init__(self, x, y, **kwargs):
        '''
        字段解析: 
            * x轴标签：x，一个分类列表
            * y轴数值：y，一个矩阵，y.shape[0]=len(x)
        额外参数: 
            * 图例：label，一个列表，len(label)=y.shape[1]
            * 标题：title
            * x, y轴名称：xlib，ylib
            * 是否显示数值标签：data_view，默认为True
        '''
        self.x = x
        self.y = y
        self.label = kwargs.get('label', '')
        self.title = kwargs.get('title', '')
        self.xlib = kwargs.get('xlib', '')
        self.ylib = kwargs.get('ylib', '')
        self.xkcd = kwargs.get('xkcd', False)
        self.guide_x = kwargs.get('guide_x', [])
        self.guide_y = kwargs.get('guide_y', [])
        self.label_name = kwargs.get('label_name', [])
        
        
        
    def check_dim(self):
        '''
        check_dim: 检查输入的x, y, label的数量
            * x轴标签：x，一个分类列表
            * y轴数值：y，一个矩阵，y.shape[0]=len(x)
            * 图例：label，一个列表，len(label)=y.shape[1]
        '''
        err = 0
        if type(self.y) == list or self.y.ndim == 1:
            '''在x, y均为一维的时候，只检查x, y'''
            try:
                assert len(self.x) == len(self.y)
            except:
                spell = color_print('len(x)!=y.shape[0]!!', 'red')
                print(spell)
                err = 1
        elif self.label == '':
            '''在label取值为默认(即没有输入label)的时候，只检查x, y'''
            try:
                assert len(self.x) == self.y.shape[0]
            except:
                spell = color_print('len(x)!=y.shape[0]!!', 'red')
                print(spell)
                err = 1
        else:
            '''其他情况，检查x, y, label'''
            try:
                assert len(self.x) == self.y.shape[0]
            except:
                spell = color_print('len(x)!=y.shape[0]!!', 'red')
                print(spell)
                err = 1
            try:
                assert len(self.label) == self.y.shape[1]
            except:
                spell = color_print('len(label)!=y.shape[1]!!', 'red')
                print(spell)
                err = 1
        '''函数返回值用来记录是否出现问题, 出现问题返回1'''
        return err
    
    def InitPlot(self):
        figure, axs = plt.subplots(figsize = (10, 5))
        return figure, axs
    
    def plot(self):
        '''调用xkcd'''
        if self.xkcd:
            with plt.xkcd():
                self.show()
        else:
            self.show()
            

            
            
            
            
class line_plot(plot_base):
    '''
    功能：
        * 绘制折线图
        * 继承plot_base的所有参数
        * 额外参数: 
            * smooth: 是否是平滑曲线
            * from_zero: y轴的坐标是否从0开始, 默认为true
            * color: 各条曲线的颜色，一个列表，len(color) = len(label)
    '''
    def __init__(self, x, y, **kwargs):
        super().__init__(x, y, **kwargs)
        self.smooth = kwargs.get('smooth', '')
        self.from_zero = kwargs.get('from_zero', True)
        self.color = kwargs.get('color', None)
            
    def show(self):
        '''折线图'''
        # 判断维数是否正确
        if self.check_dim():
            return False
        
        # 画图区域格式
        figure, ax = self.InitPlot()
        
        # 判断y轴是否从0开始
        if self.from_zero:
            ax.set_ylim(0, np.amax(self.y)*1.1)     # y轴从0开始，默认最大值为y矩阵中的最大值的1.1倍

        # 禁用科学计数法
        ax.ticklabel_format(axis="y", style='plain')
        
        # 绘图，传入x，y，label
        # 曲线平滑处理
        if self.smooth:
            temp_x = len(self.x)
            data_size = temp_x*10
            from scipy.interpolate import make_interp_spline
            for i in range(self.y.shape[1]):
                temp_y = self.y[:, i].tolist()
                xss = np.linspace(0, temp_x-1, temp_x)
                m = make_interp_spline(xss, temp_y)
                xs = np.linspace(0, temp_x-1, data_size)
                ys = m(xs)
                if self.color:  # 指定颜色
                    ax.plot(xs, ys, label = self.label[i], color = self.color[i])
                else:
                    ax.plot(xs, ys, label = self.label[i])
                ticks = ax.set_xticks(xss) # 设置刻度
                labels = ax.set_xticklabels(
                    self.x, fontsize = 'small'
                ) # 设置刻度标签
        else:
            if self.color:  # 指定颜色
                for i in range(self.y.shape[1]):
                    temp_y = self.y[:, i].tolist()
                    ax.plot(self.x, temp_y, label = self.label[i], color = self.color[i])
            else:
                ax.plot(self.x, self.y, label = self.label)

        # 判断是否需要垂直于x轴辅助线，如果有则加上
        if len(self.guide_x) != 0 and self.smooth:
            guide_x_sub = []
            for i in range(len(self.x)):
                if(self.x[i] in guide_x):
                    guide_x_sub.append(i)
            for i in guide_x_sub:
                ax.axvline(x=i,ls=":",c="black")
        elif len(self.guide_x) != 0:
            for i in self.guide_x:
                ax.axvline(x=i,ls=":",c="black")
        
        # 判断是否需要垂直于y轴辅助线，如果有则加上
        if len(self.guide_y) != 0:
            for i in self.guide_y:
                ax.axhline(y=i,ls=":",c="black")

        # 设置坐标轴名称、图标名称
        try:
            ax.set_xlabel(self.xlib)
            ax.set_ylabel(self.ylib)
            ax.set_title(self.title)
        except:
            pass

        # 美观用，如果x轴是非数字就竖着写xlib
        if type(self.x[0]) not in (int, float, np.int64):
            ax.tick_params(axis = "x", rotation = 90)

        # 增加图例
        if self.label != '':
            plt.legend(title = self.label_name)

        #绘图
        plt.show()

        return True
    
    
    
# 双y轴
def plot_2y(x, y1, y2, ylib1, ylib2, xlib, guide_x = [], max_y = False):
    '''
    双y轴通常用作两组数据对比，因此传进来的参数相对固定
        x轴：数据：列表/数组x
            坐标轴名称：通过xlib传入
        y1轴：数据：矩阵y1（可一元可多元，多元效果不好）
            坐标轴名称：通过ylib1传入
        y2轴：数据：矩阵y2（可一元可多元，多元效果不好）
            坐标轴名称：通过ylib2传入
        图表标题：字符串title
        垂直于x轴的辅助线：通过列表guide_x传入，其中每个元素为辅助线对应的横坐标
        max_y：适用于一元情况，设定两组数据对y轴的映射，不传入数据时，默认两组数据的最大值映射区间相同
               传入int数据时，表示x的第"max_y-1"个取值映射区间相同，此时，会用红色虚线标出横坐标
    '''
    # 画图区域格式
    fig, ax = plt.subplots(figsize = (10, 5))

    # 创建与轴群ax共享x轴的轴群z_ax
    z_ax = ax.twinx() 
    
    # 双y轴图中，默认最小值为0， 最大值为双方的最大值的1.1倍，映射到统一区间

    if type(max_y) == int:
        ax.set_ylim(0, y1[max_y]*1.5)
        z_ax.set_ylim(0, y2[max_y]*1.5)
        guide_x.append(x[max_y])
    elif max_y:
        print('code yourself!')
        return 0
    else:
        ax.set_ylim(0, np.amax(y1)*1.1)
        z_ax.set_ylim(0, np.amax(y2)*1.1)

    # 禁用科学计数法
    ax.ticklabel_format(axis="y", style='plain')
    z_ax.ticklabel_format(axis="y", style='plain')
    
    # ax右轴隐藏
    ax.spines['right'].set_visible(False) 
    
    # 画图
    line1 = ax.plot(x, y1, color='green', label = ylib1)
    line2 = z_ax.plot(x, y2, color='blue', label = ylib2)
    
    # 判断是否需要垂直于y轴辅助线，如果有则加上
    if len(guide_x) != 0:
        if type(max_y) == int:
            for i in range(0, len(guide_x)-1, 1):
                ax.axvline(x=guide_x[i],ls=":",c="black")
            ax.axvline(x=guide_x[len(guide_x)-1],ls=":",c="red")
        else:
            for i in guide_x:
                ax.axvline(x=i,ls=":",c="black")
    
    # 设置坐标轴和标题名称
    ax.set_ylabel(ylib1)
    ax.set_xlabel(xlib)
    z_ax.set_ylabel(ylib2)
    plt.title(ylib1 + '&' + ylib2)
    
    # 美观用，如果x轴是非数字就竖着写xlib
    if type(x[0]) not in (int, float, np.int64):
        ax.tick_params(axis = "x", rotation = 90)

    # 图例的位置，两张图分开
    handles, labels = [], []
    for ax in fig.axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)
    plt.legend(handles, labels)
    


    plt.show()
    
    
    
    
class stack_plot(plot_base):
    '''
    功能：
        * 绘制堆叠面积图
        * 继承plot_base的所有参数
        * 额外参数: 
            * Percentage: 是否是百分比堆叠面积图，默认为false
            * alpha: 透明度
            * LegendShow: 是否显示图例
    '''
    def __init__(self, x, y, **kwargs):
        super().__init__(x, y, **kwargs)
        self.Percentage = kwargs.get('Percentage', False)
        self.alpha = kwargs.get('alpha', 0.5)
        self.LegendShow = kwargs.get('LegendShow', True)
            
    def show(self):
        '''堆叠面积图'''
        # 判断维数是否正确
        if self.check_dim():
            return False
        
        # 画图区域格式
        figure, ax = self.InitPlot()
        
        # 禁用科学计数法
        ax.ticklabel_format(axis="y", style='plain')
        
        # 是否使用百分比堆叠面积图
        if self.Percentage:
            self.y = self.y.T
            self.y = self.y / self.y.sum(axis = 0)
            self.y = self.y.T

        # 绘图，传入x，y，label
        temp_y = 0
        # ls存放需添加图例的线
        ls = []
        for i in range(self.y.shape[1]):
            spell = f'line_{i}, = ax.plot(self.x, temp_y + self.y[:, i])'
            exec(spell)
            spell = f'ls.append(line_{i})'
            exec(spell)
            ax.fill_between(self.x, temp_y, temp_y + self.y[:, i], alpha = self.alpha)
            temp_y += self.y[:, i]

        # 判断是否需要垂直于x轴辅助线，如果有则加上
        if len(self.guide_x) != 0:
            for i in self.guide_x:
                ax.axvline(x=i,ls=":",c="black")

        # 判断是否需要垂直于y轴辅助线，如果有则加上
        if len(self.guide_y) != 0:
            for i in self.guide_y:
                ax.axhline(y=i,ls=":",c="black")

        # 设置坐标轴名称、图标名称
        ax.set_xlabel(self.xlib)
        ax.set_ylabel(self.ylib)
        ax.set_title(self.title)

        # 美观用，如果x轴是非数字就竖着写xlib
        if type(self.x[0]) not in (int, float, np.int64):
            ax.tick_params(axis = "x", rotation = 90)

        # 增加图例
        if self.LegendShow:
            ls.reverse()
            temp_labels = self.label.copy()
            temp_labels.reverse()
            plt.legend(handles = ls, labels = temp_labels)

        #绘图
        plt.show()

        return True