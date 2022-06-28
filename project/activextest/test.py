# -*- coding: utf-8 -*-


# 导入需要的库
import os
import win32com.client as win32
import time
import numpy as np

# 创建AP的本地服务器

# AP8.8版替换下面的36.0为34.0; 9.0替换为35.0; 10.0替换为36.0； 11.0替换为37.0
Application = win32.Dispatch('MaterialsStudio.Document')

# 获取当前文件夹的地址，三效蒸发器文件和本程序文件需放置在同一个文件夹
address = os.getcwd()

# AP的bkp文件的文件名，即三效蒸发器bkp文件的文件名
SimulationName = 'Triple-Effect Evaporator'

# 打开三效蒸发器文件
Application.InitFromArchive2(os.path.abspath(SimulationName + '.bkp'))

# 设置AP用户界面的可见性，1为可见，0为不可见
Application.Visible = 1

# 压制对话框的弹出，1为压制；0为不压制
Application.SuppressDialogs = 1


