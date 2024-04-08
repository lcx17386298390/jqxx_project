import tkinter as tk
from tkinter import ttk,filedialog,messagebox,PhotoImage
from tkinter.ttk import *
import logging
import 图像提取
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import os
import time
from PIL import Image, ImageTk
from model_train import Model_train
import re

threadPool = ThreadPoolExecutor(max_workers=5)


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("车牌识别系统")
        self.geometry("1200x800")
        self.resizable(False, False)
        # 训练状体： 设置：可训练； 未设置：不可训练
        self.train_status = threading.Event()
        self.train_status.set()
        # 设置中断训练标志
        self.train_is_stop = False
        # 模型加载路径（写在设置页里）-》初始化
        self.load_model_path_pre = None     # 预训练模型不需要加载，直接训练，又不是学习车牌数据集
        self.load_model_path_after = 'D:\My_Code_Project\三下机器学习课设\models\\train_1\car_number\model.h5'
        # 设置模型加载和保存路径-》写在设置页里，这里只是初始化
        self.load_or_save_path = './models'
        # 设置模型结构类别 -》写在设置页里，这里只是初始化
        self.selectd_model_type = 'CNN'
        self.on_pre_train = False
        self.log_frame = None

 
        # 功能子页面列表
        self.frames = {}
        for F in (SetPage, TrainPage, TestPage, YuCePage):
            frame = F(self)
            frame.grid(row=1, column=0, sticky="nsew")
            self.frames[F] = frame

        # 导航按钮
        self.nav_button = NavButton(self)
        self.nav_button.frame.grid(row=0, column=0, sticky="nw")  # 注意这里的 "nw"
        self.nav_button.show_frame(TrainPage)
        # 关闭窗口事件（关掉训练线程）
        self.protocol("WM_DELETE_WINDOW", lambda: (threading.Event().set, self.train_status.set(), self.destroy())) # 关闭窗口时，关闭线程

# 设置页面
class SetPage(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master_App = master
        self.configure(width=1170, height=400)
        self.grid(sticky='nsew') 
        master.grid_rowconfigure(1, weight=1)
        master.grid_columnconfigure(0, weight=1) 
        self.grid_propagate(1)
        # 设置是否加载模型标记
        self.is_load_model = False
        # 设置模型加载路径
        self.load_model_path = None
        self.var1 = 1

        # 添加配置
        self.model_select_label = self.__model_select_label(self)
        self.model_select_box = self.__model_select_box(self)
        self.epoch_input_label = self.__epoch_input_label(self)
        self.model_load_label = self.__model_load_label(self)
        self.model_load_box = self.__model_load_box(self)
        self.epoch_input_entry = self.__epoch_input_entry(self)
        self.batch_size_input_label = self.__batch_size_input_label(self)
        self.batch_size_input_entry = self.__batch_size_input_entry(self)
        self.test_size_input_label = self.__test_size_input_label(self)
        self.test_size_input_entry = self.__test_size_input_entry(self)
        self.random_state_input_label = self.__random_state_input_label(self)
        self.random_state_input_entry = self.__random_state_input_entry(self)
        self.model_load_or_save_folder_label = self.__model_load_or_save_folder_label(self)
        self.model_load_or_save_folder_entry = self.__model_load_or_save_folder_entry(self)
        self.on_pre_train_label = self.__on_pre_train_label(self)
        self.on_pre_train_button = self.__on_pre_train_button(self)
        self.save_sets_button = self.__save_sets_button(self)

    def scrollbar_autohide(self,vbar, hbar, widget):
        """自动隐藏滚动条"""
        def show():
            if vbar: vbar.lift(widget)
            if hbar: hbar.lift(widget)
        def hide():
            if vbar: vbar.lower(widget)
            if hbar: hbar.lower(widget)
        hide()
        widget.bind("<Enter>", lambda e: show())
        if vbar: vbar.bind("<Enter>", lambda e: show())
        if vbar: vbar.bind("<Leave>", lambda e: hide())
        if hbar: hbar.bind("<Enter>", lambda e: show())
        if hbar: hbar.bind("<Leave>", lambda e: hide())
        widget.bind("<Leave>", lambda e: hide())
    def v_scrollbar(self,vbar, widget, x, y, w, h, pw, ph):
        widget.configure(yscrollcommand=vbar.set)
        vbar.config(command=widget.yview)
        vbar.place(relx=(w + x) / pw, rely=y / ph, relheight=h / ph, anchor='ne')
    def h_scrollbar(self,hbar, widget, x, y, w, h, pw, ph):
        widget.configure(xscrollcommand=hbar.set)
        hbar.config(command=widget.xview)
        hbar.place(relx=x / pw, rely=(y + h) / ph, relwidth=w / pw, anchor='sw')
    def create_bar(self,master, widget,is_vbar,is_hbar, x, y, w, h, pw, ph):
        vbar, hbar = None, None
        if is_vbar:
            vbar = Scrollbar(master)
            self.v_scrollbar(vbar, widget, x, y, w, h, pw, ph)
        if is_hbar:
            hbar = Scrollbar(master, orient="horizontal")
            self.h_scrollbar(hbar, widget, x, y, w, h, pw, ph)
        self.scrollbar_autohide(vbar, hbar, widget)
    def __model_select_label(self,parent):
        label = Label(parent,text="训练模型结构选择",anchor="center", )
        label.place(x=190, y=90, width=90, height=30)
        return label
    def __model_select_box(self,parent):    # 模型结构选择框
        cb = Combobox(parent, state="readonly", )
        cb['values'] = ("CNN（浅层应用）","LeNet-5","AlexNet","VGGNet-16","ResNet-50","InceptionV3")
        cb.bind("<<ComboboxSelected>>", lambda event: self.model_select_box_select(event))
        cb.current(0)
        cb.place(x=322, y=87, width=150, height=30)
        return cb
    def __epoch_input_label(self,parent):
        label = Label(parent,text="epoch",anchor="center", )
        label.place(x=190, y=180, width=90, height=30)
        return label
    def __model_load_label(self,parent):
        label = Label(parent,text="训练模型加载",anchor="center", )
        label.place(x=720, y=90, width=90, height=30)
        return label
    def __model_load_box(self,parent):
        cb = Combobox(parent, state="readonly", )
        cb['values'] = ("训练新的模型")
        cb.current(0)
        cb.place(x=850, y=90, width=150, height=30)
        cb.bind("<<ComboboxSelected>>", lambda event: self.model_load_box_select(event))
        return cb
    def __epoch_input_entry(self,parent):
        ipt = Entry(parent, )
        ipt.place(x=320, y=180, width=150, height=30)
        ipt.insert(0, '10')
        ipt.bind("<FocusOut>",lambda event:  self.is_valid_char(ipt, 'int', '10'))
        return ipt
    def __batch_size_input_label(self,parent):
        label = Label(parent,text="batch_size",anchor="center", )
        label.place(x=190, y=270, width=90, height=30)
        return label
    def __batch_size_input_entry(self,parent):
        ipt = Entry(parent, )
        ipt.place(x=320, y=270, width=150, height=30)
        ipt.insert(0, '128')
        ipt.bind("<FocusOut>",lambda event:  self.is_valid_char(ipt, 'int', '128'))
        return ipt
    def __test_size_input_label(self,parent):
        label = Label(parent,text="训练集大小",anchor="center", )
        label.place(x=720, y=180, width=90, height=30)
        return label
    def __test_size_input_entry(self,parent):
        ipt = Entry(parent, )
        ipt.place(x=850, y=180, width=150, height=30)
        ipt.insert(0, '0.2')
        ipt.bind("<FocusOut>",lambda event:  self.is_valid_char(ipt, 'float', '0.2'))
        return ipt
    def __random_state_input_label(self,parent):
        label = Label(parent,text="random_state",anchor="center", )
        label.place(x=720, y=270, width=90, height=30)
        return label
    def __random_state_input_entry(self,parent):
        ipt = Entry(parent, )
        ipt.place(x=850, y=270, width=150, height=30)
        ipt.insert(0, '42')
        ipt.bind("<FocusOut>",lambda event:  self.is_valid_char(ipt, 'int', '42'))
        return ipt
    def __model_load_or_save_folder_label(self,parent):
        label = Label(parent,text="模型保存/加载文件夹",anchor="center")
        label.place(x=190, y=360, width=90, height=30)
        return label
    def __model_load_or_save_folder_entry(self,parent):
        ipt = Entry(parent)
        ipt.place(x=320, y=360, width=150, height=30)
        ipt.insert(0, './models')
        ipt.config(state='readonly')
        ipt.bind("<Button-1>",self.select_load_or_save_folder)
        return ipt
    def __on_pre_train_label(self,parent):
        label = Label(parent,text="预训练模式",anchor="center", )
        label.place(x=720, y=360, width=90, height=30)
        return label
    def __on_pre_train_button(self,parent):
        self.var1 = tk.IntVar()
        self.var1.set(0)
        cb = tk.Checkbutton(parent, text='开启',variable=self.var1, onvalue=1, offvalue=0, command=self.on_pre_train_button_select)
        cb.place(x=850, y=360, width=150, height=30)
    def __save_sets_button(self,parent):
        btn = Button(parent, text="保存配置", takefocus=False,command=self.save_sets)
        btn.place(x=1024, y=602, width=56, height=30)
        self.model_select_box.event_generate("<<ComboboxSelected>>")
        return btn
    
    # 模型选择框选择事件
    def model_select_box_select(self,event):
        # 获取当前结构类型
        model_secect_box_value = event.widget.get()
        # 获取保存加载文件夹路径
        model_load_or_save_folder = self.model_load_or_save_folder_entry.get()
        model_path = None
        dir_index =[] # 子目录索引(train1,train2,train3)
        # 修改加载模型框的值
        if model_secect_box_value == 'CNN（浅层应用）':
            self.master_App.selectd_model_type = 'CNN'
            model_path = os.path.join(model_load_or_save_folder, 'CNN')
        elif model_secect_box_value == 'LeNet-5':
            self.master_App.selectd_model_type = 'LeNet-5'
            model_path = os.path.join(model_load_or_save_folder, 'LeNet-5')
        elif model_secect_box_value == 'AlexNet':
            self.master_App.selectd_model_type = 'AlexNet'
            model_path = os.path.join(model_load_or_save_folder, 'AlexNet')
        elif model_secect_box_value == 'VGGNet-16':
            self.master_App.selectd_model_type = 'VGGNet-16'
            model_path = os.path.join(model_load_or_save_folder, 'VGGNet-16')
        elif model_secect_box_value == 'ResNet-50':
            self.master_App.selectd_model_type = 'ResNet-50'
            model_path = os.path.join(model_load_or_save_folder, 'ResNet-50')
        elif model_secect_box_value == 'InceptionV3':
            self.master_App.selectd_model_type = 'InceptionV3'
            model_path = os.path.join(model_load_or_save_folder, 'InceptionV3')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # 获取目录下的所有子目录
        dirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
        # 计算子目录有model.h5文件的数量
        for dir in dirs:
            if os.path.exists(os.path.join(model_path, dir, 'model.h5')):
                dir_index.append(dir)
        self.model_load_box['values'] = ['训练新的模型']+dir_index
        if dir_index == []:
            self.model_load_box['values'] = ('无模型可加载，训练新的模型')
        self.model_load_box.current(0)
        # 只选择了类型，没有选择加载，所以不加载模型，设置加载模型标志为False和None
        self.is_load_model = False
        self.load_model_path = None
        print(self.master_App.selectd_model_type, self.is_load_model, self.load_model_path)
        self.master_App.log_frame.add_log("选择模型结构：{}".format(self.master_App.selectd_model_type), 'info')
        self.master_App.log_frame.add_log("是否加载模型：{}".format(self.is_load_model), 'info')
        self.master_App.log_frame.add_log("加载模型路径：{}".format(self.load_model_path), 'info')


    # 保存设置
    def save_sets(self):
        pass
    
    # 预训练模式选择事件
    def on_pre_train_button_select(self):
        if self.var1.get() == 1:
            self.master_App.on_pre_train = True
        else:
            self.master_App.on_pre_train = False
        print('开启预训练：',self.master_App.on_pre_train)
        self.master_App.log_frame.add_log("开启预训练：{}".format(self.master_App.on_pre_train), 'info')

        
    # 选择加载保存模型文件夹
    def select_load_or_save_folder(self,event):
        event.widget.config(state='normal')
        directory = filedialog.askdirectory()  # 打开文件选择对话框
        if not directory:
            event.widget.config(state='readonly')
            return
        event.widget.delete(0, 'end')  # 删除 Entry 控件中的旧内容
        event.widget.insert(0, directory)  # 插入新选择的文件夹路径
        event.widget.config(state='readonly')

    # 模型加载框选择事件
    def model_load_box_select(self,event):
        model_load_box_value = event.widget.get()
        if model_load_box_value == '训练新的模型' or model_load_box_value == '无模型可加载，训练新的模型':
            self.is_load_model = False
            self.load_model_path = None
            self.master_App.load_model_path_pre = None
        else:
            self.is_load_model = True
            self.load_model_path = os.path.join(os.path.abspath(self.model_load_or_save_folder_entry.get()), self.master_App.selectd_model_type, model_load_box_value, 'model.h5')
            self.master_App.load_model_path_pre = self.load_model_path
            self.master_App.frames[TestPage].effect_image_path = os.path.join(os.path.abspath(self.model_load_or_save_folder_entry.get()), self.master_App.selectd_model_type, model_load_box_value, '训练效果.png')
            self.master_App.frames[TestPage].update_train_effect()
        print(self.is_load_model, self.load_model_path)
        self.master_App.log_frame.add_log("是否加载模型：{}".format(self.is_load_model), 'info')
        self.master_App.log_frame.add_log("加载模型路径：{}".format(self.load_model_path), 'info')

    # 限制输入类型
    def is_valid_char(event,element,input_type,default):
        s = element.get()
        if input_type == 'int':
            # 判断是否为整数
            if re.match(r'^-?\d+$', s):
                return True
            messagebox.showwarning('警告', '请输入整数')
        # 判断是否为 0-1 之间的小数
        if input_type == 'float':
            if re.match(r'^0\.\d+$', s):
                return True
            messagebox.showwarning('警告', '请输入0-1之间的小数')
        element.delete(0, tk.END)
        element.insert(0, default)
        return False



# 训练页面
class TrainPage(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        # self.configure(width=1170, height=400)
        self.grid(sticky='nsew')
        self.master_App = master
        master.grid_rowconfigure(1, weight=1)
        master.grid_columnconfigure(0, weight=1) 
        self.grid_propagate(1) # 使窗口自适应大小

        # 创建显示块
        self.view_frame = ttk.Frame(self,height=500)
        self.view_frame.pack(side=tk.TOP,fill=tk.BOTH,expand=True)

        # 创建后训练类
        self.after_train = self.AfterTrain(self)
        self.after_train.place(x=50, y=20)

        # 创建预训练类
        self.pre_train = self.PreTrain(self)
        self.pre_train.place(x=850, y=20)

        # 创建日志输出台
        self.log_frame = LogClass(self)
        self.log_frame.frame.pack(side=tk.BOTTOM, fill=tk.BOTH)
        self.master_App.log_frame = self.log_frame


    # 训练数据处理类
    class TrainDateHandle:
        def __init__(self, master, train_type='pre_train', folder_path=None, txt_file_path=None):
            self.train_type = train_type
            self.master_Train = master
            self.folder_path = folder_path
            self.txt_file_path = txt_file_path
            self.image_info_dict = {}
        # 训练启动方法
        def train_start(self):
            if self.train_type == 'pre_train':
                self.pre_train()
            elif self.train_type == 'after_train':
                self.after_train()
        
        # 车牌训练
        def after_train(self):
            # 获取模型结构选择
            model_type = self.master_Train.master_TrainPage.master_App.selectd_model_type
            # 创建模型训练对象
            # 打印
            self.master_Train.master_TrainPage.log_frame.add_log("开始训练车牌，模型结构：{}".format(model_type), 'info')
            self.model_train = Model_train(self.master_Train.master_TrainPage.master_App,self.master_Train,SetPage,model_type=model_type)
            # 使用模型测试接口
            self.model_train.test(r"D:\My_Code_Project\Image_Dateset\car_number\train\0275-90_269-246&438_534&535-534&528_246&535_247&441_533&438-0_0_3_24_31_29_26_24-187-33.jpg", train_type='car_number', 
                                  img_dir_path = self.folder_path,load_model_path=self.master_Train.master_TrainPage.master_App.load_model_path_pre)
            # 训练完成，设置训练状态
            self.model_train.train_is_stop = False
            self.model_train.train_is_stop = False

        # 预训练(不需要提取车牌操作，直接用单数字训练集)
        def pre_train(self):
            # 获取模型结构选择
            model_type = self.master_Train.master_TrainPage.master_App.selectd_model_type
            # 创建模型训练对象
            # 打印
            self.master_Train.master_TrainPage.log_frame.add_log("开始预训练，模型结构：{}".format(model_type), 'info')
            self.model_train = Model_train(self.master_Train.master_TrainPage.master_App,self.master_Train,SetPage,model_type=model_type)
            # 使用模型测试接口
            self.model_train.test('D:\My_Code_Project\Image_Dateset\single_number\VehicleLicense\Data\wan\wan_0000.jpg', train_type='single_number', 
                                  txt_file_path=self.txt_file_path, load_model_path=self.master_Train.master_TrainPage.master_App.load_model_path_pre)
            # 训练完成，设置训练状态
            self.model_train.train_is_stop = False
            self.model_train.train_is_stop = False

            

    # 进度条类(上层：预训练类/后训练类)
    class ProgressBar(ttk.Frame):
        def __init__(self,master):
            super().__init__(master, width=280, height=25)
            self.master_Train = master
            self.label = ttk.Label(self, text="训练进度：")
            self.label.place(x=0, y=0)
            self.progressbar = ttk.Progressbar(self, length=200, mode='determinate') # 显示进度条进度
            self.progressbar.place(x=60, y=0)
            self.progressbar_label = ttk.Label(self, text="0%", width=4 , font=("微软雅黑", 8))
            self.progressbar_label.place(x=self.progressbar.winfo_reqwidth()//2+60, y=11,anchor='center', height=16)
        
        # 进度条更新（参数：进度数量）
        def update_progressbar(self):
            # 判断是否暂停，改标签
            if self.master_Train.master_TrainPage.master_App.train_status.is_set():
                if self.master_Train.master_TrainPage.master_App.train_is_stop:
                    self.progressbar_label['text'] = "训练中断..."
                else:
                    self.progressbar_label['text'] = "训练完成"
                self.progressbar_label['width'] = 9
                return
            self.progressbar_label['width'] = 3
            self.progressbar['maximum'] = self.master_Train.progress_nums_all
            self.progressbar['value'] = self.master_Train.progress_nums
            self.master_Train.after(100, self.update_progressbar) # 递归调用，每100毫秒调用一次update_progressbar()方法
            self.progressbar_label['text'] = str(int(self.master_Train.progress_nums/self.master_Train.progress_nums_all*100)) + '%'

    # 后训练类+gui（上层：测试页面）
    class AfterTrain(ttk.Frame):
        def __init__(self, master):
            # 创建一个新的样式
            style = ttk.Style()
            # style.configure('My.TFrame', background='red')
            super().__init__(master.view_frame, borderwidth=2, relief='groove')
            self.place(width=300, height=150)
            self.master_TrainPage = master
            self.folder_path = None
            self.image_info_dict = {}
            # 进度条初始化
            self.progress_nums = 0
            self.progress_nums_all = 9999999999999
            self.train_data_handle = None

            # 添加标签说明
            self.type_label = ttk.Label(self, text="车牌训练", font=("微软雅黑", 11))
            self.type_label.place(x=100, y=0)
            # 添加问号提示
            query_image =Image.open('./label_image/image1.png')
            query_image = query_image.resize((15,15), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(query_image)
            self.query_label = ttk.Label(self, image=self.photo) 
            self.query_label.place(x=166,y=3)
            # 添加说明
            ToolTip(self.query_label, "此训练模块应选择车牌数据集")

            # 加入文件夹选择frame(按钮+输入框)
            self.select_frame = ttk.Frame(self, width=200, height=100)
            self.select_frame.folder_path = tk.StringVar()
            self.select_frame.folder_path.set('D:/My_Code_Project/三下机器学习课设/解压数据包/车牌/CCPD2020/test')
            self.select_frame.place(x=70, y=40)
            # 路径显示框
            self.select_frame.folder_entry = ttk.Entry(self.select_frame, textvariable=self.select_frame.folder_path, state=tk.DISABLED)
            self.select_frame.folder_entry.place(x=0, y=0)
            # 选择按钮
            self.select_frame.select_button = ttk.Button(self.select_frame, text="训练选择", command=self.select_folder)
            self.select_frame.select_button.place(x=0,y=30)
            # 训练按钮
            self.select_frame.train_button = ttk.Button(self.select_frame, text="开始训练", command=self.train_click_add_thread)
            self.select_frame.train_button.place(x=75,y=30)
            # 进度条初始化
            self.progress_bar = self.master_TrainPage.ProgressBar(self)
            self.progress_bar.place(x=15, y=120)

        
        # 选择按钮点击事件
        def select_folder(self):
            old_folder = self.select_frame.folder_path.get()
            folder_selected = filedialog.askdirectory()
            if not folder_selected:
                folder_selected = old_folder
            self.select_frame.folder_path.set(folder_selected)

        # 训练按钮点击事件加入线程
        def train_click_add_thread(self):
            threading.Thread(target=self.train_click).start()

        # 训练按钮点击事件
        def train_click(self):
            folder_path = self.select_frame.folder_path.get()
            if folder_path == '      请选择训练文件夹' or not folder_path:
                self.master_TrainPage.log_frame.add_log("请选择训练文件夹", 'warning')
            else:
                # 开始训练
                if self.master_TrainPage.master.train_status.is_set():
                    self.select_frame.train_button.config(text="停止训练")
                    self.master_TrainPage.master_App.train_is_stop = False
                    self.master_TrainPage.master.train_status.clear()
                    self.master_TrainPage.log_frame.add_log("开始训练\n训练进度······", 'info')
                    # 禁用选择文件夹按钮
                    self.select_frame.select_button.config(state=tk.DISABLED, cursor='no')
                    self.train_data_handle = self.master_TrainPage.TrainDateHandle(self, train_type='after_train', folder_path=folder_path, txt_file_path=folder_path)
                    # 进度条更新
                    self.progress_bar.update_progressbar()
                    # 使用多个线程训练
                    future = threadPool.submit(self.train_data_handle.train_start())
                    # 训练完成，设置训练状态
                    self.master_TrainPage.master.train_status.set()
                    self.select_frame.train_button.config(text="开始训练")
                    # print('训练完成')
                    
                else:     # 停止训练
                    self.select_frame.train_button.config(text="开始训练")
                    self.master_TrainPage.log_frame.add_log("人为停止训练", 'info')
                    self.master_TrainPage.master.train_status.set()
                    self.master_TrainPage.master_App.train_is_stop = True
                    print("人为停止训练")
                    # 启用选择文件夹按钮
                    self.select_frame.select_button.config(state=tk.NORMAL, cursor='arrow')

        


    # 预训练类+gui
    class PreTrain(ttk.Frame):
        def __init__(self, master):
            # 创建一个新的样式
            style = ttk.Style()
            # style.configure('My.TFrame', background='red')
            super().__init__(master.view_frame,borderwidth=2, relief='groove')
            self.place(width=300, height=150)
            self.master_TrainPage = master
            self.folder_path = None
            self.image_info_dict = {}
            self.progress_nums = 0
            self.progress_nums_all = 9999999999999

            # 添加标签说明
            self.type_label = ttk.Label(self, text="预训练", font=("微软雅黑", 11))
            self.type_label.place(x=110, y=0)
            
            # 添加问号提示
            query_image =Image.open('./label_image/image1.png')
            query_image = query_image.resize((15,15), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(query_image)
            self.query_label = ttk.Label(self, image=self.photo) 
            self.query_label.place(x=166,y=3)
            # 添加说明
            ToolTip(self.query_label, "预训练模块是提前对单\n字符进行模型训练并保存，\n然后再学习车牌数据集")

            # 加入文件夹选择frame(按钮+输入框)
            self.select_frame = ttk.Frame(self, width=200, height=100)
            self.select_frame.folder_path = tk.StringVar()
            self.select_frame.folder_path.set("D:\My_Code_Project\Image_Dateset\single_number\VehicleLicense\\trainval.txt")
            self.select_frame.place(x=70,y=40)
            # 路径显示框
            self.select_frame.folder_entry = ttk.Entry(self.select_frame, textvariable=self.select_frame.folder_path, state=tk.DISABLED)
            self.select_frame.folder_entry.place(x=0, y=0)
            # 选择按钮
            self.select_frame.select_button = ttk.Button(self.select_frame, text="训练选择", command=self.select_txt)
            self.select_frame.select_button.place(x=0,y=30)
            # 训练按钮
            self.select_frame.train_button = ttk.Button(self.select_frame, text="开始训练", command=self.train_click_add_thread)
            self.select_frame.train_button.place(x=75,y=30)
            # 进度条初始化
            self.progress_bar = self.master_TrainPage.ProgressBar(self)
            self.progress_bar.place(x=15, y=120)

        # 选择按钮点击事件
        def select_txt(self):
            old_path = self.select_frame.folder_path.get()
            txt_path = filedialog.askopenfilename(filetypes=[("单字符训练导入文件", "*.txt")])
            # folder_selected = filedialog.askdirectory(filetypes=[("文件夹", ".txt")])
            txt_path_selected = txt_path
            if not txt_path_selected:
                txt_path = old_path
            self.select_frame.folder_path.set(txt_path)

        # 训练按钮点击事件加入线程
        def train_click_add_thread(self):
            threading.Thread(target=self.train_click).start()

        # 训练按钮点击事件
        def train_click(self):
            folder_path = self.select_frame.folder_path.get()   # 这是预训练，不需要选择文件夹
            txt_file_path = self.select_frame.folder_path.get()
            if folder_path == '      请选择训练文件夹' or not folder_path:
                self.master_TrainPage.log_frame.add_log("请选择训练文件夹", 'warning')
            else:
                # 开始训练
                if self.master_TrainPage.master_App.train_status.is_set():
                    self.select_frame.train_button.config(text="停止训练")
                    self.master_TrainPage.master_App.train_status.clear()
                    self.master_TrainPage.log_frame.add_log("开始训练\n训练进度······", 'info')
                    # 禁用选择文件夹按钮
                    self.select_frame.select_button.config(state=tk.DISABLED, cursor='no')
                    self.train_data_handle = self.master_TrainPage.TrainDateHandle(self, train_type='pre_train', folder_path=folder_path, txt_file_path=folder_path)
                    # 进度条更新
                    self.progress_bar.update_progressbar()
                    # 使用多个线程训练
                    future = threadPool.submit(self.train_data_handle.train_start())
                    print('训练完成')
                    self.master_TrainPage.master_App.log_frame.add_log("训练完成", 'info')
                    self.select_frame.train_button.config(text="开始训练")
                    self.master_TrainPage.master_App.train_status.set()
                    
                else:     # 停止训练
                    self.select_frame.train_button.config(text="开始训练")
                    self.master_TrainPage.log_frame.add_log("预训练停止中，请稍等···", 'info')
                    self.master_TrainPage.master_App.train_status.set()
                    print("人为停止训练")
                    self.master_TrainPage.master_App.log_frame.add_log("人为停止训练", 'info')
                    # 启用选择文件夹按钮
                    self.select_frame.select_button.config(state=tk.NORMAL, cursor='arrow')


# 测试页面
class TestPage(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.grid(sticky='nsew')
        master.grid_rowconfigure(1, weight=1)
        master.grid_columnconfigure(0, weight=1) 
        self.grid_propagate(1)
        self.master_App = master
        # 图片路径
        self.effect_image_path = None

        self.tk_label_lup57mn5 = self.__tk_label_lup57mn5(self)
        self.tk_label_lup5joph = self.__tk_label_lup5joph(self)

    def __tk_label_lup57mn5(self,parent):
        label = Label(parent,text="请加载模型或训练模型",anchor="center", borderwidth=1, relief="solid")
        label.place(x=190, y=120, width=800, height=400)
        return label
    def __tk_label_lup5joph(self,parent):
        label = Label(parent,text="模型训练效果",anchor="center", )
        label.place(x=80, y=45, width=90, height=30)
        return label
    
    # 更新训练效果显示
    def update_train_effect(self):
        if self.effect_image_path:
            # 读取图片并转换为Tkinter可以使用的格式
            image = Image.open(self.effect_image_path)
            image = image.resize((800,400), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(image)
            self.tk_label_lup57mn5.configure(image=self.photo)
        

# 预测页面
class YuCePage(ttk.Frame):   
    def __init__(self, master):
        super().__init__(master)
        self.grid(sticky='nsew')
        master.grid_rowconfigure(1, weight=1)
        master.grid_columnconfigure(0, weight=1) 
        self.grid_propagate(1)
        self.image_path = None
        self.master_App = master
        self.car_number = None
        
        self.tk_label_luo4j9hx = self.__tk_label_luo4j9hx(self)
        self.tk_button_luo4l71o = self.__tk_button_luo4l71o(self)
        self.tk_button_luo4l92v = self.__tk_button_luo4l92v(self)
        self.tk_label_luo4nqp2 = self.__tk_label_luo4nqp2(self)

    def __tk_label_luo4j9hx(self,parent):
        label = Label(parent,text="请选择图片",anchor="center", borderwidth=1, relief="solid")
        label.place(x=140, y=200, width=400, height=100)
        return label
    def __tk_button_luo4l71o(self,parent):
        btn = Button(parent, text="图片选择", takefocus=False,command=self.select_image)
        btn.place(x=200, y=360, width=90, height=30)
        return btn
    def __tk_button_luo4l92v(self,parent):
        btn = Button(parent, text="开始识别", takefocus=False,command=self.start_shibie)
        btn.place(x=380, y=360, width=90, height=30)
        return btn
    def __tk_label_luo4nqp2(self,parent):
        label = Label(parent,text="车牌号XXXXXXXX",anchor="center", borderwidth=1, relief="solid")
        label.place(x=700, y=208, width=280, height=80)
        return label
    
    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if file_path:
            # 读取图片并转换为Tkinter可以使用的格式
            self.image_path = file_path
            image = Image.open(file_path)
            self.photo = ImageTk.PhotoImage(image)
            self.tk_label_luo4j9hx.configure(image=self.photo)

    # 更新车牌号显示Label
    def update_car_number(self):
        self.tk_label_luo4nqp2['text'] = self.car_number

    def start_shibie(self):
        # 没有开始训练
        if self.master_App.frames[TrainPage].after_train.train_data_handle is None:
            model_train = Model_train(self.master_App,None,None,self.master_App.selectd_model_type)
            self.car_number = model_train.shibie_test(img_path=self.image_path,load_model_path=self.master_App.load_model_path_pre)
            self.update_car_number()
        else:   #已有训练
            self.car_number = self.master_App.frames[TrainPage].after_train.train_data_handle.model_train.shibie_test(img_path=self.image_path,load_model_path=self.master_App.load_model_path_pre)
            self.update_car_number()
# 自定义的日志处理器，用于将日志输出到指定的文本框中
class TextHandler(logging.Handler):
    def __init__(self, text):
        logging.Handler.__init__(self)
        self.text = text

    def emit(self, record):
        msg = self.format(record)
        self.text.configure(state='normal')
        self.text.insert(tk.END, msg + '\n')
        self.text.configure(state='disabled')
        # 自动滚动到底部
        self.text.yview(tk.END)

# 打印类，一个frame，包含log按钮、log文本框和滚动条
class LogClass:
    def __init__(self, master):
        self.master = master
        # 创建日志输出台和滚动条
        self.frame = ttk.Frame(self.master,height=200)
        self.scrollbar = tk.Scrollbar(self.frame, orient=tk.VERTICAL)
        self.console = tk.Text(self.frame, yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.console.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)  
        self.console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 创建日志处理器
        self.logger = logging.getLogger()
        self.logger.addHandler(TextHandler(self.console))
        self.logger.setLevel(logging.INFO)
        logging.warning("日志记录器已启动")

    def add_log(self, log, type='info'):
        if type == 'info':
            logging.info(log)
        elif type == 'warning':
            logging.warning(log)
        elif type == 'error':
            logging.error(log)

# 导航按钮类
class NavButton:
    def __init__(self,master):
        self.master = master
        # 创建按钮框架
        style = ttk.Style(self.master)
        style.configure("TButton", font=("微软雅黑", 8), borderwidth=0)

        self.frame = ttk.Frame(self.master)

        train_button = ttk.Button(self.frame, text="数据集训练", command=lambda: self.show_frame(TrainPage))
        train_button.pack(side='left', padx=0)

        set_button = ttk.Button(self.frame, text="测试结果", command=lambda: self.show_frame(TestPage))
        set_button.pack(side='left', padx=0)

        test_button = ttk.Button(self.frame, text="预测", command=lambda: self.show_frame(YuCePage))
        test_button.pack(side='left', padx=0)

        set_button = ttk.Button(self.frame, text="设置", command=lambda: self.show_frame(SetPage))
        set_button.pack(side='left', padx=0)


    # 显示指定页面
    def show_frame(self, page):
        self.master.frame = self.master.frames[page]
        self.master.frame.tkraise() #tkraise()方法用于将指定的窗口部件移动到其父级的堆栈顶部
        # if page.__name__ == 'TrainPage':
        #     # 创建日志输出台
        #     self.master.log_frame = LogClass(self.master)
        #     self.master.log _frame.frame.grid(row=2, column=0, sticky="ew")
        # else :
        #     if hasattr(self.master, 'log_frame'):
        #         self.master.log_frame.frame.grid_forget()

# 工具提示类，鼠标悬停在小部件上时显示工具提示
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x = y = 0
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tooltip, text=self.text, background="#ffffe0", relief="solid", borderwidth=1).pack()

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


if __name__ == "__main__":
    app = Application()
    app.mainloop()
 