import tkinter as tk
from tkinter import ttk,filedialog
import logging
import 图像提取
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import os
import time

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
    
# 设置页面
class SetPage(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.configure(width=1170, height=400)
        self.grid(sticky='nsew') 
        master.grid_rowconfigure(1, weight=1)
        master.grid_columnconfigure(0, weight=1) 
        self.grid_propagate(1)
        label = ttk.Label(self, text="设置")
        label.place(x=180, y=140)        

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


    # 训练数据处理类
    class TrainDateHandle:
        def __init__(self, folder_path, master):
            self.master_Train = master
            self.folder_path = folder_path
            self.image_info_dict = {}
        # 训练启动方法
        def train_start(self):
            self.get_image_info()
        
        # 获取图像信息
        def get_image_info(self):
            files = os.listdir(self.folder_path)
            self.txtq = 图像提取.TuXiangTiQu(self.folder_path)
            for name in files:
                # 判断是否暂停
                if self.master_Train.master_TrainPage.master_App.train_status.is_set():
                    self.master_Train.log_frame.add_log("人为停止训练", 'info')
                    self.master_Train.progress_nums = 0
                    print(self.txtq.image_info_dict)
                    return
                # 获取图片信息
                image_info = self.txtq.start(name)
                self.master_Train.progress_nums += 1
                self.master_Train.master_TrainPage.log_frame.add_log("图片信息：{}".format(image_info),'info')
                print("图片信息：{}".format(image_info))
                self.master_Train.master_TrainPage.log_frame.add_log("训练进度：{}/{}".format(self.master_Train.progress_nums, len(files)), 'info')
                self.master_Train.progress_nums_all = len(files) # 进度条总数值 

    # 进度条类(上层：预训练类/后训练类)
    class ProgressBar(ttk.Frame):
        def __init__(self,master):
            super().__init__(master, width=280, height=25)
            self.master_Train = master
            self.label = ttk.Label(self, text="训练进度：")
            self.label.place(x=0, y=0)
            self.progressbar = ttk.Progressbar(self, length=200, mode='determinate') # 显示进度条进度
            self.progressbar.place(x=60, y=0)
            self.progressbar_label = ttk.Label(self, text="0%", width=3 , font=("微软雅黑", 8))
            self.progressbar_label.place(x=self.progressbar.winfo_reqwidth()//2+60, y=11,anchor='center', height=16)
        
        # 进度条更新（参数：进度数量）
        def update_progressbar(self):
            # 判断是否暂停，改标签
            if self.master_Train.master_TrainPage.master_App.train_status.is_set():
                # self.progressbar['value'] = 0
                self.progressbar_label['text'] = "训练中断..."
                # self.progressbar_label = ttk.Label(self, text="0%", width=3 , font=("微软雅黑", 8))
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
            style.configure('My.TFrame', background='red')
            super().__init__(master.view_frame, style='My.TFrame')
            self.place(width=300, height=150)
            self.master_TrainPage = master
            self.folder_path = None
            self.image_info_dict = {}
            # 进度条初始化
            self.progress_nums = 0
            self.progress_nums_all = 9999999999999

            # 加入文件夹选择frame(按钮+输入框)
            self.select_frame = ttk.Frame(self, width=200, height=100)
            self.select_frame.folder_path = tk.StringVar()
            self.select_frame.folder_path.set('D:/My_Code_Project/三下机器学习课设/解压数据包/车牌/CCPD2020/test')
            self.select_frame.place(x=40, y=10)
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
            self.progress_bar.place(x=0, y=100)

        
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
                    self.master_TrainPage.master.train_status.clear()
                    self.master_TrainPage.log_frame.add_log("开始训练\n训练进度······", 'info')
                    # 禁用选择文件夹按钮
                    self.select_frame.select_button.config(state=tk.DISABLED, cursor='no')
                    self.train_data_handle = self.master_TrainPage.TrainDateHandle(folder_path,self)
                    # 进度条更新
                    self.progress_bar.update_progressbar()
                    # 使用多个线程训练
                    future = threadPool.submit(self.train_data_handle.train_start())
                    
                else:     # 停止训练
                    self.select_frame.train_button.config(text="开始训练")
                    self.master_TrainPage.log_frame.add_log("人为停止训练", 'info')
                    self.master_TrainPage.master.train_status.set()
                    print("人为停止训练")
                    # 启用选择文件夹按钮
                    self.select_frame.select_button.config(state=tk.NORMAL, cursor='arrow')

        


    # 预训练类+gui
    class PreTrain(ttk.Frame):
        def __init__(self, master):
            # 创建一个新的样式
            style = ttk.Style()
            style.configure('My.TFrame', background='red')
            super().__init__(master.view_frame, style='My.TFrame')
            self.place(x=0, y=200, width=300, height=150)
            self.master_TrainPage = master
            self.folder_path = None
            self.image_info_dict = {}
            self.progress_nums = 0
            self.progress_nums_all = 9999999999999

            # 加入文件夹选择frame(按钮+输入框)
            self.select_frame = ttk.Frame(self, width=200, height=100)
            self.select_frame.folder_path = tk.StringVar()
            self.select_frame.folder_path.set('D:/My_Code_Project/三下机器学习课设/解压数据包/车牌/CCPD2020/test')
            self.select_frame.place(x=40,y=10)
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
            self.progress_bar.place(x=0, y=100)

        # 预训练启动方法
        def pre_train_start(self):
            self.get_image_info()

        # 获取图像信息
        def get_image_info(self):
            files = os.listdir(self.folder_path)
            self.txtq = 图像提取.TuXiangTiQu(self.folder_path)
            for name in files:
                # 判断是否暂停
                if self.master.master.train_status.is_set():
                    self.master.log_frame.add_log("人为停止训练", 'info')
                    self.master.progress_nums = 0
                    print(self.txtq.image_info_dict)
                    return
                # 获取图片信息
                image_info = self.txtq.start(name)
                self.master.progress_nums += 1
                self.master.log_frame.add_log("图片信息：{}".format(image_info),'info')
                print("图片信息：{}".format(image_info))
                self.master.log_frame.add_log("训练进度：{}/{}".format(self.master.progress_nums, len(files)), 'info')
                self.master.progress_nums_all = len(files) # 进度条总数值
        
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


# 测试页面
class TestPage(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.grid(sticky='nsew')
        master.grid_rowconfigure(1, weight=1)
        master.grid_columnconfigure(0, weight=1) 
        self.grid_propagate(1)
        label = ttk.Label(self, text="训练结果展示")
        label.place(x=180, y=140)

class YuCePage(ttk.Frame):   
    def __init__(self, master):
        super().__init__(master)
        self.grid(sticky='nsew')
        master.grid_rowconfigure(1, weight=1)
        master.grid_columnconfigure(0, weight=1) 
        self.grid_propagate(1)
        label = ttk.Label(self, text="预测")
        label.place(x=180, y=140)

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



if __name__ == "__main__":
    app = Application()
    app.mainloop()
 