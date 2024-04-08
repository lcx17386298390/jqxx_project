# import random
# from tkinter import *
# from tkinter.ttk import *
# class WinGUI(Tk):
#     def __init__(self):
#         super().__init__()
#         self.__win()
#         self.tk_label_luo4j9hx = self.__tk_label_luo4j9hx(self)
#         self.tk_button_luo4l71o = self.__tk_button_luo4l71o(self)
#         self.tk_button_luo4l92v = self.__tk_button_luo4l92v(self)
#         self.tk_label_luo4nqp2 = self.__tk_label_luo4nqp2(self)
#     def __win(self):
#         self.title("Tkinter布局助手")
#         # 设置窗口大小、居中
#         width = 1200
#         height = 800
#         screenwidth = self.winfo_screenwidth()
#         screenheight = self.winfo_screenheight()
#         geometry = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
#         self.geometry(geometry)
        
#         self.resizable(width=False, height=False)
        
#     def scrollbar_autohide(self,vbar, hbar, widget):
#         """自动隐藏滚动条"""
#         def show():
#             if vbar: vbar.lift(widget)
#             if hbar: hbar.lift(widget)
#         def hide():
#             if vbar: vbar.lower(widget)
#             if hbar: hbar.lower(widget)
#         hide()
#         widget.bind("<Enter>", lambda e: show())
#         if vbar: vbar.bind("<Enter>", lambda e: show())
#         if vbar: vbar.bind("<Leave>", lambda e: hide())
#         if hbar: hbar.bind("<Enter>", lambda e: show())
#         if hbar: hbar.bind("<Leave>", lambda e: hide())
#         widget.bind("<Leave>", lambda e: hide())
    
#     def v_scrollbar(self,vbar, widget, x, y, w, h, pw, ph):
#         widget.configure(yscrollcommand=vbar.set)
#         vbar.config(command=widget.yview)
#         vbar.place(relx=(w + x) / pw, rely=y / ph, relheight=h / ph, anchor='ne')
#     def h_scrollbar(self,hbar, widget, x, y, w, h, pw, ph):
#         widget.configure(xscrollcommand=hbar.set)
#         hbar.config(command=widget.xview)
#         hbar.place(relx=x / pw, rely=(y + h) / ph, relwidth=w / pw, anchor='sw')
#     def create_bar(self,master, widget,is_vbar,is_hbar, x, y, w, h, pw, ph):
#         vbar, hbar = None, None
#         if is_vbar:
#             vbar = Scrollbar(master)
#             self.v_scrollbar(vbar, widget, x, y, w, h, pw, ph)
#         if is_hbar:
#             hbar = Scrollbar(master, orient="horizontal")
#             self.h_scrollbar(hbar, widget, x, y, w, h, pw, ph)
#         self.scrollbar_autohide(vbar, hbar, widget)
#     def __tk_label_luo4j9hx(self,parent):
#         label = Label(parent,text="标签",anchor="center", )
#         label.place(x=140, y=200, width=400, height=100)
#         return label
#     def __tk_button_luo4l71o(self,parent):
#         btn = Button(parent, text="图片选择", takefocus=False,)
#         btn.place(x=200, y=360, width=90, height=30)
#         return btn
#     def __tk_button_luo4l92v(self,parent):
#         btn = Button(parent, text="开始识别", takefocus=False,)
#         btn.place(x=380, y=360, width=90, height=30)
#         return btn
#     def __tk_label_luo4nqp2(self,parent):
#         label = Label(parent,text="标签",anchor="center", )
#         label.place(x=700, y=208, width=280, height=80)
#         return label
# class Win(WinGUI):
#     def __init__(self, controller):
#         self.ctl = controller
#         super().__init__()
#         self.__event_bind()
#         self.__style_config()
#         self.ctl.init(self)
#     def __event_bind(self):
#         pass
#     def __style_config(self):
#         pass
# if __name__ == "__main__":
#     win = WinGUI()
#     win.mainloop()

import random
from tkinter import *
from tkinter.ttk import *
class WinGUI(Tk):
    def __init__(self):
        super().__init__()
        self.__win()
        self.tk_label_lup57mn5 = self.__tk_label_lup57mn5(self)
        self.tk_label_lup5joph = self.__tk_label_lup5joph(self)
    def __win(self):
        self.title("Tkinter布局助手")
        # 设置窗口大小、居中
        width = 1200
        height = 800
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        geometry = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.geometry(geometry)
        
        self.resizable(width=False, height=False)
        
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
    def __tk_label_lup57mn5(self,parent):
        label = Label(parent,text="请加载模型或训练模型",anchor="center", )
        label.place(x=190, y=120, width=800, height=400)
        return label
    def __tk_label_lup5joph(self,parent):
        label = Label(parent,text="模型训练效果",anchor="center", )
        label.place(x=80, y=45, width=90, height=30)
        return label
class Win(WinGUI):
    def __init__(self, controller):
        self.ctl = controller
        super().__init__()
        self.__event_bind()
        self.__style_config()
        self.ctl.init(self)
    def __event_bind(self):
        pass
    def __style_config(self):
        pass
if __name__ == "__main__":
    win = WinGUI()
    win.mainloop()
