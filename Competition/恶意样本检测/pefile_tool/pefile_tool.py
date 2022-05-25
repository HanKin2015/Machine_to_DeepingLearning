from tkinter import *
from tkinter.ttk import *
import tkinter
from tkinter import filedialog
import pefile
import re
import os
import windnd
from tkinter.filedialog import *
import win32gui
import win32ui
from PIL import Image
 
 
def main():
    root = Tk()
    root.title('剑工坊-PE分析工具')  # 程序的标题名称
    root.geometry("780x520+360+140")  # 窗口的大小及页面的显示位置
    root.resizable(False, False)  # 固定页面不可放大缩小
    root.iconbitmap("4.ico")  # 程序的图标
 
    photo = PhotoImage(file="./Key.png")
    theLabel = tkinter.Label(root, image=photo)
    theLabel.place(x=-1, y=-1)
 
    # 打开文件   核心文件存储
    var_file = tkinter.StringVar()
    tkinter.Entry(root, width=70, borderwidth=0,fg='#ea0f0f', bg='#ffffff', textvariable=var_file).place(x=105, y=10)
 
    # PE
    var_PE = tkinter.StringVar()
    tkinter.Entry(root, width=14, borderwidth=0,fg='#ea0f0f', bg='#ffffff',textvariable=var_PE).place(x=100, y=66)
 
    # EP段
    var_EP = tkinter.StringVar()
    tkinter.Entry(root, width=14, borderwidth=0,fg='#ea0f0f', bg='#ffffff', textvariable=var_EP).place(x=100, y=88)
 
    # 病毒检测
    var_BD = tkinter.StringVar()
    tkinter.Entry(root, width=14, borderwidth=0,fg='#ea0f0f', bg='#ffffff', textvariable=var_BD).place(x=322, y=66)
 
    # RVA大小
    var_RVA = tkinter.StringVar()
    tkinter.Entry(root, width=14, borderwidth=0,fg='#ea0f0f', bg='#ffffff', textvariable=var_RVA).place(x=322, y=90)
 
    # 入口点
    var_entry  = tkinter.StringVar()
    tkinter.Entry(root, width=14, borderwidth=0,fg='#ea0f0f', bg='#ffffff', textvariable=var_entry).place(x=100, y=176)
 
    # 效验和
    var_Validate  = tkinter.StringVar()
    tkinter.Entry(root, width=14, borderwidth=0,fg='#ea0f0f', bg='#ffffff', textvariable=var_Validate).place(x=100, y=196)
 
    # 子系统
    var_Subsystem  = tkinter.StringVar()
    tkinter.Entry(root, width=14, borderwidth=0,fg='#ea0f0f', bg='#ffffff', textvariable=var_Subsystem).place(x=100, y=216)
 
    # 头部大小
    var_head  = tkinter.StringVar()
    tkinter.Entry(root, width=14, borderwidth=0,fg='#ea0f0f', bg='#ffffff', textvariable=var_head).place(x=94, y=280)
 
    # 镜像大小
    var_image  = tkinter.StringVar()
    tkinter.Entry(root, width=14, borderwidth=0,fg='#ea0f0f', bg='#ffffff', textvariable=var_image).place(x=94, y=304)
 
    # 代码基址
    var_Code  = tkinter.StringVar()
    tkinter.Entry(root, width=14, borderwidth=0,fg='#ea0f0f', bg='#ffffff', textvariable=var_Code).place(x=94, y=330)
 
    # 数据基址
    var_data  = tkinter.StringVar()
    tkinter.Entry(root, width=14, borderwidth=0,fg='#ea0f0f', bg='#ffffff', textvariable=var_data).place(x=94, y=356)
 
    # 标志字
    var_sign  = tkinter.StringVar()
    tkinter.Entry(root, width=14, borderwidth=0,fg='#ea0f0f', bg='#ffffff', textvariable=var_sign).place(x=330, y=280)
 
    # 特征值
    var_features  = tkinter.StringVar()
    tkinter.Entry(root, width=14, borderwidth=0,fg='#ea0f0f', bg='#ffffff', textvariable=var_features).place(x=330, y=302)
 
    # 节数目
    var_Number  = tkinter.StringVar()
    tkinter.Entry(root, width=14, borderwidth=0,fg='#ea0f0f', bg='#ffffff', textvariable=var_Number).place(x=330, y=326)
 
    # 时间日期标志
    var_Date_flag  = tkinter.StringVar()
    tkinter.Entry(root, width=14, borderwidth=0,fg='#ea0f0f', bg='#ffffff', textvariable=var_Date_flag).place(x=330, y=410)
 
    # 可选头部大小
    var_optional_header  = tkinter.StringVar()
    tkinter.Entry(root, width=14, borderwidth=0,fg='#ea0f0f', bg='#ffffff', textvariable=var_optional_header).place(x=330, y=434)
 
    # 服务器版本
    var_server  = tkinter.StringVar()
    tkinter.Entry(root, width=14, borderwidth=0,fg='#ea0f0f', bg='#ffffff', textvariable=var_server).place(x=330, y=458)
 
    # 节对齐度
    var_Section_alignment  = tkinter.StringVar()
    tkinter.Entry(root, width=14, borderwidth=0,fg='#ea0f0f', bg='#ffffff', textvariable=var_Section_alignment).place(x=94, y=410)
 
    # 限定大小
    var_Limit_size  = tkinter.StringVar()
    tkinter.Entry(root, width=14, borderwidth=0,fg='#ea0f0f', bg='#ffffff', textvariable=var_Limit_size).place(x=94, y=434)
 
    # 镜像基址
    var_limit  = tkinter.StringVar()
    tkinter.Entry(root, width=14, borderwidth=0,fg='#ea0f0f', bg='#ffffff', textvariable=var_limit).place(x=94, y=458)
 
    # 文件对齐
    var_File_Alignment  = tkinter.StringVar()
    tkinter.Entry(root, width=14, borderwidth=0,fg='#ea0f0f', bg='#ffffff', textvariable=var_File_Alignment).place(x=94, y=482)
 
    # 文件说明
    var_Document_description = tkinter.StringVar()
    tkinter.Entry(root, width=30, borderwidth=0, fg='#ea0f0f', bg='#ffffff', textvariable=var_Document_description).place(x=530, y=312)
 
    # 版权
    var_copyright = tkinter.StringVar()
    tkinter.Entry(root, width=30, borderwidth=0, fg='#ea0f0f', bg='#ffffff', textvariable=var_copyright).place(x=530, y=348)
 
    # 文件版本
    var_File_version = tkinter.StringVar()
    tkinter.Entry(root, width=30, borderwidth=0, fg='#ea0f0f', bg='#ffffff', textvariable=var_File_version).place(x=530, y=384)
 
    # 产品版本
    var_product_version = tkinter.StringVar()
    tkinter.Entry(root, width=30, borderwidth=0, fg='#ea0f0f', bg='#ffffff', textvariable=var_product_version).place(x=530, y=420)
 
    # 签名
    var_autograph = tkinter.StringVar()
    tkinter.Entry(root, width=30, borderwidth=0, fg='#ea0f0f', bg='#ffffff', textvariable=var_autograph).place(x=530, y=456)
 
    # 大名
    var_Dname = tkinter.StringVar()
    tkinter.Entry(root, width=18, borderwidth=0, fg='#ea0f0f', bg='#ffffff',font=('',16),textvariable=var_Dname).place(x=530, y=114)
 
    def QK():  # 清空内容
        var_EP.set('')
        var_BD.set('')
        var_RVA.set('')
        var_entry.set('')
        var_Validate.set('')
        var_Subsystem.set('')
        var_head.set('')
        var_image.set('')
        var_Code.set('')
        var_data.set('')
        var_sign.set('')
        var_features.set('')
        var_Number.set('')
        var_Date_flag.set('')
        var_optional_header.set('')
        var_server.set('')
        var_Section_alignment.set('')
        var_Limit_size.set('')
        var_limit.set('')
        var_File_Alignment.set('')
        var_Document_description.set('')
        var_copyright.set('')
        var_File_version.set('')
        var_product_version.set('')
        var_autograph.set('')
        var_Dname.set('')
 
    # 图标
    image_file_3 = tkinter.PhotoImage(file="pictures.png")  # 软件第一次打开时要呈现的图片
    Button(root, image=image_file_3).place(x=471, y=104)
 
    # 更换软件图标
    def picture():
        try:
            image_file_3.config(file='icon.png')  # 替换
        except:
            pass
 
    ico_x = 32
    def ICON(exePath2):
        try:
 
            exePath = exePath2.replace("\\", "/")  # 替换
            large, small = win32gui.ExtractIconEx(f'{exePath}', 0)
            useIcon = large[0]
            destroyIcon = small[0]
            win32gui.DestroyIcon(destroyIcon)
            hdc = win32ui.CreateDCFromHandle(win32gui.GetDC(0))
            hbmp = win32ui.CreateBitmap()
            hbmp.CreateCompatibleBitmap(hdc, ico_x, ico_x)
            hdc = hdc.CreateCompatibleDC()
            hdc.SelectObject(hbmp)
            hdc.DrawIcon((0, 0), useIcon)
            bmpstr = hbmp.GetBitmapBits(True)
            img = Image.frombuffer(
                'RGBA',
                (32, 32),
                bmpstr, 'raw', 'BGRA', 0, 1
            )
            img.save('icon.png')
        except:
            pass
 
    def PE():   # 检测是否是PE文件
 
        try:
            fileinfo = os.stat(var_file.get())
 
            def formatTime(atime):
                import time
                return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(atime))
            ICON(var_file.get())
            picture()
            PE_file = pefile.PE(var_file.get())  # 读取pe文件
            PE_file_PE = PE_file.NT_HEADERS   # 检测是否是PE
            PE_file_MZ = PE_file.DOS_HEADER   # 检测是否是MZ
            PE_file_2 = PE_file.OPTIONAL_HEADER
            PE_file_3 = PE_file.FILE_HEADER
            PE_file_5 = PE_file.VS_FIXEDFILEINFO
            if hex(PE_file_MZ.e_magic) == '0x5a4d' and hex(PE_file_PE.Signature) == '0x4550':
                var_PE.set('PE文件')
 
                # EP段
                PE_q = PE_file.sections[0]  # 读取第一段   EP段
                EP_value = str(PE_q).replace(" ", "")  # 将空格替换为无（空值）
                file_EP = re.findall('Name:(.*)', EP_value)[0]  # 读取EP段
                var_EP.set(file_EP)
 
                # 病毒
                var_BD.set('暂未开发')
 
                # RVA检测
                var_RVA.set(hex(PE_file_2.NumberOfRvaAndSizes))  # RVA数目及大小
 
                # 入口点
                var_entry.set(hex(PE_file_2.AddressOfEntryPoint))
 
                # 效验和
                var_Validate.set(hex(PE_file_2.CheckSum))
 
                # 子系统
                var_Subsystem.set(hex(PE_file_2.Subsystem))
 
                # 头部大小
                var_head.set(hex(PE_file_2.SizeOfHeaders))
 
                # 镜像大小
                var_image.set(hex(PE_file_2.SizeOfImage))
 
                # 代码基址
                var_Code.set(hex(PE_file_2.BaseOfCode))
 
                # 数据基址
                var_data.set(hex(PE_file_2.BaseOfData))
 
                # 标志字
                var_sign.set(hex(PE_file_2.Magic))
 
                # 特征值
                var_features.set(hex(PE_file_3.Characteristics))
 
                # 节数目  //※PE文件中区块数量
                var_Number.set(hex(PE_file_3.NumberOfSections))
 
                # 时间日期标志
                var_Date_flag.set(hex(PE_file_3.TimeDateStamp))
 
                # 可选头部大小
                var_optional_header.set(hex(PE_file_3.SizeOfOptionalHeader))
 
                # 服务器版本
                var_server.set(hex(PE_file_2.MajorLinkerVersion))
 
                # 节对齐度
                var_Section_alignment.set(hex(PE_file_2.SectionAlignment))
 
                # 限定大小
                var_Limit_size.set(hex(PE_file_2.SizeOfInitializedData))
 
                # 镜像基址
                var_limit.set(hex(PE_file_2.ImageBase))
 
                # 文件对齐
                var_File_Alignment.set(hex(PE_file_2.FileAlignment))
 
                # 访问日期
                var_Document_description.set(formatTime(fileinfo.st_atime))
 
                # 安装日期
                var_copyright.set(formatTime(fileinfo.st_ctime))
 
                # 文件版本
                var_File_version.set(hex(PE_file_5[0].FileVersionMS))
 
                # 产品版本
                var_product_version.set(hex(PE_file_5[0].ProductVersionMS))
 
                # 签名
                var_autograph.set(hex(PE_file_5[0].Signature))
 
                # 大名
                var_Dname.set(os.path.basename(var_file.get()))
            else:
                var_PE.set('非有效PE文件！！')
                QK()
        except:
            var_PE.set('非有效PE文件！！')
            QK()
 
 
    def file(files):   # 使用拖拽
        msg = '\n'.join((item.decode('gbk') for item in files))
        files = msg.replace("\\", "/")  # 替换
        if not os.path.isfile(files):  # 判断是否为文件
            var_file.set('错误文件路径不存在  --  不可多选！！  请检测！！！')
        else:
            var_file.set(files)    # 判断为文件则执行
            PE()
 
 
 
 
    def dai_mck():   # 具体代码查看
        root_sk = Tk()
        root_sk.title(var_file.get())
        root_sk.geometry('600x500+200+200')
        root_sk.configure(background='#333333')
        root_sk.iconbitmap("4.ico")  # 程序的图标
 
        text = tkinter.Text(root_sk, width=140, heigh=60, bg='#333333', undo=True, fg='#ffffff',borderwidth=0)  # 宽度为80个字母(40个汉字)，高度为1个行高
 
        scroll = tkinter.Scrollbar(root_sk)
        # 放到窗口的右侧, 填充Y竖直方向
        scroll.pack(side=tkinter.RIGHT, fill=tkinter.Y)
 
        # 两个控件关联
        scroll.config(command=text.yview)
        text.config(yscrollcommand=scroll.set)
        text.pack()
        try:
            PE = pefile.PE(var_file.get())    # 读取pe文件
            text.insert(tkinter.INSERT, f'\n\n{PE}')
        except:
            PE = '\n\n请检查路径文件或文档是否出错不存在！！！！！！！！！！！！'
            text.insert(tkinter.INSERT, PE)
 
 
 
 
        def mysaveas():  # 另存为
            global filename
            f = asksaveasfilename(initialfile="未命名.txt", defaultextension=".txt")
            filename = f
            fh = open(f, 'w')
            msg = text.get(1.0, END)
            fh.write(msg)
            fh.close()
            root_sk.title("记事本 " + os.path.basename(f))
 
        def move():  # 撤销
            text.edit_undo()
 
        menubar = Menu(root_sk)
        root_sk.config(menu=menubar)
        menubar.add_cascade(label='另存为', command=mysaveas)
        menubar.add_cascade(label='撤销', command=move)
        root_sk.mainloop()
 
 
 
 
    def getfile():   # 使用定位文件
        file_path = filedialog.askopenfilename(filetypes=[('*.EXE', '*.exe'), ('*.dll', '*.DLL')])
        var_file.set(file_path)
        PE()
 
    windnd.hook_dropfiles(theLabel, func=file)  # 背景
    # 按钮控件
    Button(root, text='\n具体代码查看\n',width=18, command=dai_mck).place(x=284, y=172)
    Button(root, text='打开文件',  command=getfile).place(x=610, y=8)
 
 
    def Label():   # 标签
        # 标签
        tkinter.Label(root, bg="#ffffff", text='小木_.').place(x=710, y=14)
    Label()   # 标签
 
 
    root.mainloop() #运行
 
 
 
 
if __name__ == '__main__':
    main()
 
 
 