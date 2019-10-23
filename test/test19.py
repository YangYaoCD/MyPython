import os
import time

# star=os.path.getatime("C:\\Users\\yangyao\\Desktop\\新建文件夹\\毕设答辩.pptx")
# star=time.ctime(star)
# # aw=time.strftime(aw)
# print(star)
# end=os.path.getctime("C:\\Users\\yangyao\\Desktop\\新建文件夹\\毕设答辩.pptx")
# end=time.ctime(end)
# print(end)
# ti=os.path.getsize("C:\\Users\\yangyao\\Desktop\\新建文件夹\\毕设答辩.pptx")
# print("{}KB".format(ti/1000))
# os.system("C:\\Program Files (x86)\\Microsoft Office\\root\\Office16\\WINWORD.EXE\C:\\Users\\yangyao\\Desktop\\新建文件夹\\201505060514杨垚.docx")
# os.system("C:\\Windows\\System32\\mspaint.exe C:\\Users\\yangyao\\Desktop\\py1.png")
print(os.getcwd())
print(os.getlogin())
print(os.cpu_count())
os.close()