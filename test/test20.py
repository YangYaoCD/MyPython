import PyPDF2
import matplotlib.image as mping


def add_watermark(wmFile, pageObj):
    # 打开水印pdf文件
    lena=mping.imread(wmFile)
    # 将水印pdf的首页与传入的原始pdf的页进行合并
    pageObj.mergePage(lena)
    lena.close()
    return pageObj
def main():
    # 水印pdf的名称
    watermark = 'C:\\Users\\yangyao\\Desktop\\py1.png'
    # 原始pdf的名称
    origFileName = 'C:\\Users\\yangyao\\Desktop\\14.pdf'
    # 合并后新的pdf名称
    newFileName = 'C:\\Users\\yangyao\\Desktop\\watermark_example.pdf'
    # 打开原始的pdf文件,获取文件指针
    pdfFileObj = open(origFileName, 'rb')
    # 创建reader对象
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    # 创建一个指向新的pdf文件的指针
    pdfWriter = PyPDF2.PdfFileWriter()
    # 通过迭代将水印添加到原始pdf的每一页
    for page in range(pdfReader.numPages):
        wmPageObj = add_watermark(watermark, pdfReader.getPage(page))
        # 将合并后的即添加了水印的page对象添加到pdfWriter
        # pdfWriter.addpage(wmPageObj)
        pdfWriter.addPage(wmPageObj)
    # 打开新的pdf文件
    newFile = open(newFileName, 'wb')
    # 将已经添加完水印的pdfWriter对象写入文件
    pdfWriter.write(newFile)
    # 关闭原始和新的pdf
    pdfFileObj.close()
    newFile.close()
# if __name__ == '__main__':
#     main()