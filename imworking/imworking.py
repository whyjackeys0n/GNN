import win32com.client
import time

for i in range(1, 10):
    xlApp = win32com.client.Dispatch("Excel.Application")
    xlApp.Visible = 1

    sleeptime = 1

    workBook = xlApp.Workbooks.Open(r"D:\MyTest.xlsx")
    time.sleep(sleeptime)
    workBook.ActiveSheet.Cells(1, 1).Value = "hello"
    time.sleep(sleeptime)
    workBook.Close(SaveChanges=0)
    time.sleep(sleeptime)
    xlApp.Quit()
