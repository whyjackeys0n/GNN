import win32com.client
import time

for i in range(1, 1000000):
    xlApp = win32com.client.Dispatch("Excel.Application")
    xlApp.Visible = 1

    sleep_time = 30

    workBook = xlApp.Workbooks.Open(r"D:\MyTest.xlsx")
    time.sleep(sleep_time)
    workBook.ActiveSheet.Cells(1, 1).Value = "hello"
    time.sleep(sleep_time)
    workBook.Close(SaveChanges=0)
    time.sleep(sleep_time)
    xlApp.Quit()
