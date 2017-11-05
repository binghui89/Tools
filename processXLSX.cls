VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "ThisWorkbook"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = True
Dim iFile(1 To 100000) As String
Dim count As Integer

Sub do_process()
    ipath = ThisWorkbook.Path
    ' On Error Resume Next
    ' First of all, convert all xls file in to xlsx file
    count = 0
    zdir ipath
    For i = 1 To count
        If iFile(i) Like "*.xls" And iFile(i) <> ThisWorkbook.FullName Then
            MyFile = iFile(i)
            FilePath = Replace(MyFile, ".xls", ".xlsx")
            If Dir(FilePath, 16) = Empty Then
                Set WBookOther = Workbooks.Open(MyFile)
                Application.ScreenUpdating = False
                ActiveWorkbook.SaveAs Filename:=FilePath, FileFormat:=xlOpenXMLWorkbook, CreateBackup:=False
                WBookOther.Close SaveChanges:=False
                Application.ScreenUpdating = True
            End If
        End If
    Next
    
    ' Now, copy all processed sheets into target xlsx files and remove links to the source spreadsheet
    count = 0
    zdir ipath
    For i = 1 To count
        If iFile(i) Like "*.xlsx" And iFile(i) <> ThisWorkbook.FullName Then
            MyFile = iFile(i)
            copy_sheets_and_replace MyFile
            ' copy_sheets MyFile
        End If
    Next
End Sub

Sub zdir(p)
  Set fs = CreateObject("scripting.filesystemobject")
  For Each f In fs.GetFolder(p).Files
    If f <> ThisWorkbook.FullName Then count = count + 1: iFile(count) = f
  Next
  For Each m In fs.GetFolder(p).SubFolders
      zdir m
  Next
End Sub

Sub copy_sheets_and_replace(MyFile)
    Dim iName As String
    Dim to_be_replace As String
    Dim target As Workbook
    iName = ActiveWorkbook.Name
    to_be_replace = "[" & iName & "]"
    
    ' Copy sheets to target spreadsheet
    Set target = Workbooks.Open(MyFile)
    Sheets(Array("p_Activity_electric", "Actual CF", "p_Capacity_electric")).Copy _
        Before:=target.Sheets("Capacity_electric")
    
    ' Remove links to the source spreadsheet
    target.Sheets("p_Activity_electric").Activate
    Cells.Replace What:=to_be_replace, Replacement:="", LookAt:=xlPart, _
        SearchOrder:=xlByRows, MatchCase:=False, SearchFormat:=False, _
        ReplaceFormat:=False
    target.Sheets("p_Capacity_electric").Activate
    Cells.Replace What:=to_be_replace, Replacement:="", LookAt:=xlPart, _
        SearchOrder:=xlByRows, MatchCase:=False, SearchFormat:=False, _
        ReplaceFormat:=False
    target.Sheets("Actual CF").Activate
    Cells.Replace What:=to_be_replace, Replacement:="", LookAt:=xlPart, _
        SearchOrder:=xlByRows, MatchCase:=False, SearchFormat:=False, _
        ReplaceFormat:=False
    target.Sheets("Activity_electric").Activate
    Cells.Replace What:="-", Replacement:="0", LookAt:=xlPart, _
        SearchOrder:=xlByRows, MatchCase:=False, SearchFormat:=False, _
        ReplaceFormat:=False
    target.Sheets("Capacity_electric").Activate
    Cells.Replace What:="-", Replacement:="0", LookAt:=xlPart, _
        SearchOrder:=xlByRows, MatchCase:=False, SearchFormat:=False, _
        ReplaceFormat:=False
        
    target.Save
    target.Close

End Sub

