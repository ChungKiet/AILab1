using XLSX
testResult = [1, 2, 3]
XLSX.openxlsx("Result.xlsx", mode="rw") do xf
    sheet = xf[1]
    for (idx, res) in  enumerate(testResult)
        sheet["B"*string(idx)] = res #row number = B2
    end
end