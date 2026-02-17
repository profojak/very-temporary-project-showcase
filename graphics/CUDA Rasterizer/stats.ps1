<# CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   stats.ps1 ---------------------------------------------------------------- #>


$totalCount = 0
Get-ChildItem -Path "src" -Recurse -File |
    ForEach-Object {
        $lineCount = (Get-Content $_.FullName | Measure-Object -Line).Lines
        $totalCount += $lineCount
        "{0,-15}{1,5}" -f $($_.Name), $lineCount
    }
Write-Host "--------------------"
"{0, 20}" -f $totalCount


<# stats.ps1 #>