<# CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   build.ps1 ---------------------------------------------------------------- #>


param(
    [string]$target = "help", # Build target: help, debug, release, clean
    [bool]$verbose = $false   # Use verbose output: $true, $false 
)


# Source configuration
$srcDir = "src"
$binDir = "bin"
$exeName = "CUDA-pipeline"
$srcHost = @(
    "main.c"
)
$srcDevice = @(
    "core/input.c"
    "core/math.cu"
    "core/verbose.c"
    "core/mesh.c"
    "core/pipeline.cu"
    "core/scene.c"
    "platform/memory.c"
    "platform/mutex.c"
    "platform/print.c"
    "platform/thread.c"
    "platform/timer.c"
    "platform/window.c"
    "render/pixel.cu"
    "render/primitive.cu"
    "render/rasterize.cu"
    "render/rop.cu"
    "render/vertex.cu"
)

# Compilation configuration
$nvccDefault = @(
    "--device-c"                     # Compilation to object files
    "--gpu-architecture=native"      # Detect and generate code for system GPU
)
$nvccDebug = @(
    "--optimize=1"              # Specify optimization level for host code
    "--define-macro=_DEBUG"     # Define _DEBUG to compile _DEBUG guarded code
)
$nvccRelease = @(
    "--optimize=3"              # Specify optimization level for host code
    "--dopt=on"                 # Enable device code optimization
    "--Werror=all-warnings"     # Treat warnings as errors
)
$clDefault = "/GA /W3"          # Optimize for Windows application
                                # Set output warning level
$clDebug = "/O1 /D _DEBUG"      # Specify optimization level for host code
                                # Define _DEBUG to compile _DEBUG guarded code
$clRelease = "/O2 /Gw /WX"      # Specify optimization level for host code
                                # Enable whole-program global data optimization
                                # Treat warnings as errors

# Linking configuration
$linkerDefault = @(
    "--gpu-architecture=native" # Detect and generate code for system GPU
)
$libs = "Shlwapi,Winmm,User32,Gdi32" # PathFindFileName: print.c
                                     # timeBeginPeriod, timeEndPeriod: timer.c
                                     # Window functions: window.c
                                     # BitBlt: framebuffer.cu


<# Entry -------------------------------------------------------------------- #>


<# Entry point of build script #>
function Main {
    param(
        [string]$target, # Build target: help, debug, release, clean
        [bool]$verbose   # Use verbose output: $true, $false 
    )

    switch ($target) {
        "help" {
            Write-Help
        }

        "debug" {
            Build
        }

        "release" {
            Destroy-Directory
            Build
        }

        "clean" {
            Destroy-Directory
        }

        default {
            Write-Target -valid $false
        }
    }
} <# Build #>


<# Compilation and linking -------------------------------------------------- #>


<# Compile and link #>
function Build {
    Write-Target -valid $true

    Check-Compiler
    Create-Directory

    # Compilation flags
    $nvccFlags = $nvccDefault
    $clFlags = $clDefault
    if ($target -eq "debug") {
        $nvccFlags += $nvccDebug
        $clFlags += " " + $clDebug
    } elseif ($target -eq "release") {
        $nvccFlags += $nvccRelease
        $clFlags += " " + $clRelease
    }

    # Compilation of C host code with MSVC compiler cl.exe (--x=c flag)
    Write-Compilation $nvccFlags $clFlags "c"

    foreach ($src in $srcHost) {
        $file = Get-Item "$srcDir\$src"
        if (Check-Hash $file -eq $true) {
            continue
        }

        Create-Object $file $nvccFlags $clFlags "c"
        Create-Hash $file
    }

    # Compilation of C++ device code with CUDA compiler nvcc.exe
    Write-Compilation $nvccFlags $clFlags "cu"

    foreach ($src in $srcDevice) {
        $file = Get-Item "$srcDir\$src"
        if (Check-Hash $file -eq $true) {
            continue
        }

        Create-Object $file $nvccFlags $clFlags "cu"
        Create-Hash $file
    }

    # Linking
    Create-Executable
} <# Build #>


<# Create and destroy ------------------------------------------------------- #>


<# Source file to object file #>
function Create-Object {
    param(
        $file,         # Source file
        $nvccFlags,    # Device compiler flags
        $clFlags,      # Host compiler flags
        [string]$code  # Compile host or device code: c, cu
    )

    $obj = "$binDir\obj\$($file.BaseName).obj"

    & nvcc @nvccFlags --x=$code "$srcDir\$src" --output-file=$obj `
        --compiler-options $clFlags
    if ($LASTEXITCODE -ne 0) {
        exit
    }
}


<# Object files to executable #>
function Create-Executable {
    Write-Linking

    $objs = @()
    Get-ChildItem -Path "$binDir\obj" | ForEach-Object {
        $objs += "$binDir\obj\$($_.Name)"
    }

    & nvcc @linkerDefault @objs --output-file="$binDir\$target\$exeName.exe" `
        --library $libs
    if ($LASTEXITCODE -ne 0) {
        exit
    }
} <# Create-Executable #>


<# Save file hash in hash directory #>
function Create-Hash {
    param(
        $file # Source file
    )

    $sha = Get-FileHash -Path $file -Algorithm SHA256
    $sha.Hash | Set-Content -Path "$binDir\hash\$($file.BaseName).hash"
} <# Create-Hash #>


<# Create build directory #>
function Create-Directory {
    New-Item -ItemType Directory -Path "$binDir" `
        -ErrorAction SilentlyContinue | Out-Null
    New-Item -ItemType Directory -Path "$binDir\obj" `
        -ErrorAction SilentlyContinue | Out-Null
    New-Item -ItemType Directory -Path "$binDir\hash" `
        -ErrorAction SilentlyContinue | Out-Null
    New-Item -ItemType Directory -Path "$binDir\$target" `
        -ErrorAction SilentlyContinue | Out-Null
} <# Create-Directory #>


<# Destroy build directory #>
function Destroy-Directory {
    Remove-Item "$binDir" -Recurse -ErrorAction SilentlyContinue
} <# Destroy-Directory #>


<# Check -------------------------------------------------------------------- #>


<# Check for nvcc CUDA compiler on system #>
function Check-Compiler {
    if ((Get-Command nvcc -ErrorAction SilentlyContinue).Name -ne "nvcc.exe") {
        Write-Compiler -valid $false
        exit
    } else {
        Write-Compiler -valid $true
    }
} <# Check-Compiler #>


<# Check for file hash #>
function Check-Hash {
    param(
        $file # Source file
    )

    $sha = Get-FileHash -Path $file -Algorithm SHA256
    $hash = Get-Content "$binDir\hash\$($file.BaseName).hash" `
        -ErrorAction SilentlyContinue

    if ($sha.Hash -eq $hash) {
        Write-Hash -valid $true -file $file.Name
        return $true
    } else {
        Write-Hash -valid $false -file $file.Name
        return $false
    }
} <# Check-Hash #>


<# Write -------------------------------------------------------------------- #>


<# Write help #>
function Write-Help {
    # Usage example
    Write-Host `
        ".\build.ps1 <target> <verbose>" `
        -ForegroundColor Yellow
    Write-Host `
        ".\build.ps1 -target <target> -verbose <verbose>`n" `
        -ForegroundColor Yellow

    # Arguments
    Write-Host `
        "<target> specifies build target: help, debug, release, clean"
    Write-Host `
        "    help: print this help"
    Write-Host `
        "    debug: build with all warnings, enable asserts"
    Write-Host `
        "    release: build with optimizations, treat warnings as errors"
    Write-Host `
        "    clean: remove build directory"
    Write-Host `
        "<verbose> specifies verbose output: `$true or `$false`n"

    # Color coding
    Write-Host `
        "Red: critical error" `
        -ForegroundColor Red
    Write-Host `
        "Yellow: warning, usage example" `
        -ForegroundColor Yellow
    Write-Host `
        "White: info"
    Write-Host `
        "Gray: verbose info" `
        -ForegroundColor DarkGray
} <# Write-Help #>


<# Write info about target #>
function Write-Target {
    param(
        [bool]$valid # Target valid: $true, $false
    )

    if (-not ($valid)) {
        Write-Host `
            "Invalid build target `"$target`"!" `
            "Valid build targets: help, debug, release, clean" `
            -ForegroundColor Red
        Write-Host `
            ".\build.ps1 <target>" `
            -ForegroundColor Yellow

    } elseif ($verbose) {
        Write-Host `
            "Target: $target`n" `
            -ForegroundColor DarkGray
    }
} <# Write-Target #>


<# Write info about compiler #>
function Write-Compiler {
    param(
        [bool]$valid # nvcc CUDA compiler installed on system: $true, $false
    )

    if (-not ($valid)) {
        Write-Host `
            "CUDA compiler not found! Add absolute path to" `
            "`"nvcc.exe`" compiler to `"Path`" environment variable." `
            -ForegroundColor Red

    } elseif ($verbose) {
        Write-Host `
            "$(nvcc --version | findstr /C:"nvcc")" `
            -ForegroundColor DarkGray
        Write-Host `
            "$(nvcc --version | findstr /C:"Build")" `
            -ForegroundColor DarkGray
        Write-Host `
            "$((Get-Command nvcc).Source)" `
            -ForegroundColor DarkGray
    }
} <# Write-Compiler #>


<# Write info about file hash #>
function Write-Hash {
    param(
        [bool]$valid, # File hash exists: $true, $false
        [string]$file # Source file
    )

    if ($valid -and $verbose) {
        Write-Host `
            "$file compiled, skipping..." `
            -ForegroundColor DarkGray
    }
} <# Write-Hash #>


<# Write compilation command #>
function Write-Compilation {
    param(
        [string]$nvccFlags, # Device compiler flags
        [string]$clFlags,   # Host compiler flags
        [string]$code       # Compile host or device code: c, cu
    )

    if ($verbose) {
        if ($code -eq "cu") {
            Write-Host `
                "`nDevice code compilation:" `
                -ForegroundColor DarkGray

        } elseif ($code -eq "c") {
            Write-Host `
                "`nHost code compilation:" `
                -ForegroundColor DarkGray
        }

        Write-Host `
            "nvcc $nvccFlags --x=$code <source> --output-file=<object>" `
            "--compiler-options `"$clFlags`"" `
            -ForegroundColor Yellow
    }
} <# Write-Compilation #>


<# Write linking command #>
function Write-Linking {
    if ($verbose) {
        Write-Host `
            "`nLinking:" `
            -ForegroundColor DarkGray
        Write-Host `
            "nvcc $linkerDefault <objects> --output-file=$exeName.exe" `
            "--library $libs" `
            -ForegroundColor Yellow
    }
} <# Write-Linking #>


<# -------------------------------------------------------------------------- #>


Main $target $verbose


<# build.ps1 #>