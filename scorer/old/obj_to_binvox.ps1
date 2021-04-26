param (
    [Parameter(Mandatory=$true,
    HelpMessage="Path to cuda-voxelizer excutable.")]
    [string]
    $Voxelizer,

    [Parameter(Mandatory=$true,
    HelpMessage="Directory to recursively look for OBJ files.")]
    [string]
    $ModelDir,

    [Parameter(Mandatory=$true,
    HelpMessage="Directory to output the result volumes.")]
    [string]
    $OutputDir,

    [Parameter(Mandatory=$false,
    HelpMessage="Target size of each output volume.")]
    [int]
    $Size=128
)

Get-ChildItem -Path $ModelDir -Recurse -Filter "*.obj" |
ForEach-Object -ThrottleLimit 8 -Parallel {
  Start-Process -FilePath $using:Voxelizer -ArgumentList "-f `"$_`" -s $using:Size" -Wait -NoNewWindow
}

Get-ChildItem -Path "$ModelDir\*.binvox" -Recurse | Move-Item -Destination $OutputDir -Force