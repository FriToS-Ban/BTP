# Places the vector figures (SVG) and the app screenshot into the poster with PowerPoint itself,
# reports any text box whose text does not fit, saves the editable PPTX and exports the print PDF.
#   powershell -ExecutionPolicy Bypass -File reports/poster/finish_poster.ps1
param(
    [string]$Dir = (Split-Path -Parent $MyInvocation.MyCommand.Path),
    [string]$Draft = "poster_draft.pptx",
    [string]$OutName = "BTP_poster_A1",
    [int]$PreviewWidth = 3600
)
$ErrorActionPreference = "Stop"
$k = 72.0 / 25.4
$draftPath = Join-Path $Dir $Draft
$pl = Get-Content (Join-Path $Dir "placements.json") -Raw | ConvertFrom-Json

$ppt = New-Object -ComObject PowerPoint.Application
$pres = $ppt.Presentations.Open($draftPath, 0, 0, 0)
$slide = $pres.Slides.Item(1)
try {
    foreach ($p in $pl) {
        if ($p.kind -eq "svg") { $file = Join-Path $Dir ("figures\" + $p.name + ".svg") }
        else { $file = Join-Path $Dir ("figures\" + $p.name + ".png") }
        if (-not (Test-Path $file)) { Write-Output ("MISSING  " + $p.name); continue }
        try { $slide.Shapes.Item("figure slot " + $p.name).Delete() } catch {}
        $pic = $slide.Shapes.AddPicture($file, 0, -1, $p.x * $k, $p.y * $k)
        $pic.LockAspectRatio = -1
        if ($p.kind -eq "svg") {
            $pic.LockAspectRatio = 0
            $pic.Left = $p.x * $k; $pic.Top = $p.y * $k; $pic.Width = $p.w * $k; $pic.Height = $p.h * $k
        } else {
            $scale = [Math]::Min(($p.w * $k) / $pic.Width, ($p.h * $k) / $pic.Height)
            $pic.Width = $pic.Width * $scale
            $pic.Left = $p.x * $k + ($p.w * $k - $pic.Width) / 2
            $pic.Top = $p.y * $k + ($p.h * $k - $pic.Height) / 2
            $pic.Line.Visible = -1; $pic.Line.ForeColor.RGB = 0xBBBBBB; $pic.Line.Weight = 1.0
        }
        $pic.Name = "figure " + $p.name
        $pic.AlternativeText = $p.name
    }

    # text-fit check: text taller than its box, or running past the page
    $bad = 0
    foreach ($s in $slide.Shapes) {
        if ($s.HasTextFrame -and $s.TextFrame2.HasText) {
            $tf = $s.TextFrame2
            $room = $s.Height - $tf.MarginTop - $tf.MarginBottom
            $need = $tf.TextRange.BoundHeight
            if ($need -gt $room + 1.5) {
                $bad++
                Write-Output ("OVERFLOW {0,-45} needs {1,6:N1} mm, box {2,6:N1} mm" -f $s.Name, ($need / $k), ($room / $k))
            }
        }
        if (($s.Top + $s.Height) -gt ($pres.PageSetup.SlideHeight + 0.5) -or ($s.Left + $s.Width) -gt ($pres.PageSetup.SlideWidth + 0.5)) {
            Write-Output ("OFFPAGE  {0}" -f $s.Name)
        }
    }
    Write-Output ("text boxes that overflow: {0}" -f $bad)

    $outPptx = Join-Path $Dir ($OutName + ".pptx")
    $outPdf = Join-Path $Dir ($OutName + ".pdf")
    $pres.SaveAs($outPptx)
    try {
        $pres.ExportAsFixedFormat($outPdf, 2, 2, 0, 1, 1, 0, $null, 1, "", $true, $true, $true, $true, $false)
    } catch {
        Write-Output ("ExportAsFixedFormat failed ({0}); using SaveAs PDF" -f $_.Exception.Message)
        $pres.SaveCopyAs($outPdf, 32)
    }
    $h = [int]($PreviewWidth * $pres.PageSetup.SlideHeight / $pres.PageSetup.SlideWidth)
    $slide.Export((Join-Path $Dir "preview\poster_preview.png"), "PNG", $PreviewWidth, $h)
    Write-Output "saved $outPptx and $outPdf"
} finally {
    $pres.Close()
    $ppt.Quit()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt)
}
