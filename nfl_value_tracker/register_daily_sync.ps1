$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$PSScriptRoot\sync_nflverse.bat`"" -WorkingDirectory "$PSScriptRoot"
$trigger = New-ScheduledTaskTrigger -Daily -At 9:00AM
Register-ScheduledTask -Action $action -Trigger $trigger -TaskName "NFLVerse_Daily_Sync" -Description "Daily sync of NFL rosters and performance metrics via nflverse"
Write-Host "Task registered successfully."
