param (
    [scriptblock]$Command
)

# 获取开始时间
$startTime = Get-Date

# 执行命令
& $Command

# 获取结束时间
$endTime = Get-Date

# 计算执行时间
$elapsedTime = $endTime - $startTime

# 将执行时间输出为毫秒
Write-Host "命令执行时间: $($elapsedTime.TotalMilliseconds) 毫秒"
