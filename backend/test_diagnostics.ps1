# PowerShell test script for Diagnostic API endpoints
# Tests each component of the query pipeline to identify timeout bottlenecks

$baseUrl = "http://localhost:8000"
$apiBase = "$baseUrl/api"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Diagnostic API Endpoints - PowerShell Test Script" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Test 1: Ollama Connectivity
Write-Host "=== Test 1: Ollama Connectivity ===" -ForegroundColor Yellow
Write-Host "Testing Ollama connection and model availability..." -ForegroundColor Gray
Write-Host ""

try {
    $response = Invoke-RestMethod -Uri "$apiBase/diagnostics/ollama" `
        -Method Get
    
    Write-Host "[RESULT] Ollama Diagnostic:" -ForegroundColor Cyan
    Write-Host "  Connection Status: $($response.connection_status)" -ForegroundColor $(if ($response.connection_status -eq "connected") { "Green" } else { "Red" })
    Write-Host "  Model: $($response.model)" -ForegroundColor Gray
    Write-Host "  Base URL: $($response.base_url)" -ForegroundColor Gray
    if ($response.response_time) {
        Write-Host "  Response Time: $([math]::Round($response.response_time, 3))s" -ForegroundColor Gray
    }
    if ($response.error) {
        Write-Host "  Error: $($response.error)" -ForegroundColor Red
    }
    Write-Host ""
}
catch {
    Write-Host "[ERROR] Failed to test Ollama connectivity: $_" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
}

# Test 2: Embedding Generation
Write-Host "=== Test 2: Embedding Generation ===" -ForegroundColor Yellow
Write-Host "Testing embedding generation speed..." -ForegroundColor Gray
Write-Host ""

$testQuery = "What are the main risks in this project?"
try {
    $response = Invoke-RestMethod -Uri "$apiBase/diagnostics/embedding?query=$([System.Web.HttpUtility]::UrlEncode($testQuery))" `
        -Method Get
    
    Write-Host "[RESULT] Embedding Diagnostic:" -ForegroundColor Cyan
    Write-Host "  Query: $($response.query)" -ForegroundColor Gray
    if ($response.success) {
        Write-Host "  Status: [SUCCESS]" -ForegroundColor Green
        Write-Host "  Embedding Time: $([math]::Round($response.embedding_time, 3))s" -ForegroundColor Gray
        Write-Host "  Embedding Dimension: $($response.embedding_dimension)" -ForegroundColor Gray
    }
    else {
        Write-Host "  Status: [FAILED]" -ForegroundColor Red
        Write-Host "  Error: $($response.error)" -ForegroundColor Red
    }
    Write-Host ""
}
catch {
    Write-Host "[ERROR] Failed to test embedding generation: $_" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
}

# Test 3: Vector Search
Write-Host "=== Test 3: Vector Search ===" -ForegroundColor Yellow
Write-Host "Testing vector store search speed..." -ForegroundColor Gray
Write-Host ""

try {
    $response = Invoke-RestMethod -Uri "$apiBase/diagnostics/vector-search?query=$([System.Web.HttpUtility]::UrlEncode($testQuery))&top_k=5" `
        -Method Get
    
    Write-Host "[RESULT] Vector Search Diagnostic:" -ForegroundColor Cyan
    Write-Host "  Query: $($response.query)" -ForegroundColor Gray
    if ($response.success) {
        Write-Host "  Status: [SUCCESS]" -ForegroundColor Green
        Write-Host "  Embedding Time: $([math]::Round($response.embedding_time, 3))s" -ForegroundColor Gray
        Write-Host "  Search Time: $([math]::Round($response.search_time, 3))s" -ForegroundColor Gray
        Write-Host "  Total Time: $([math]::Round($response.total_time, 3))s" -ForegroundColor Gray
        Write-Host "  Results Found: $($response.results_count)" -ForegroundColor Gray
    }
    else {
        Write-Host "  Status: [FAILED]" -ForegroundColor Red
        Write-Host "  Error: $($response.error)" -ForegroundColor Red
    }
    Write-Host ""
}
catch {
    Write-Host "[ERROR] Failed to test vector search: $_" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
}

# Test 4: Full Query Timing
Write-Host "=== Test 4: Full Query Timing Breakdown ===" -ForegroundColor Yellow
Write-Host "Testing complete query pipeline with timing breakdown..." -ForegroundColor Gray
Write-Host ""

$queryBody = @{
    query = $testQuery
    top_k = 5
} | ConvertTo-Json

try {
    Write-Host "Sending query request (this may take a while if LLM is slow)..." -ForegroundColor Gray
    $response = Invoke-RestMethod -Uri "$apiBase/diagnostics/query-timing" `
        -Method Post `
        -Body $queryBody `
        -ContentType "application/json"
    
    Write-Host "[RESULT] Query Timing Breakdown:" -ForegroundColor Cyan
    Write-Host "  Query: $($response.query)" -ForegroundColor Gray
    Write-Host "  Total Time: $([math]::Round($response.total_time, 3))s" -ForegroundColor $(if ($response.total_time -lt 10) { "Green" } elseif ($response.total_time -lt 60) { "Yellow" } else { "Red" })
    Write-Host ""
    Write-Host "  Timing Breakdown:" -ForegroundColor Cyan
    Write-Host "    Embedding Generation: $([math]::Round($response.embedding_time, 3))s" -ForegroundColor Gray
    Write-Host "    Vector Search: $([math]::Round($response.search_time, 3))s" -ForegroundColor Gray
    if ($response.llm_time) {
        $llmColor = if ($response.llm_time -lt 5) { "Green" } elseif ($response.llm_time -lt 30) { "Yellow" } else { "Red" }
        Write-Host "    LLM Response: $([math]::Round($response.llm_time, 3))s" -ForegroundColor $llmColor
    }
    else {
        Write-Host "    LLM Response: [FAILED]" -ForegroundColor Red
    }
    Write-Host ""
    
    if ($response.timings) {
        Write-Host "  Detailed Timings:" -ForegroundColor Cyan
        $response.timings.PSObject.Properties | ForEach-Object {
            Write-Host "    $($_.Name): $([math]::Round($_.Value, 3))s" -ForegroundColor Gray
        }
        Write-Host ""
    }
    
    if ($response.success) {
        Write-Host "  Status: [SUCCESS]" -ForegroundColor Green
    }
    else {
        Write-Host "  Status: [FAILED]" -ForegroundColor Red
        if ($response.error) {
            Write-Host "  Error: $($response.error)" -ForegroundColor Red
        }
    }
    Write-Host ""
}
catch {
    Write-Host "[ERROR] Failed to test query timing: $_" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    if ($_.Exception.Response) {
        try {
            $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
            $responseBody = $reader.ReadToEnd()
            Write-Host "Response Body: $responseBody" -ForegroundColor Red
        }
        catch {
            # Ignore errors reading response
        }
    }
    Write-Host ""
}

# Summary
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Diagnostic Tests Complete" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Interpretation Guide:" -ForegroundColor Yellow
Write-Host "  - Embedding generation should be < 1 second" -ForegroundColor Gray
Write-Host "  - Vector search should be < 2 seconds" -ForegroundColor Gray
Write-Host "  - LLM response time varies by model and hardware:" -ForegroundColor Gray
Write-Host "    * < 5 seconds: Fast" -ForegroundColor Green
Write-Host "    * 5-30 seconds: Acceptable" -ForegroundColor Yellow
Write-Host "    * > 30 seconds: Slow (may timeout)" -ForegroundColor Red
Write-Host ""
Write-Host "If LLM response time is consistently high, consider:" -ForegroundColor Yellow
Write-Host "  - Using a smaller/faster model" -ForegroundColor Gray
Write-Host "  - Increasing OLLAMA_TIMEOUT in configuration" -ForegroundColor Gray
Write-Host "  - Checking Ollama container resources (CPU/memory)" -ForegroundColor Gray
Write-Host ""



