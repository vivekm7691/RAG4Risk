# PowerShell test script for Phase 3 API endpoints
# Converted from curl commands

$baseUrl = "http://localhost:8000"
$apiBase = "$baseUrl/api"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Phase 3 API Endpoints - PowerShell Test Script" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Test 1: Document Upload
Write-Host "=== Test 1: Document Upload ===" -ForegroundColor Yellow
Write-Host "Note: This requires a test document file (test_document.docx)" -ForegroundColor Gray
Write-Host ""

$testFile = "test_document.docx"
if (Test-Path $testFile) {
    try {
        # Use .NET HttpClient for multipart form data (works across all PowerShell versions)
        Add-Type -AssemblyName System.Net.Http
        
        $httpClient = New-Object System.Net.Http.HttpClient
        $multipartContent = New-Object System.Net.Http.MultipartFormDataContent
        
        # Add file - resolve path safely
        $resolvedPath = Resolve-Path $testFile -ErrorAction Stop
        $fileStream = [System.IO.File]::OpenRead($resolvedPath.Path)
        $fileName = [System.IO.Path]::GetFileName($resolvedPath.Path)
        $fileContent = New-Object System.Net.Http.StreamContent($fileStream)
        $fileContent.Headers.ContentType = New-Object System.Net.Http.Headers.MediaTypeHeaderValue("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        $multipartContent.Add($fileContent, "file", $fileName)
        
        # Add project_name
        $projectNameContent = New-Object System.Net.Http.StringContent("Test Project")
        $multipartContent.Add($projectNameContent, "project_name")
        
        # Add document_type
        $docTypeContent = New-Object System.Net.Http.StringContent("statement of work")
        $multipartContent.Add($docTypeContent, "document_type")
        
        # Send request
        $response = $httpClient.PostAsync("$apiBase/documents/upload", $multipartContent).Result
        $responseContent = $response.Content.ReadAsStringAsync().Result
        
        if ($response.IsSuccessStatusCode) {
            $response = $responseContent | ConvertFrom-Json
        }
        else {
            throw "Upload failed: $responseContent"
        }
        
        # Cleanup
        $fileStream.Close()
        $httpClient.Dispose()
        
        Write-Host "[SUCCESS] Document uploaded successfully!" -ForegroundColor Green
        Write-Host "Response:" -ForegroundColor Gray
        $response | ConvertTo-Json -Depth 5
        $documentId = $response.document_id
        Write-Host ""
    }
    catch {
        Write-Host "[ERROR] Failed to upload document: $_" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
    }
}
else {
    Write-Host "[SKIP] Test file '$testFile' not found. Skipping upload test." -ForegroundColor Yellow
    Write-Host "To test upload, create a test .docx file or use an existing document." -ForegroundColor Gray
    Write-Host ""
}

# Test 2: Query Endpoint
Write-Host "=== Test 2: Query Endpoint ===" -ForegroundColor Yellow
Write-Host ""

try {
    $queryBody = @{
        query = "What is the project about?"
        project_name = "Test Project"
    } | ConvertTo-Json
    
    $response = Invoke-RestMethod -Uri "$apiBase/query" `
        -Method Post `
        -Body $queryBody `
        -ContentType "application/json"
    
    Write-Host "[SUCCESS] Query executed successfully!" -ForegroundColor Green
    Write-Host "Response:" -ForegroundColor Gray
    $response | ConvertTo-Json -Depth 5
    Write-Host ""
}
catch {
    Write-Host "[ERROR] Failed to execute query: $_" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
}

# Test 3: Query with Filters (for Excel documents)
Write-Host "=== Test 3: Query with Filters ===" -ForegroundColor Yellow
Write-Host ""

try {
    $queryBody = @{
        query = "What are the high severity risks?"
        project_name = "Test Project"
        top_k = 5
        filters = @{
            severity = "High"
            status = "Open"
        }
    } | ConvertTo-Json -Depth 3
    
    $response = Invoke-RestMethod -Uri "$apiBase/query" `
        -Method Post `
        -Body $queryBody `
        -ContentType "application/json"
    
    Write-Host "[SUCCESS] Query with filters executed successfully!" -ForegroundColor Green
    Write-Host "Response:" -ForegroundColor Gray
    $response | ConvertTo-Json -Depth 5
    Write-Host ""
}
catch {
    Write-Host "[ERROR] Failed to execute query with filters: $_" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
}

# Test 4: Streaming Query Endpoint
Write-Host "=== Test 4: Streaming Query Endpoint ===" -ForegroundColor Yellow
Write-Host "Testing Server-Sent Events (SSE) streaming response..." -ForegroundColor Gray
Write-Host ""

try {
    $queryBody = @{
        query = "What is the project about?"
        project_name = "Test Project"
        top_k = 5
    } | ConvertTo-Json
    
    # Use Invoke-WebRequest for streaming
    $response = Invoke-WebRequest -Uri "$apiBase/query/stream" `
        -Method Post `
        -Body $queryBody `
        -ContentType "application/json" `
        -Headers @{"Accept" = "text/event-stream"} `
        -UseBasicParsing
    
    Write-Host "[SUCCESS] Streaming query initiated!" -ForegroundColor Green
    Write-Host "Status Code: $($response.StatusCode)" -ForegroundColor Gray
    Write-Host "Content-Type: $($response.Headers['Content-Type'])" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Parsing SSE stream..." -ForegroundColor Gray
    Write-Host ""
    
    # Parse SSE format (data: {...}\n\n)
    $streamContent = $response.Content
    $lines = $streamContent -split "`n"
    $chunksReceived = 0
    $sourcesReceived = $false
    $doneReceived = $false
    
    foreach ($line in $lines) {
        if ($line -match "^data:\s*(.+)$") {
            $jsonData = $matches[1].Trim()
            try {
                $data = $jsonData | ConvertFrom-Json
                $msgType = $data.type
                
                switch ($msgType) {
                    "sources" {
                        $sourcesReceived = $true
                        $sourceCount = if ($data.sources) { $data.sources.Count } else { 0 }
                        Write-Host "  [SOURCES] Received $sourceCount source(s)" -ForegroundColor Cyan
                    }
                    "chunk" {
                        $chunksReceived++
                        $chunkText = $data.text
                        if ($chunkText.Length -gt 50) {
                            $chunkText = $chunkText.Substring(0, 50) + "..."
                        }
                        Write-Host "  [CHUNK $chunksReceived] $chunkText" -ForegroundColor Green
                    }
                    "done" {
                        $doneReceived = $true
                        $totalTime = $data.total_time
                        Write-Host "  [DONE] Stream completed in ${totalTime}s" -ForegroundColor Yellow
                    }
                    "error" {
                        Write-Host "  [ERROR] $($data.message)" -ForegroundColor Red
                    }
                }
            }
            catch {
                Write-Host "  [WARN] Failed to parse JSON: $jsonData" -ForegroundColor Yellow
            }
        }
    }
    
    Write-Host ""
    if ($sourcesReceived -and $doneReceived) {
        Write-Host "[SUCCESS] Streaming query completed! ($chunksReceived chunks received)" -ForegroundColor Green
    }
    else {
        Write-Host "[WARN] Stream may be incomplete (sources: $sourcesReceived, done: $doneReceived)" -ForegroundColor Yellow
    }
    Write-Host ""
}
catch {
    Write-Host "[ERROR] Failed to execute streaming query: $_" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
}

# Test 5: List Documents
Write-Host "=== Test 5: List Documents ===" -ForegroundColor Yellow
Write-Host ""

try {
    $response = Invoke-RestMethod -Uri "$apiBase/documents" `
        -Method Get
    
    Write-Host "[SUCCESS] Documents listed successfully!" -ForegroundColor Green
    Write-Host "Response:" -ForegroundColor Gray
    $response | ConvertTo-Json -Depth 5
    Write-Host ""
}
catch {
    Write-Host "[ERROR] Failed to list documents: $_" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Test script completed!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Additional PowerShell commands:" -ForegroundColor Yellow
Write-Host ""
Write-Host "# Document Upload (if you have a file):" -ForegroundColor Gray
Write-Host '$formData = @{' -ForegroundColor White
Write-Host '    file = Get-Item "path/to/your/document.docx"' -ForegroundColor White
Write-Host '    project_name = "Test Project"' -ForegroundColor White
Write-Host '    document_type = "statement of work"' -ForegroundColor White
Write-Host '}' -ForegroundColor White
Write-Host 'Invoke-RestMethod -Uri "http://localhost:8000/api/documents/upload" -Method Post -Form $formData' -ForegroundColor White
Write-Host ""
Write-Host "# Query (non-streaming):" -ForegroundColor Gray
Write-Host '$body = @{ query = "What is the project about?"; project_name = "Test Project" } | ConvertTo-Json' -ForegroundColor White
Write-Host 'Invoke-RestMethod -Uri "http://localhost:8000/api/query" -Method Post -Body $body -ContentType "application/json"' -ForegroundColor White
Write-Host ""
Write-Host "# Query (streaming):" -ForegroundColor Gray
Write-Host '$body = @{ query = "What is the project about?"; project_name = "Test Project" } | ConvertTo-Json' -ForegroundColor White
Write-Host '$response = Invoke-WebRequest -Uri "http://localhost:8000/api/query/stream" -Method Post -Body $body -ContentType "application/json" -Headers @{"Accept"="text/event-stream"}' -ForegroundColor White
Write-Host '$response.Content' -ForegroundColor White
Write-Host ""

