# PowerShell script to test deduplication functionality
# This script uploads the same document twice and verifies that duplicates are prevented

param (
    [Parameter(Mandatory=$true)]
    [string]$DocumentPath
)

$ErrorActionPreference = "Stop"

$baseUrl = "http://localhost:8000"
$apiBase = "$baseUrl/api"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Deduplication Test Script" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script will:" -ForegroundColor Yellow
Write-Host "  1. Upload the document (First Upload)" -ForegroundColor Gray
Write-Host "  2. List documents to see what was added" -ForegroundColor Gray
Write-Host "  3. Upload the SAME document again (Second Upload)" -ForegroundColor Gray
Write-Host "  4. Verify that duplicates were prevented" -ForegroundColor Gray
Write-Host ""

if (-not (Test-Path $DocumentPath)) {
    Write-Host "[ERROR] Document file not found: $DocumentPath" -ForegroundColor Red
    exit 1
}

# Function to upload a document
function Upload-Document {
    param (
        [string]$FilePath,
        [string]$UploadLabel
    )
    
    Write-Host "=== $UploadLabel ===" -ForegroundColor Yellow
    Write-Host "Document: $FilePath" -ForegroundColor Gray
    Write-Host ""
    
    try {
        Add-Type -AssemblyName System.Net.Http
        
        $httpClient = New-Object System.Net.Http.HttpClient
        $multipartContent = New-Object System.Net.Http.MultipartFormDataContent
        
        $resolvedPath = Resolve-Path $FilePath -ErrorAction Stop
        $fileStream = [System.IO.File]::OpenRead($resolvedPath.Path)
        $fileName = [System.IO.Path]::GetFileName($resolvedPath.Path)
        $fileContent = New-Object System.Net.Http.StreamContent($fileStream)
        $fileContent.Headers.ContentType = New-Object System.Net.Http.Headers.MediaTypeHeaderValue("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        $multipartContent.Add($fileContent, "file", $fileName)
        
        $projectNameContent = New-Object System.Net.Http.StringContent("Test Project")
        $multipartContent.Add($projectNameContent, "project_name")
        
        $docTypeContent = New-Object System.Net.Http.StringContent("statement of work")
        $multipartContent.Add($docTypeContent, "document_type")
        
        $response = $httpClient.PostAsync("$apiBase/documents/upload", $multipartContent).Result
        $responseContent = $response.Content.ReadAsStringAsync().Result
        
        if ($response.IsSuccessStatusCode) {
            $uploadResponse = $responseContent | ConvertFrom-Json
            Write-Host "[SUCCESS] Document uploaded successfully!" -ForegroundColor Green
            Write-Host "Document ID: $($uploadResponse.document_id)" -ForegroundColor Gray
            Write-Host "Message: $($uploadResponse.message)" -ForegroundColor Gray
            Write-Host ""
            
            # Extract chunk counts from message
            if ($uploadResponse.message -match "(\d+) chunks created") {
                $chunksCreated = [int]$matches[1]
            } else {
                $chunksCreated = 0
            }
            if ($uploadResponse.message -match "(\d+) new chunks added") {
                $chunksAdded = [int]$matches[1]
            } else {
                $chunksAdded = 0
            }
            
            return @{
                DocumentId = $uploadResponse.document_id
                ChunksCreated = $chunksCreated
                ChunksAdded = $chunksAdded
                Message = $uploadResponse.message
            }
        }
        else {
            throw "Upload failed: $responseContent"
        }
        
        $fileStream.Close()
        $httpClient.Dispose()
    }
    catch {
        Write-Host "[ERROR] Failed to upload document: $_" -ForegroundColor Red
        throw
    }
}

# Function to list documents
function Get-Documents {
    Write-Host "=== Listing Documents ===" -ForegroundColor Yellow
    Write-Host ""
    
    try {
        $response = Invoke-RestMethod -Uri "$apiBase/documents" -Method Get
        
        Write-Host "Total documents: $($response.Count)" -ForegroundColor Gray
        Write-Host ""
        
        foreach ($doc in $response) {
            Write-Host "  - $($doc.file_name) (ID: $($doc.document_id))" -ForegroundColor White
            Write-Host "    Project: $($doc.project_name)" -ForegroundColor DarkGray
            Write-Host "    Type: $($doc.document_type)" -ForegroundColor DarkGray
            Write-Host "    Uploaded: $($doc.upload_date)" -ForegroundColor DarkGray
            Write-Host ""
        }
        
        return $response
    }
    catch {
        Write-Host "[ERROR] Failed to list documents: $_" -ForegroundColor Red
        throw
    }
}

# Step 1: First Upload
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STEP 1: First Upload" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$firstUpload = Upload-Document -FilePath $DocumentPath -UploadLabel "First Upload"
$firstDocumentId = $firstUpload.DocumentId
$firstChunksCreated = $firstUpload.ChunksCreated
$firstChunksAdded = $firstUpload.ChunksAdded

Write-Host "First upload results:" -ForegroundColor Gray
Write-Host "  Chunks created: $firstChunksCreated" -ForegroundColor Gray
Write-Host "  Chunks added: $firstChunksAdded" -ForegroundColor Gray
Write-Host ""

# Step 2: List Documents (after first upload)
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STEP 2: List Documents (After First Upload)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$documentsAfterFirst = Get-Documents

# Step 3: Second Upload (same document)
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STEP 3: Second Upload (Same Document)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Uploading the SAME document again to test deduplication..." -ForegroundColor Yellow
Write-Host ""

Start-Sleep -Seconds 2  # Small delay to ensure logs are readable

$secondUpload = Upload-Document -FilePath $DocumentPath -UploadLabel "Second Upload (Duplicate)"
$secondDocumentId = $secondUpload.DocumentId
$secondChunksCreated = $secondUpload.ChunksCreated
$secondChunksAdded = $secondUpload.ChunksAdded

Write-Host "Second upload results:" -ForegroundColor Gray
Write-Host "  Chunks created: $secondChunksCreated" -ForegroundColor Gray
Write-Host "  Chunks added: $secondChunksAdded" -ForegroundColor Gray
Write-Host ""

# Step 4: List Documents (after second upload)
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STEP 4: List Documents (After Second Upload)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$documentsAfterSecond = Get-Documents

# Step 5: Verification
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "STEP 5: Verification" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "First Upload:" -ForegroundColor Yellow
Write-Host "  Document ID: $firstDocumentId" -ForegroundColor Gray
Write-Host "  Chunks created: $firstChunksCreated" -ForegroundColor Gray
Write-Host "  Chunks added: $firstChunksAdded" -ForegroundColor Gray
Write-Host ""

Write-Host "Second Upload:" -ForegroundColor Yellow
Write-Host "  Document ID: $secondDocumentId" -ForegroundColor Gray
Write-Host "  Chunks created: $secondChunksCreated" -ForegroundColor Gray
Write-Host "  Chunks added: $secondChunksAdded" -ForegroundColor Gray
Write-Host ""

Write-Host "Expected Behavior:" -ForegroundColor Yellow
Write-Host "  - First upload: All chunks should be added (chunks_added = chunks_created)" -ForegroundColor Gray
Write-Host "  - Second upload: No chunks should be added (chunks_added = 0)" -ForegroundColor Gray
Write-Host "  - Document IDs should be different (new UUID each time)" -ForegroundColor Gray
Write-Host "  - Document count: Should stay the same (second document has no chunks, so it won't appear in list)" -ForegroundColor Gray
Write-Host "    Note: list_documents only shows documents that have chunks in the vector store." -ForegroundColor DarkGray
Write-Host "    Since the second upload added 0 chunks (duplicates), it has no chunks and won't be listed." -ForegroundColor DarkGray
Write-Host ""

# Verify results
$testPassed = $true
$testFailures = @()

if ($firstChunksAdded -ne $firstChunksCreated) {
    $testPassed = $false
    $testFailures += "First upload: Expected chunks_added ($firstChunksAdded) to equal chunks_created ($firstChunksCreated)"
}

if ($secondChunksAdded -ne 0) {
    $testPassed = $false
    $testFailures += "Second upload: Expected chunks_added to be 0 (duplicates should be skipped), but got $secondChunksAdded"
}

if ($firstDocumentId -eq $secondDocumentId) {
    $testPassed = $false
    $testFailures += "Document IDs should be different, but both are: $firstDocumentId"
}

# Document count should stay the same because second upload added 0 chunks
# (list_documents only shows documents that have chunks in the vector store)
if ($documentsAfterSecond.Count -ne $documentsAfterFirst.Count) {
    $testPassed = $false
    $testFailures += "Expected document count to stay the same ($($documentsAfterFirst.Count)), but got: $($documentsAfterSecond.Count)"
    $testFailures += "  (This is expected: second document has no chunks, so it won't appear in list_documents)"
}

# Print results
if ($testPassed) {
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "[SUCCESS] Deduplication test PASSED!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Summary:" -ForegroundColor Yellow
    Write-Host "  ✓ First upload added $firstChunksAdded chunks (as expected)" -ForegroundColor Green
    Write-Host "  ✓ Second upload added 0 chunks (duplicates prevented by content_hash)" -ForegroundColor Green
    Write-Host "  ✓ Document IDs are different (new UUIDs generated each upload)" -ForegroundColor Green
    Write-Host "  ✓ Document count stayed the same (second document has no chunks, so not listed)" -ForegroundColor Green
    Write-Host ""
    Write-Host "Deduplication is working correctly!" -ForegroundColor Green
    Write-Host "  - Content hash-based deduplication prevented duplicate chunks" -ForegroundColor DarkGray
    Write-Host "  - The second document record was created but has no chunks (all were duplicates)" -ForegroundColor DarkGray
}
else {
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host "[FAILED] Deduplication test FAILED!" -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Failures:" -ForegroundColor Yellow
    foreach ($failure in $testFailures) {
        Write-Host "  ✗ $failure" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "Please check the backend logs for more details." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "1. Check backend logs for deduplication summary messages" -ForegroundColor Gray
Write-Host "2. Look for log entries like:" -ForegroundColor Gray
Write-Host "   'Deduplication summary: X chunks processed, Y added, Z skipped by content_hash'" -ForegroundColor DarkGray
Write-Host "3. Verify that second upload shows chunks skipped by content_hash" -ForegroundColor Gray
Write-Host ""

