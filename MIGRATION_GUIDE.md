# Migration Guide for v2.1.0 Updates

This guide covers the migration steps for the major version updates in the requirements.

## 🔄 Major Version Updates

### 1. ChromaDB v1.0.16 (from v0.4.18)

**Breaking Changes:**
- API changes in collection management
- Different client initialization
- Updated query syntax

**Migration Steps:**
```python
# Old ChromaDB v0.4.x
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("md_heading_chunks")

# New ChromaDB v1.0.x
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="md_heading_chunks",
    metadata={"hnsw:space": "cosine"}
)
```

**Required Code Changes:**
- Update collection creation in `main.py`
- Update query syntax in query endpoints
- Update embedding storage format

### 2. MinerU v2.1.0 (from v0.1.0)

**New Features:**
- Improved PDF processing performance
- Better image extraction
- Enhanced markdown output
- New command-line options

**Migration Steps:**
```bash
# Update MinerU with core dependencies
pip uninstall mineru
pip install "mineru[core]==2.1.0"

# Install additional dependencies
pip install huggingface_hub==0.20.3 sentence-transformers==2.2.2 chromadb==1.0.16 pdf2image==1.17.0 PyMuPDF==1.26.3

# Test new version
mineru --version
mineru process test.pdf --output ./output
```

**Required Code Changes:**
- Update MinieuProcessor service to handle new output format
- Test with new MinerU output structure
- Update image path resolution if needed

### 3. FastAPI v0.116.1 (from v0.104.1)

**New Features:**
- Improved performance
- Better error handling
- Enhanced OpenAPI documentation

**Migration Steps:**
- No breaking changes expected
- Test all endpoints after update
- Check OpenAPI documentation

### 4. PyMuPDF v1.26.3 (from v1.23.8)

**Improvements:**
- Better PDF parsing
- Enhanced image extraction
- Bug fixes

**Migration Steps:**
- No breaking changes expected
- Test PDF processing functionality

## 🛠️ Migration Process

### Step 1: Backup Current Data
```bash
# Backup ChromaDB data
cp -r ./chroma_db ./chroma_db_backup

# Backup Minieu output
cp -r ./minieu_output ./minieu_output_backup
```

### Step 2: Update Dependencies
```bash
# Update requirements
pip install -r requirements.txt --upgrade

# Or use the setup script
./scripts/setup.sh
```

### Step 3: Test ChromaDB Migration
```python
# Test script to verify ChromaDB compatibility
import chromadb

try:
    client = chromadb.PersistentClient(path="./chroma_db")
    collections = client.list_collections()
    print(f"Found {len(collections)} collections")
    
    for collection in collections:
        print(f"Collection: {collection.name}")
        print(f"Count: {collection.count()}")
        
except Exception as e:
    print(f"Migration needed: {e}")
```

### Step 4: Update Application Code

#### Update ChromaDB Usage in main.py:
```python
# Old version
md_collection = client.get_or_create_collection("md_heading_chunks")

# New version
md_collection = client.get_or_create_collection(
    name="md_heading_chunks",
    metadata={"hnsw:space": "cosine"}
)
```

#### Update Query Syntax:
```python
# Old version
search_results = md_collection.query(
    query_texts=[request.query],
    n_results=request.max_results,
    include=["metadatas", "documents"]
)

# New version (if needed)
search_results = md_collection.query(
    query_texts=[request.query],
    n_results=request.max_results,
    include=["metadatas", "documents"]
)
```

### Step 5: Test MinerU Integration
```bash
# Test MinerU processing
mineru process test.pdf --output ./test_output

# Check output structure
ls -la ./test_output/
```

### Step 6: Verify All Endpoints
```bash
# Start the application
python -m uvicorn main:app --reload

# Test health endpoint
curl http://localhost:8000/health/

# Test PDF upload
curl -X POST "http://localhost:8000/upload-pdf/" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test.pdf"

# Test query endpoint
curl -X POST "http://localhost:8000/query/" \
     -H "Content-Type: application/json" \
     -d '{"pdf_filename": "test.pdf", "query": "test"}'
```

## 🚨 Potential Issues and Solutions

### ChromaDB Migration Issues

**Issue:** Collection not found after update
```python
# Solution: Recreate collections
client.delete_collection("md_heading_chunks")
collection = client.create_collection(
    name="md_heading_chunks",
    metadata={"hnsw:space": "cosine"}
)
```

**Issue:** Query syntax errors
```python
# Solution: Check ChromaDB documentation for new syntax
# Most queries should work the same way
```

### MinerU Issues

**Issue:** Different output structure
```bash
# Solution: Update MinieuProcessor to handle new structure
# Check the new output format and update path resolution
```

**Issue:** Command-line options changed
```bash
# Solution: Check new MinerU help
mineru --help
mineru process --help
```

### FastAPI Issues

**Issue:** Deprecated imports
```python
# Solution: Update imports if needed
# Most imports should remain the same
```

## ✅ Verification Checklist

- [ ] ChromaDB collections accessible
- [ ] PDF upload and processing works
- [ ] Query endpoints return results
- [ ] Image serving works correctly
- [ ] Rules generation functions properly
- [ ] All debug endpoints accessible
- [ ] Health check passes
- [ ] API documentation loads correctly

## 🔧 Rollback Plan

If issues occur, you can rollback:

```bash
# Restore ChromaDB data
rm -rf ./chroma_db
cp -r ./chroma_db_backup ./chroma_db

# Restore Minieu output
rm -rf ./minieu_output
cp -r ./minieu_output_backup ./minieu_output

# Downgrade packages
pip install fastapi==0.104.1
pip install chromadb==0.4.18
pip install "mineru[core]==0.1.0"
pip install huggingface_hub==0.16.4
pip install sentence-transformers==2.2.2
pip install pdf2image==1.16.3
pip install PyMuPDF==1.23.8
```

## 📞 Support

If you encounter issues during migration:
1. Check the application logs: `tail -f app.log`
2. Test individual components
3. Review ChromaDB and MinerU documentation
4. Create an issue with detailed error information
