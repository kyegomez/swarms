# Bug Report: Swarms Codebase Issues

## Bug 1: Error Handling in Daemon Thread (Critical)

**Location:** `swarms/structs/agent.py` lines 1446-1453

**Description:** The `_handle_run_error` method creates a daemon thread to handle errors, but the exception raised in `__handle_run_error` will not propagate back to the main thread. This causes silent failures where errors are logged but not properly propagated to the caller.

**Type:** Concurrency/Error Handling Bug

**Severity:** Critical - Can lead to silent failures

**Current Code:**
```python
def _handle_run_error(self, error: any):
    process_thread = threading.Thread(
        target=self.__handle_run_error,
        args=(error,),
        daemon=True,
    )
    process_thread.start()
```

**Problem:** 
- The daemon thread will exit when the main thread exits
- The `raise error` at the end of `__handle_run_error` occurs in the daemon thread, not the main thread
- This means exceptions are lost and not properly handled by the calling code

**Fix:** Remove the threading wrapper and call the error handler directly, or use proper exception propagation.

---

## Bug 2: Method Name Typo (Logic Error)

**Location:** `swarms/structs/agent.py` lines 2128 and 2122

**Description:** There are two related typos in the response filtering functionality:
1. The method `apply_reponse_filters` has a typo in the name - it should be `apply_response_filters`
2. The `add_response_filter` method accesses `self.reponse_filters` instead of `self.response_filters`

**Type:** Naming/Logic Error

**Severity:** Medium - Can cause AttributeError when called

**Current Code:**
```python
def add_response_filter(self, filter_word: str) -> None:
    logger.info(f"Adding response filter: {filter_word}")
    self.reponse_filters.append(filter_word)  # TYPO: reponse_filters

def apply_reponse_filters(self, response: str) -> str:  # TYPO: apply_reponse_filters
    """
    Apply the response filters to the response
    """
    logger.info(
        f"Applying response filters to response: {response}"
    )
    for word in self.response_filters:
        response = response.replace(word, "[FILTERED]")
    return response
```

**Problem:** 
- Method name is misspelled: `apply_reponse_filters` instead of `apply_response_filters`
- Attribute access is misspelled: `self.reponse_filters` instead of `self.response_filters`
- The method is called correctly in `filtered_run` method, suggesting these are typos

**Fix:** Fix both typos to use correct spelling.

---

## Bug 3: Document Ingestion Logic Error (Data Loss)

**Location:** `swarms/structs/agent.py` lines 2193-2212

**Description:** The `ingest_docs` method has a logic error where it processes all documents in a loop but only retains the data from the last document. All previous documents are processed but their data is overwritten and lost.

**Type:** Logic Error

**Severity:** High - Causes data loss

**Current Code:**
```python
def ingest_docs(self, docs: List[str], *args, **kwargs):
    """Ingest the docs into the memory

    Args:
        docs (List[str]): Documents of pdfs, text, csvs

    Returns:
        None
    """
    try:
        for doc in docs:
            data = data_to_text(doc)

        return self.short_memory.add(
            role=self.user_name, content=data
        )
    except Exception as error:
        logger.info(f"Error ingesting docs: {error}", "red")
```

**Problem:**
- The `data` variable is overwritten on each iteration
- Only the last document's data is actually added to memory
- All previous documents are processed but their data is lost
- The method should either process documents individually or combine all data

**Fix:** Accumulate all document data or process each document individually.

---

## Impact Assessment

1. **Bug 1 (Critical):** Can cause silent failures in production, making debugging difficult
2. **Bug 2 (Medium):** Will cause AttributeError when the method is called correctly
3. **Bug 3 (High):** Causes data loss when ingesting multiple documents

## Fixes Applied

### Bug 1 Fix - Error Handling
**Status:** ✅ FIXED

Changed the `_handle_run_error` method to call `__handle_run_error` directly instead of using a daemon thread:
```python
def _handle_run_error(self, error: any):
    # Handle error directly instead of using daemon thread
    # to ensure proper exception propagation
    self.__handle_run_error(error)
```

### Bug 2 Fix - Method Name Typos
**Status:** ✅ FIXED

Fixed both typos in the response filtering functionality:
1. Renamed `apply_reponse_filters` to `apply_response_filters`
2. Fixed `self.reponse_filters` to `self.response_filters`

### Bug 3 Fix - Document Ingestion Logic
**Status:** ✅ FIXED

Modified the `ingest_docs` method to process all documents and combine their content:
```python
def ingest_docs(self, docs: List[str], *args, **kwargs):
    try:
        # Process all documents and combine their content
        all_data = []
        for doc in docs:
            data = data_to_text(doc)
            all_data.append(f"Document: {doc}\n{data}")
        
        # Combine all document content
        combined_data = "\n\n".join(all_data)

        return self.short_memory.add(
            role=self.user_name, content=combined_data
        )
    except Exception as error:
        logger.info(f"Error ingesting docs: {error}", "red")
```

## Recommendations

1. ✅ Fixed the error handling to properly propagate exceptions
2. ✅ Corrected the method name typos
3. ✅ Fixed the document ingestion logic to process all documents
4. Add unit tests to prevent similar issues in the future
5. Consider adding linting rules to catch method name typos
6. Consider code review processes to catch similar issues