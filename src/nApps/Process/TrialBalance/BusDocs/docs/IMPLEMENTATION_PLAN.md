 plan # Trial Balance Application - Complete Implementation Plan

**Project:** Enterprise Trial Balance with Petri Net Workflow, AI Commentary, and Data Lineage  
**Duration:** 4 Weeks (20 working days)  
**Start Date:** January 27, 2026  
**Methodology:** Layer-by-Layer (Option A)

---

## 🎯 Project Overview

Building a world-class trial balance system with:
- ✅ **Petri Net Workflow** (13-stage IFRS/DOI compliant process)
- ✅ **High-Performance Calculation Engine** (Zig, 6.8x faster)
- ✅ **ORD/CSN Metadata & Lineage** (Enterprise Data Mesh)
- ✅ **AI-Powered Commentary** (nLocalModels LLM integration)
- ✅ **7 Fully Functional Pages** (SAP Fiori UI5)
- ✅ **Real Hong Kong Sample Data** (Nov 2025)

---

## 📊 Implementation Layers

```
LAYER 1 (Week 1) → Foundation: Data + Calculation + Petri Net Core
LAYER 2 (Week 2) → Business Logic: Rules + Workflow + Metadata
LAYER 3 (Week 3) → APIs: All REST endpoints for all features
LAYER 4 (Week 4) → Frontend: Components + Pages + Testing
```

---

# 🗓️ WEEK 1: FOUNDATION LAYER

## Day 1 (Monday) - Data Models & Structures

### Morning Session (4 hours)
- [ ] **Task 1.1:** Create core data models
  - **File:** `backend/src/models/trial_balance_models.zig`
  - **Contents:**
    - `TrialBalanceEntry` struct
    - `VarianceEntry` struct
    - `AccountMaster` struct
    - `ChecklistItem` struct
    - `DatasetMetadata` struct
  - **Success Criteria:** All structs compile, have proper field types

- [ ] **Task 1.2:** Create data quality enums
  - **File:** `backend/src/models/data_quality.zig`
  - **Contents:**
    - `DataQuality` enum (verified, unverified, suspect)
    - `AccountType` enum (Asset, Liability, Equity, Revenue, Expense)
    - `ChecklistStatus` enum (Pending, InProgress, Complete)
  - **Success Criteria:** Enums defined with proper values

### Afternoon Session (4 hours)
- [ ] **Task 1.3:** Create CSV loader foundation
  - **File:** `backend/src/data/csv_loader.zig`
  - **Functions:**
    - `loadTrialBalanceData()` - Load MTD-TB.csv
    - `loadVarianceData()` - Load PL Variance.csv
    - `loadChecklistData()` - Load Checklist.csv
    - `loadAccountNames()` - Load Names.csv
  - **Success Criteria:** Can parse CSV files, return structs

- [ ] **Task 1.4:** Test with real HKG data
  - **Test File:** `backend/src/data/test_csv_loader.zig`
  - **Test Cases:**
    - Load HKG_PL review Nov'25(MTD-TB).csv
    - Verify record count
    - Validate data types
  - **Success Criteria:** All tests pass, data loads correctly

### End of Day Checkpoint
- ✅ All data models compile
- ✅ CSV loader can read Hong Kong sample data
- ✅ Tests pass

---

## Day 2 (Tuesday) - Data Storage & Retrieval

### Morning Session (4 hours)
- [ ] **Task 2.1:** Create in-memory data store
  - **File:** `backend/src/data/data_store.zig`
  - **Functions:**
    - `DataStore.init()` - Initialize storage
    - `DataStore.loadFromCSV()` - Load all CSV files
    - `DataStore.getTrialBalance()` - Query trial balance
    - `DataStore.getVariances()` - Query variances
  - **Success Criteria:** Can store and retrieve all data types

- [ ] **Task 2.2:** Add filtering capabilities
  - **File:** `backend/src/data/query_builder.zig`
  - **Functions:**
    - `filterByBusinessUnit()` - Filter by BU
    - `filterByPeriod()` - Filter by date range
    - `filterByAccountType()` - Filter by type
  - **Success Criteria:** Filters work correctly

### Afternoon Session (4 hours)
- [ ] **Task 2.3:** Add pagination support
  - **File:** `backend/src/data/pagination.zig`
  - **Functions:**
    - `paginate()` - Return page of results
    - `getTotalCount()` - Get total records
  - **Success Criteria:** Can paginate large datasets

- [ ] **Task 2.4:** Create data aggregation utilities
  - **File:** `backend/src/data/aggregators.zig`
  - **Functions:**
    - `aggregateByAccountType()` - Sum by type
    - `calculateTotals()` - Total debits/credits
    - `groupByPeriod()` - Group by month
  - **Success Criteria:** Aggregations produce correct totals

### End of Day Checkpoint
- ✅ Data store operational with HKG data
- ✅ Can query and filter data
- ✅ Pagination works

---

## Day 3 (Wednesday) - Calculation Engine Integration

### Morning Session (4 hours)
- [ ] **Task 3.1:** Create calculation service wrapper
  - **File:** `backend/src/services/calculation_service.zig`
  - **Functions:**
    - `CalculationService.init()` - Initialize with balance engine
    - `calculateTrialBalance()` - Call balance_engine_unified.zig
    - `verifyBalance()` - Check debits == credits
  - **Success Criteria:** Can call unified engine

- [ ] **Task 3.2:** Integrate balance_engine_unified.zig
  - **File:** Update `backend/build.zig`
  - **Changes:**
    - Add dependency on BusDocs/models/calculation/
    - Link balance_engine_unified.zig
  - **Success Criteria:** Project compiles with engine

### Afternoon Session (4 hours)
- [ ] **Task 3.3:** Create balance calculator
  - **File:** `backend/src/services/balance_calculator.zig`
  - **Functions:**
    - `calculateClosingBalance()` - Opening + Debits - Credits
    - `validateBalance()` - Verify calculations
    - `calculateNetPosition()` - Assets - Liabilities
  - **Success Criteria:** Calculations match Excel formulas

- [ ] **Task 3.4:** Test calculations with HKG data
  - **Test File:** `backend/src/services/test_calculations.zig`
  - **Test Cases:**
    - Load HKG data
    - Calculate trial balance
    - Verify totals match expected
  - **Success Criteria:** All calculations correct

### End of Day Checkpoint
- ✅ Calculation engine integrated
- ✅ Trial balance calculations work
- ✅ Tests pass with real data

---

## Day 4 (Thursday) - Variance Calculations

### Morning Session (4 hours)
- [ ] **Task 4.1:** Create variance calculator
  - **File:** `backend/src/services/variance_calculator.zig`
  - **Functions:**
    - `calculateVariance()` - Current vs Previous
    - `calculateVariancePercent()` - Percentage change
    - `identifySignificantVariances()` - Flag >10%
  - **Success Criteria:** Variance formulas correct

- [ ] **Task 4.2:** Add period-over-period comparison
  - **Functions:**
    - `comparePeriods()` - Month-over-month
    - `compareYTD()` - Year-to-date comparison
    - `calculateTrends()` - 3-month trends
  - **Success Criteria:** Comparisons work correctly

### Afternoon Session (4 hours)
- [ ] **Task 4.3:** Create YTD aggregation
  - **File:** `backend/src/services/ytd_aggregator.zig`
  - **Functions:**
    - `aggregateYTD()` - Sum YTD by account
    - `calculateYTDByType()` - Group by account type
    - `generateYTDReport()` - Format for API
  - **Success Criteria:** YTD totals correct

- [ ] **Task 4.4:** Test variance calculations
  - **Test File:** `backend/src/services/test_variances.zig`
  - **Test Cases:**
    - Compare Nov vs Oct HKG data
    - Verify variance calculations
    - Test threshold detection
  - **Success Criteria:** All variance tests pass

### End of Day Checkpoint
- ✅ Variance calculations working
- ✅ YTD aggregation complete
- ✅ Tests pass

---

## Day 5 (Friday) - Petri Net Foundation

### Morning Session (4 hours)
- [ ] **Task 5.1:** Create Petri Net core structures
  - **File:** `backend/src/workflow/petrinet_engine.zig`
  - **Structures:**
    - `Place` struct (id, tokens, data)
    - `Transition` struct (id, inputs, outputs, guard, action)
    - `PetriNetEngine` struct (places, transitions)
  - **Success Criteria:** Core structures compile

- [ ] **Task 5.2:** Implement token flow
  - **Functions:**
    - `isTransitionEnabled()` - Check if can fire
    - `fireTransition()` - Execute transition
    - `advanceToken()` - Move token through net
  - **Success Criteria:** Token flow logic works

### Afternoon Session (4 hours)
- [ ] **Task 5.3:** Create workflow state tracker
  - **File:** `backend/src/workflow/workflow_state.zig`
  - **Functions:**
    - `WorkflowState.init()` - Initialize state
    - `getCurrentStage()` - Get current position
    - `getCompletedStages()` - List completed
    - `saveState()` - Persist state
  - **Success Criteria:** State tracking works

- [ ] **Task 5.4:** Load workflow from workflow_config.json
  - **Functions:**
    - `loadWorkflowConfig()` - Parse JSON config
    - `createPetriNet()` - Build net from config
    - `initializeMarking()` - Set initial tokens
  - **Success Criteria:** Can load 13-stage workflow

### End of Day Checkpoint
- ✅ Petri Net engine operational
- ✅ Workflow state tracking works
- ✅ Can load 13-stage configuration

### Week 1 Summary
- ✅ **Data Layer:** Complete with HKG sample data
- ✅ **Calculations:** Trial balance + variance calculations working
- ✅ **Petri Net:** Core engine with token flow

---

# 🗓️ WEEK 2: BUSINESS LOGIC LAYER

## Day 6 (Monday) - Business Rules Foundation

### Morning Session (4 hours)
- [ ] **Task 6.1:** Create business rules structures
  - **File:** `backend/src/workflow/business_rules.zig`
  - **Structures:**
    - `Rule` struct (id, name, category, condition, action)
    - `RuleCategory` enum (validation, compliance, threshold, calculation)
    - `RuleViolation` struct (rule_id, severity, message)
  - **Success Criteria:** Rule structures compile

- [ ] **Task 6.2:** Create rules engine
  - **Functions:**
    - `BusinessRulesEngine.init()` - Initialize engine
    - `loadRules()` - Load all rules
    - `evaluateRules()` - Run rules on data
  - **Success Criteria:** Engine can evaluate rules

### Afternoon Session (4 hours)
- [ ] **Task 6.3:** Define IFRS validation rules
  - **Rules to create:**
    - R001: GCOA Mapping Complete
    - R002: Mandatory Fields Present
    - R003: Debit/Credit Indicator Valid
    - R004: Fiscal Period Valid
  - **Success Criteria:** IFRS rules defined

- [ ] **Task 6.4:** Define compliance rules
  - **Rules to create:**
    - C001: Debit = Credit Balance
    - C002: IFRS Classifications Assigned
    - C003: Account Code Range Valid
  - **Success Criteria:** Compliance rules defined

### End of Day Checkpoint
- ✅ Business rules engine operational
- ✅ IFRS + compliance rules defined

---

## Day 7 (Tuesday) - Rule Validators

### Morning Session (4 hours)
- [ ] **Task 7.1:** Create rule validators
  - **File:** `backend/src/workflow/rule_validators.zig`
  - **Validators:**
    - `validateGCOAMapping()` - Check GCOA
    - `validateMandatoryFields()` - Check required fields
    - `checkDebitCreditBalance()` - Verify balance
    - `checkIFRSClassification()` - Verify categories
  - **Success Criteria:** All validators work

- [ ] **Task 7.2:** Create threshold validators
  - **Validators:**
    - `checkBSThreshold()` - $100M or 10%
    - `checkPLThreshold()` - $3M or 10%
    - `checkCommentaryCoverage()` - 90% coverage
  - **Success Criteria:** Threshold detection works

### Afternoon Session (4 hours)
- [ ] **Task 7.3:** Create calculation validators
  - **Validators:**
    - `validateClosingBalance()` - Check formula
    - `validateVarianceCalculation()` - Check variance
    - `validateYTDTotals()` - Check YTD sums
  - **Success Criteria:** Calculation validation works

- [ ] **Task 7.4:** Test rule violations
  - **Test File:** `backend/src/workflow/test_rules.zig`
  - **Test Cases:**
    - Test each validator with good data
    - Test with invalid data
    - Verify violations detected
  - **Success Criteria:** All tests pass

### End of Day Checkpoint
- ✅ All rule validators working
- ✅ Tests pass

---

## Day 8 (Wednesday) - Workflow Integration

### Morning Session (4 hours)
- [ ] **Task 8.1:** Create workflow executor
  - **File:** `backend/src/workflow/workflow_executor.zig`
  - **Functions:**
    - `WorkflowExecutor.init()` - Initialize with Petri Net + Rules + Calc Engine
    - `executeStage()` - Execute one workflow stage
    - `getStageHandler()` - Get handler for stage
  - **Success Criteria:** Can execute stages

- [ ] **Task 8.2:** Create stage handlers structure
  - **File:** `backend/src/workflow/stage_handlers.zig`
  - **Handlers for stages 1-5:**
    - `handleExtract()` - S/4 Data Extraction
    - `handleValidate()` - IFRS Validation
    - `handleFXConversion()` - Currency Conversion
    - `handleAggregate()` - Balance Aggregation
    - `handleCalculateTrialBalance()` - TB Calculation
  - **Success Criteria:** First 5 handlers work

### Afternoon Session (4 hours)
- [ ] **Task 8.3:** Create remaining stage handlers
  - **Handlers for stages 6-13:**
    - `handleIFRSCompliance()` - IFRS Check
    - `handleVarianceAnalysis()` - Variance Calculation
    - `handleThresholdCheck()` - Apply Thresholds
    - `handleCommentaryCollection()` - Collect Commentary
    - `handleMakerReview()` - Maker Approval
    - `handleCheckerReview()` - Checker Approval
    - `handleReportGeneration()` - Generate Reports
    - `handleAuditArchive()` - Archive Audit Trail
  - **Success Criteria:** All 13 handlers implemented

- [ ] **Task 8.4:** Test workflow execution
  - **Test File:** `backend/src/workflow/test_workflow.zig`
  - **Test Cases:**
    - Execute each stage individually
    - Execute full workflow
    - Verify state transitions
  - **Success Criteria:** Full workflow executes

### End of Day Checkpoint
- ✅ Workflow executor operational
- ✅ All 13 stage handlers working
- ✅ Full workflow can execute

---

## Day 9 (Thursday) - Metadata & Lineage (Part 1)

### Morning Session (4 hours)
- [ ] **Task 9.1:** Create lineage tracker
  - **File:** `backend/src/metadata/lineage_tracker.zig`
  - **Structures:**
    - `LineageEntry` struct
    - `TransformationType` enum
  - **Functions:**
    - `LineageTracker.init()`
    - `trackTransformation()` - Record transformation
    - `getLineageChain()` - Get full chain
    - `calculateHash()` - Data hash for verification
  - **Success Criteria:** Can track transformations

- [ ] **Task 9.2:** Integrate lineage with workflow
  - **Changes to:** `backend/src/workflow/workflow_executor.zig`
  - **Updates:**
    - Add lineage_tracker parameter
    - Track each stage transformation
    - Store input/output hashes
  - **Success Criteria:** Lineage captured during workflow

### Afternoon Session (4 hours)
- [ ] **Task 9.3:** Create ORD document structure
  - **File:** `BusDocs/models/ord/trial-balance-product.json`
  - **Contents:**
    - Product definition
    - API resources
    - Data products
    - Output ports
  - **Success Criteria:** Valid ORD v1.9 JSON

- [ ] **Task 9.4:** Create ORD manager
  - **File:** `backend/src/metadata/ord_manager.zig`
  - **Functions:**
    - `loadORDDocument()` - Load ORD JSON
    - `getDataProducts()` - List products
    - `getAPIResources()` - List APIs
  - **Success Criteria:** Can parse ORD documents

### End of Day Checkpoint
- ✅ Lineage tracking integrated
- ✅ ORD document created
- ✅ ORD manager working

---

## Day 10 (Friday) - Metadata & Lineage (Part 2)

### Morning Session (4 hours)
- [ ] **Task 10.1:** Create CSN schema definitions
  - **File:** `BusDocs/models/csn/trial-balance.csn.json`
  - **Entities to define:**
    - TrialBalanceEntry
    - DataLineage
    - VarianceEntry
    - ChecklistItem
  - **Success Criteria:** Valid CSN JSON

- [ ] **Task 10.2:** Create CSN manager
  - **File:** `backend/src/metadata/csn_manager.zig`
  - **Functions:**
    - `loadCSNSchema()` - Load CSN JSON
    - `getEntityDefinition()` - Get entity schema
    - `validateAgainstSchema()` - Validate data
  - **Success Criteria:** Can validate against CSN

### Afternoon Session (4 hours)
- [ ] **Task 10.3:** Create data quality tracker
  - **File:** `backend/src/metadata/data_quality.zig`
  - **Functions:**
    - `calculateQualityScore()` - Score data quality
    - `identifyIssues()` - Find quality problems
    - `generateQualityReport()` - Report quality
  - **Success Criteria:** Quality metrics work

- [ ] **Task 10.4:** Test metadata layer
  - **Test File:** `backend/src/metadata/test_metadata.zig`
  - **Test Cases:**
    - Test lineage chain creation
    - Test ORD document loading
    - Test CSN validation
    - Test quality scoring
  - **Success Criteria:** All metadata tests pass

### End of Day Checkpoint
- ✅ CSN schemas created
- ✅ Data quality tracking works
- ✅ Full metadata layer operational

### Week 2 Summary
- ✅ **Business Rules:** Complete IFRS/DOI/threshold rules
- ✅ **Workflow:** 13-stage execution with handlers
- ✅ **Metadata:** ORD/CSN + lineage tracking

---

# 🗓️ WEEK 3: API LAYER

## Day 11 (Monday) - Core Trial Balance APIs

### Morning Session (4 hours)
- [ ] **Task 11.1:** Create trial balance API
  - **File:** `backend/src/api/trial_balance_api.zig`
  - **Endpoints:**
    - `GET /api/v1/trial-balance/overview` - Dashboard data
    - `GET /api/v1/trial-balance/ytd-analysis` - YTD data
    - `GET /api/v1/trial-balance/raw-data` - Raw entries
  - **Success Criteria:** Endpoints return JSON

- [ ] **Task 11.2:** Add query parameters
  - **Parameters:**
    - `business_unit` - Filter by BU
    - `period` - Filter by period
    - `page` - Pagination
    - `pageSize` - Page size
  - **Success Criteria:** Filtering works

### Afternoon Session (4 hours)
- [ ] **Task 11.3:** Create variance API
  - **Endpoints:**
    - `GET /api/v1/trial-balance/variance` - Variance data
    - `POST /api/v1/trial-balance/calculate` - Trigger calculation
  - **Success Criteria:** Variance endpoint works

- [ ] **Task 11.4:** Update main.zig with routing
  - **File:** `backend/src/main.zig`
  - **Changes:**
    - Add route handler
    - Map URLs to API functions
    - Add CORS headers
  - **Success Criteria:** Server routes correctly

### End of Day Checkpoint
- ✅ Core TB APIs working
- ✅ Can query data via HTTP

---

## Day 12 (Tuesday) - Workflow APIs

### Morning Session (4 hours)
- [ ] **Task 12.1:** Create workflow API
  - **File:** `backend/src/api/workflow_api.zig`
  - **Endpoints:**
    - `GET /api/v1/workflow/state` - Current state
    - `POST /api/v1/workflow/execute/:stage_id` - Execute stage
    - `GET /api/v1/workflow/rules` - List rules
  - **Success Criteria:** Workflow control via API

- [ ] **Task 12.2:** Add workflow validation endpoint
  - **Endpoint:**
    - `POST /api/v1/workflow/validate` - Validate data against rules
  - **Request:** JSON data + rule category
  - **Response:** List of violations
  - **Success Criteria:** Validation works

### Afternoon Session (4 hours)
- [ ] **Task 12.3:** Add checklist API
  - **Endpoints:**
    - `GET /api/v1/trial-balance/checklist` - Get checklist
    - `POST /api/v1/trial-balance/checklist/:id/update` - Update item
  - **Success Criteria:** Checklist operations work

- [ ] **Task 12.4:** Test workflow APIs
  - **Test:** Use curl/Postman
  - **Tests:**
    - Get workflow state
    - Execute a stage
    - Validate data
    - Update checklist
  - **Success Criteria:** All workflow APIs work

### End of Day Checkpoint
- ✅ Workflow APIs complete
- ✅ Can control workflow via HTTP

---

## Day 13 (Wednesday) - AI APIs

### Morning Session (4 hours)
- [ ] **Task 13.1:** Create commentary service
  - **File:** `backend/src/ai/commentary_service.zig`
  - **Functions:**
    - `CommentaryService.init()` - Initialize
    - `generateVarianceCommentary()` - Generate commentary
    - `generateSmartAction()` - Smart action response
    - `callLLM()` - Call nLocalModels
  - **Success Criteria:** Can call nLocalModels API

- [ ] **Task 13.2:** Create smart actions engine
  - **File:** `backend/src/ai/smart_actions.zig`
  - **Functions:**
    - `executeAction()` - Execute smart action
    - `handleOverviewAction()` - Overview page actions
    - `handleYTDAction()` - YTD page actions
    - `generateBulkCommentary()` - Bulk generation
  - **Success Criteria:** Smart actions work

### Afternoon Session (4 hours)
- [ ] **Task 13.3:** Create AI API endpoints
  - **File:** `backend/src/api/ai_api.zig`
  - **Endpoints:**
    - `POST /api/v1/ai/commentary/generate` - Single commentary
    - `POST /api/v1/ai/smart-action` - Execute smart action
    - `POST /api/v1/ai/commentary/bulk` - Bulk generation
  - **Success Criteria:** AI endpoints work

- [ ] **Task 13.4:** Test AI integration
  - **Tests:**
    - Generate commentary for variance
    - Execute smart action
    - Bulk generate commentaries
  - **Success Criteria:** AI responses received

### End of Day Checkpoint
- ✅ AI service integrated
- ✅ Commentary generation works
- ✅ Smart actions operational

---

## Day 14 (Thursday) - Metadata APIs

### Morning Session (4 hours)
- [ ] **Task 14.1:** Create metadata API
  - **File:** `backend/src/api/metadata_api.zig`
  - **Endpoints:**
    - `GET /api/v1/data-products` - List data products
    - `GET /api/v1/data-products/:id/lineage` - Get lineage
    - `GET /ord/trial-balance-product.json` - ORD document
    - `GET /csn/trial-balance.csn.json` - CSN schema
  - **Success Criteria:** Metadata endpoints work

- [ ] **Task 14.2:** Add quality metrics endpoint
  - **Endpoint:**
    - `GET /api/v1/data-products/:id/quality` - Quality metrics
  - **Response:** Quality score + issues
  - **Success Criteria:** Quality API works

### Afternoon Session (4 hours)
- [ ] **Task 14.3:** Create response formatters
  - **File:** `backend/src/api/response_formatters.zig`
  - **Functions:**
    - `formatOverviewResponse()` - Format overview
    - `formatGraphData()` - Format for NetworkGraph
    - `formatTableData()` - Format for tables
    - `formatLineageChain()` - Format lineage
  - **Success Criteria:** Consistent JSON formatting

- [ ] **Task 14.4:** Add error handling
  - **File:** `backend/src/api/error_handler.zig`
  - **Functions:**
    - `handleError()` - Format error response
    - `logError()` - Log to file
  - **Success Criteria:** Errors handled gracefully

### End of Day Checkpoint
- ✅ Metadata APIs complete
- ✅ Error handling in place

---

## Day 15 (Friday) - API Testing & Documentation

### Morning Session (4 hours)
- [ ] **Task 15.1:** Create API test suite
  - **File:** `backend/src/api/test_api_endpoints.zig`
  - **Tests for:**
    - All trial balance endpoints
    - All workflow endpoints
    - All AI endpoints
    - All metadata endpoints
  - **Success Criteria:** All API tests pass

- [ ] **Task 15.2:** Create OpenAPI specification
  - **File:** `docs/07-api-reference/trial-balance-openapi.yaml`
  - **Include:**
    - All endpoints
    - Request/response schemas
    - Authentication
  - **Success Criteria:** Valid OpenAPI 3.0 spec

### Afternoon Session (4 hours)
- [ ] **Task 15.3:** Create API usage examples
  - **File:** `BusDocs/docs/API_EXAMPLES.md`
  - **Examples for:**
    - Querying trial balance
    - Executing workflow
    - Generating AI commentary
    - Getting lineage
  - **Success Criteria:** Examples work

- [ ] **Task 15.4:** Performance testing
  - **Tests:**
    - Load test endpoints
    - Measure response times
    - Test with large datasets
  - **Success Criteria:** <500ms response time

### End of Day Checkpoint
- ✅ All APIs tested
- ✅ Documentation complete
- ✅ Performance acceptable

### Week 3 Summary
- ✅ **Core APIs:** Trial balance + variance
- ✅ **Workflow APIs:** Full workflow control
- ✅ **AI APIs:** Commentary + smart actions
- ✅ **Metadata APIs:** ORD/CSN + lineage

---

# 🗓️ WEEK 4: FRONTEND LAYER

## Day 16 (Monday) - Core Components

### Morning Session (4 hours)
- [ ] **Task 16.0:** Reorganize existing components
  - **Move existing controls to components directory:**
    - Move `webapp/control/NetworkGraphControl.js` → `webapp/components/NetworkGraph/NetworkGraph.js`
    - Move `webapp/control/ProcessFlowControl.js` → `webapp/components/ProcessFlow/ProcessFlow.js`
  - **Update imports in controllers** that reference these components
  - **Success Criteria:** Existing components work in new location

- [ ] **Task 16.1:** Create DataService
  - **File:** `webapp/service/DataService.js`
  - **Methods:**
    - `getOverviewData()` - Call overview API
    - `getYTDAnalysis()` - Call YTD API
    - `getRawData()` - Call raw data API
    - `getVarianceData()` - Call variance API
    - `getChecklist()` - Call checklist API
    - `generateCommentary()` - Call AI API
    - `executeSmartAction()` - Call smart action API
  - **Success Criteria:** All API calls work

- [ ] **Task 16.2:** Create SmartActionButton component
  - **File:** `webapp/components/SmartActionButton/SmartActionButton.js`
  - **SAP Fiori Components:**
    - Extends `sap.m.Button`
    - Uses `sap.m.Dialog` for AI response display
    - Uses `sap.m.BusyIndicator` for loading state (local, inline)
    - Uses `sap.m.MessageBox` for errors
    - Uses `sap.m.MessageStrip` for warnings in dialog
  - **Properties:**
    - `actionId` (string) - Action identifier
    - `contextData` (object) - Context for AI
    - `icon` (string) - Button icon
  - **Accessibility Requirements:**
    - Implement ARIA labels (`aria-label`, `aria-describedby`)
    - Keyboard navigation support (Enter/Space to trigger)
    - Focus management (return focus after dialog closes)
    - Screen reader announcements for AI response
  - **Extension Standards:**
    - Properly call `Button.extend()` with metadata
    - Override `onAfterRendering()` for custom behavior
    - Implement proper event handlers with `attachPress()`
  - **Reference:** https://sapui5.hana.ondemand.com/sdk/#/api/sap.m.Button
  - **Success Criteria:** Button works with full accessibility

### Afternoon Session (4 hours)
- [ ] **Task 16.3:** Create AICommentaryGenerator component
  - **File:** `webapp/components/AICommentaryGenerator/AICommentaryGenerator.js`
  - **SAP Fiori Components:**
    - `sap.m.Button` - Generate/Regenerate buttons
    - `sap.m.Dialog` - Commentary display dialog
    - `sap.m.TextArea` - Editable commentary text
    - `sap.m.VBox` - Layout container
    - `sap.m.BusyDialog` - AI thinking indicator
  - **Features:**
    - Generate commentary from variance data
    - Edit commentary before saving
    - Copy to clipboard functionality
    - Regenerate with different prompt
  - **Reference:** https://sapui5.hana.ondemand.com/sdk/#/api/sap.m.Dialog
  - **Success Criteria:** Generator works

- [ ] **Task 16.4:** Create LineageViewer component
  - **File:** `webapp/components/LineageViewer/LineageViewer.js`
  - **D3.js Visualizations (Multiple View Modes):**
    - **Mode 1: DAG (Directed Acyclic Graph)** - Default lineage view
      - Library: `d3-dag` v0.11+
      - Layout: Sugiyama layout for hierarchical data flow
      - Use case: Show transformation chain (Extract → Transform → Load)
    - **Mode 2: Sankey Diagram** - Data volume flow
      - Library: `d3-sankey` v0.12+
      - Use case: Show data volume through transformation stages
      - Width of links = data volume
    - **Mode 3: NetworkGraph** - Alternative graph view
      - Reuse existing `webapp/components/NetworkGraph/NetworkGraph.js`
      - Use case: Interactive exploration of complex lineages
  - **SAP Fiori Components:**
    - `sap.m.IconTabBar` - Switch between DAG/Sankey/Network views
    - `sap.m.FlexBox` - Layout container
    - `sap.m.Toolbar` - Zoom/pan controls
    - `sap.m.Button` - View mode switcher
  - **D3.js Implementation Details:**
    - **DAG Configuration:**
      - Node size: 80x40px
      - Edge arrows: 8px
      - Vertical spacing: 100px
      - Horizontal spacing: 150px
    - **Sankey Configuration:**
      - Node width: 15px
      - Node padding: 10px
      - Link color: gradient based on data quality
  - **D3.js Fiori Alignment Standards:**
    - **Colors:** Use Fiori semantic palette
      - Success: `#107E3E` (sapUiPositive)
      - Warning: `#E9730C` (sapUiCritical)
      - Error: `#B00` (sapUiNegative)
      - Neutral: `#6A6D70` (sapUiNeutral)
    - **Spacing:** Use Fiori units (0.5rem, 1rem, 2rem)
    - **Typography:** Match Fiori font family (72 font)
    - **Interactions:**
      - Hover: Show tooltips using `sap.m.Popover` with lineage details
      - Click: Fire custom events for data drill-down
      - Zoom: Keyboard shortcuts (+ / - keys) + mouse wheel
      - Pan: Click-drag or arrow keys
  - **Accessibility Requirements:**
    - SVG `<title>` and `<desc>` elements for screen readers
    - Keyboard navigation for nodes (Tab, Arrow keys)
    - Focus indicators matching Fiori styles
    - ARIA live regions for dynamic updates
    - `role="img"` on SVG with proper labels
  - **D3.js Modules Required:**
    - `d3-dag` - DAG layout
    - `d3-sankey` - Sankey diagram
    - `d3-selection` - DOM manipulation
    - `d3-zoom` - Zoom/pan
    - `d3-shape` - Path generators
  - **Reference:** https://github.com/erikbrinkman/d3-dag
  - **Success Criteria:** All 3 view modes work with Fiori compliance

- [ ] **Task 16.5:** Create WorkflowModelBuilder component
  - **File:** `webapp/components/WorkflowModelBuilder/WorkflowModelBuilder.js`
  - **Purpose:** Visual Petri Net editor for customizing workflow
  - **D3.js Implementation:**
    - **Drag-and-Drop Canvas:**
      - Library: `d3-drag` v3+
      - Place nodes: Circles (places) and Rectangles (transitions)
      - Connect with directed edges (arcs)
    - **Visual Elements:**
      - Places: Circles (40px diameter) with token count
      - Transitions: Rectangles (60x30px) with labels
      - Arcs: Curved paths with arrowheads
  - **Features:**
    - Add/remove places and transitions
    - Connect places to transitions (drag to connect)
    - Edit place/transition properties (double-click)
    - Set initial marking (token placement)
    - Validate Petri Net rules (proper connections)
    - Save/load workflow configurations (JSON)
    - Export as workflow_config.json for backend
  - **SAP Fiori Components:**
    - `sap.m.Panel` - Main canvas container
    - `sap.m.Toolbar` - Tools (Add Place, Add Transition, Connect, Delete)
    - `sap.m.Dialog` - Property editor
    - `sap.m.MessageBox` - Validation errors
  - **Integration:**
    - Leverage existing `ProcessFlow` component for display
    - Generate workflow_config.json compatible with backend
    - Visual preview using ProcessFlow
  - **D3.js Modules Required:**
    - `d3-drag` - Drag-and-drop
    - `d3-selection` - Node manipulation
    - `d3-zoom` - Canvas zoom/pan
    - `d3-shape` - Curved arcs
  - **Success Criteria:** Can build and edit Petri Net workflows visually

### End of Day Checkpoint
- ✅ Core components created
- ✅ All components functional

---

## Day 17 (Tuesday) - Update Pages (Part 1)

### Morning Session (4 hours)
- [ ] **Task 17.1:** Update Home controller
  - **File:** `webapp/controller/Home.controller.js` + `view/Home.view.xml`
  - **SAP Fiori Components:**
    - `sap.m.GenericTile` (6 tiles for navigation)
    - `sap.m.TileContent` - Tile content area
    - `sap.m.NumericContent` - Display counts/metrics
    - `sap.m.Page` - Page container
    - `sap.m.FlexBox` - Responsive tile layout
  - **Smart Actions:**
    - SmartActionButton: "💡 Explain this process"
    - SmartActionButton: "🎯 What should I focus on?"
  - **Layout:** 2x3 grid of tiles with header toolbar
  - **Reference:** https://sapui5.hana.ondemand.com/sdk/#/api/sap.m.GenericTile
  - **Success Criteria:** Smart actions on Home page

- [ ] **Task 17.2:** Update Overview controller
  - **File:** `webapp/controller/Overview.controller.js` + `view/Overview.view.xml`
  - **SAP Fiori Components:**
    - `sap.m.ObjectHeader` - Page header with title
    - `sap.suite.ui.commons.NumericTile` (4 tiles: Debits, Credits, Balance, Accounts)
    - `sap.m.Panel` - Group related content
    - `sap.m.Table` - Summary table
    - `sap.m.OverflowToolbar` - Action toolbar
    - `sap.m.ColumnListItem` - Table rows
  - **Smart Actions:**
    - "💡 Explain these metrics"
    - "⚠️ Diagnose balance issues"
    - "🔮 Predict next month"
  - **Layout:** Header + 4 metric tiles + summary table
  - **Reference:** https://sapui5.hana.ondemand.com/sdk/#/api/sap.suite.ui.commons.NumericTile
  - **Success Criteria:** Overview shows real data + AI

### Afternoon Session (4 hours)
- [ ] **Task 17.3:** Update YTD Analysis controller
  - **File:** `webapp/controller/YTDAnalysis.controller.js` + `view/YTDAnalysis.view.xml`
  - **SAP Fiori Components:**
    - `sap.ui.table.AnalyticalTable` - High-performance table with grouping
    - `sap.m.IconTabBar` - Tabs (Table, Force-Directed Graph, Sankey Flow)
    - Custom `NetworkGraph` component - From `webapp/components/NetworkGraph/`
    - `sap.m.SearchField` - Filter/search
    - `sap.m.MultiComboBox` - Multi-select filters
    - `sap.m.OverflowToolbar` - Toolbar with actions
    - `sap.m.Button` - "✨ Generate All Commentary" (bulk)
    - `sap.m.BusyIndicator` - Show while table loads
    - `sap.m.MessageToast` - Success notifications
  - **D3.js Visualizations (IconTabBar):**
    - **Tab 1: Table View** - AnalyticalTable (standard)
    - **Tab 2: Network Graph** - Force-Directed Graph
      - Library: `d3-force` v3+
      - Use existing NetworkGraph component
      - Show account relationships (Assets → Liabilities → Equity)
      - Node size = account balance magnitude
      - Edge thickness = relationship strength
    - **Tab 3: Sankey Flow** - Revenue/Expense Flow
      - Library: `d3-sankey` v0.12+
      - Show: Revenue → Operating Expenses → Net Income
      - Link width = monetary amount
      - Color coding by account type (Revenue=green, Expense=red)
  - **D3.js Configuration:**
    - **Force-Directed:**
      - Force strength: -300 (repulsion)
      - Link distance: 100px
      - Charge: -500
      - Center force: enabled
    - **Sankey:**
      - Node padding: 20px
      - Node width: 20px
      - Alignment: justify
  - **Smart Actions:**
    - "📈 Analyze trends"
    - "🔍 Find anomalies"
    - "💬 Generate bulk commentary"
  - **Layout:** IconTabBar with 3 views (Table, Force Graph, Sankey)
  - **Reference:** https://sapui5.hana.ondemand.com/sdk/#/api/sap.ui.table.AnalyticalTable
  - **Success Criteria:** All 3 visualization modes functional

- [ ] **Task 17.4:** Update Raw Data controller
  - **File:** `webapp/controller/RawData.controller.js` + `view/RawData.view.xml`
  - **SAP Fiori Components:**
    - `sap.ui.table.Table` - High-performance table for large datasets
    - `sap.m.SearchField` - Search functionality
    - `sap.m.MultiComboBox` - Column filters
    - `sap.m.Toolbar` - Filter toolbar
    - `sap.ui.table.Column` - Table columns
    - `sap.m.Text` - Cell content
    - Pagination controls (built into table)
  - **Smart Actions:**
    - "🔎 Explain this entry"
    - "🏷️ Suggest reclassification"
  - **Layout:** Toolbar + Table with virtual scrolling
  - **Reference:** https://sapui5.hana.ondemand.com/sdk/#/api/sap.ui.table.Table
  - **Success Criteria:** Raw data page functional

### End of Day Checkpoint
- ✅ 4 pages updated with real data
- ✅ Smart actions working

---

## Day 18 (Wednesday) - Update Pages (Part 2)

### Morning Session (4 hours)
- [ ] **Task 18.1:** Update BS Variance controller
  - **File:** `webapp/controller/BSVariance.controller.js` + `view/BSVariance.view.xml`
  - **SAP Fiori Components:**
    - `sap.ui.table.TreeTable` - Hierarchical variance display
    - `sap.m.ObjectStatus` - Status indicators (positive/negative variance)
    - `sap.m.Text` - Variance amounts with conditional formatting
    - Custom `AICommentaryGenerator` - Per-row commentary
    - `sap.m.OverflowToolbar` - Action toolbar
    - `sap.m.Button` - Commentary generation button per row
    - `sap.m.MessageStrip` - Show threshold warnings at top
    - `sap.m.BusyIndicator` - Show while calculating variances
  - **Conditional Formatting:**
    - Red (`Error`) for negative variances > threshold
    - Green (`Success`) for positive variances > threshold
    - Gray (`None`) for within threshold
  - **Smart Actions:**
    - "🎯 Highlight material items"
    - "📝 Draft management summary"
    - "✨ Generate commentary" (per row)
  - **Layout:** TreeTable with hierarchical account structure
  - **Reference:** https://sapui5.hana.ondemand.com/sdk/#/api/sap.ui.table.TreeTable
  - **Success Criteria:** Variance page with AI commentary

- [ ] **Task 18.2:** Update Checklist controller
  - **File:** `webapp/controller/Checklist.controller.js` + `view/Checklist.view.xml`
  - **SAP Fiori Components:**
    - Custom `ProcessFlow` component - From `webapp/components/ProcessFlow/`
    - Custom `WorkflowModelBuilder` - From `webapp/components/WorkflowModelBuilder/`
    - `sap.m.List` - Checklist items
    - `sap.m.CustomListItem` - Custom item template
    - `sap.m.CheckBox` - Completion checkboxes
    - `sap.m.Button` - Action buttons + "Edit Workflow" button
    - `sap.m.ObjectStatus` - Item status (Pending, InProgress, Complete)
    - `sap.m.Label` - Item descriptions
    - `sap.m.IconTabBar` - Tabs (Workflow View, Edit Workflow)
  - **Two Modes:**
    - **View Mode:** ProcessFlow showing 13-stage workflow execution
    - **Edit Mode:** WorkflowModelBuilder for customizing workflow
  - **Smart Actions:**
    - "✅ Review checklist completeness"
    - "💡 Suggest next steps"
    - "⚙️ Customize workflow"
  - **Layout:** IconTabBar with ProcessFlow (view) + WorkflowModelBuilder (edit)
  - **Reference:** https://sapui5.hana.ondemand.com/sdk/#/api/sap.m.List
  - **Success Criteria:** Can view AND edit workflow

### Afternoon Session (4 hours)
- [ ] **Task 18.3:** Update Metadata controller
  - **File:** `webapp/controller/Metadata.controller.js` + `view/Metadata.view.xml`
  - **SAP Fiori Components:**
    - `sap.m.IconTabBar` - Tabs (ORD, CSN, Lineage, Quality)
    - Custom `LineageViewer` component - From `webapp/components/LineageViewer/`
    - `sap.m.ObjectAttribute` - Metadata key-value pairs
    - `sap.m.MessageStrip` - Quality warnings/info (multiple types)
    - `sap.suite.ui.commons.MicroProcessFlow` - Mini process flow
    - `sap.m.Panel` - Section grouping
    - `sap.m.VBox` - Vertical layout
    - `sap.ui.layout.form.SimpleForm` - ORD/CSN display
    - `sap.m.MessagePopover` - Aggregate quality issues
  - **MessageStrip Usage:**
    - Information (blue) for lineage explanations
    - Warning (orange) for minor quality issues
    - Error (red) for critical quality issues
    - Success (green) for validation passes
  - **Tabs:**
    - **ORD Tab:** Display data product definitions
    - **CSN Tab:** Display schema definitions
    - **Lineage Tab:** LineageViewer with 3 sub-modes
      - DAG view (default) - Transformation chain
      - Sankey view - Data volume flows
      - Network view - Alternative graph
    - **Quality Tab:** Quality metrics + MessagePopover for issues
  - **D3.js in Lineage Tab:**
    - Use LineageViewer component's DAG/Sankey/Network modes
    - Show transformation stages with data hashes
    - Interactive exploration of lineage chain
  - **Smart Actions:**
    - "🔗 Explain data lineage"
    - "🛡️ Check data quality"
    - "📊 Switch visualization mode"
  - **Layout:** IconTabBar with 4 specialized views
  - **Reference:** https://sapui5.hana.ondemand.com/sdk/#/api/sap.m.IconTabBar
  - **Success Criteria:** Metadata page with all visualization modes

- [ ] **Task 18.4:** Update all view XMLs with consistent patterns
  - **Files:** All 7 view XML files
  - **SAP Fiori Patterns Applied:**
    - **Page Structure:** `sap.m.Page` with `customHeader` and `content`
    - **Toolbars:** `sap.m.OverflowToolbar` for responsive actions
    - **Responsive Layout:** `sap.ui.layout.Grid` or `sap.m.FlexBox`
    - **Common Components:**
      - `sap.m.Title` for page titles
      - `sap.m.Toolbar` for action bars
      - `sap.m.ToolbarSpacer` for spacing
      - `sap.m.Button` with proper icons
      - Custom `SmartActionButton` for AI features
  - **Consistent Styling:**
    - Use `sapUiSizeCompact` CSS class
    - Standard SAP Fiori spacing (0.5rem, 1rem, 2rem)
    - Consistent icon usage (sap-icon://)
  - **Reference:** https://experience.sap.com/fiori-design-web/
  - **Success Criteria:** All views updated with Fiori guidelines

### End of Day Checkpoint
- ✅ All 7 pages updated
- ✅ All smart actions integrated

---

## Day 19 (Thursday) - Integration Testing

### Morning Session (4 hours)
- [ ] **Task 19.1:** End-to-end testing
  - **Tests:**
    - Start backend server
    - Start frontend server
    - Navigate through all pages
    - Test all smart actions
    - Test AI commentary generation
    - Test workflow execution
  - **Success Criteria:** Everything works together

- [ ] **Task 19.2:** Error scenario testing
  - **Tests:**
    - Test with backend offline
    - Test with invalid data
    - Test with AI service offline
    - Test with large datasets
  - **Success Criteria:** Graceful error handling

### Afternoon Session (4 hours)
- [ ] **Task 19.3:** Performance testing
  - **Tests:**
    - Page load times (<2s)
    - API response times (<500ms)
    - Large dataset handling (10,000+ rows)
    - AI response times (<3s)
  - **Success Criteria:** All metrics within targets

- [ ] **Task 19.4:** Theme density & cross-browser testing
  - **Theme Density Tests:**
    - **Compact Mode** (`sapUiSizeCompact`):
      - Test all pages with 16px touch targets
      - Verify custom components render correctly
      - Check table row heights
      - Validate toolbar spacing
    - **Cozy Mode** (default):
      - Test all pages with 48px touch targets
      - Verify touch-friendly interactions
      - Check mobile responsiveness
    - **Responsive Breakpoints:**
      - S (<600px): Mobile phone layout
      - M (600-1024px): Tablet layout
      - L (1024-1440px): Desktop layout
      - XL (>1440px): Large desktop layout
  - **Cross-Browser Tests:**
    - Chrome (latest): Full functionality
    - Safari (latest): Full functionality
    - Firefox (latest): Full functionality
    - Edge (latest): Full functionality
  - **Accessibility Tests:**
    - Screen reader navigation (VoiceOver/NVDA)
    - Keyboard-only navigation
    - Color contrast validation (WCAG 2.1 AA)
    - Focus indicators visible
  - **Success Criteria:** Works perfectly in all modes and browsers
---

# 📊 PROJECT COMPLETION CHECKLIST

## Core Features
- [ ] Petri Net workflow engine (13 stages)
- [ ] HANA data schema
- [ ] Calculation engine (5 versions)
- [ ] ORD/CSN metadata + lineage
- [ ] AI commentary generation
- [ ] Smart actions (10+ actions)
- [ ] Complete backend APIs
- [ ] Complete frontend (7 pages)
- [ ] Real data integration (HKG Nov 2025)

## Technical Requirements
- [ ] All tests pass
- [ ] Performance: <2s page loads, <500ms API
- [ ] Security: Proper error handling
- [ ] Documentation: Complete
- [ ] Code quality: Clean, commented

## Business Requirements
- [ ] IFRS compliant
- [ ] DOI v2.0 compliant
- [ ] 90% variance coverage
- [ ] M+7 reporting deadline support
- [ ] Audit trail complete

---

# 📈 Success Metrics

| Metric | Target | Status |
|--------|--------|--------|
| **Calculation Performance** | 6.8x faster than baseline | ⏳ TBD |
| **API Response Time** | <500ms | ⏳ TBD |
| **Page Load Time** | <2s | ⏳ TBD |
| **Test Coverage** | >80% | ⏳ TBD |
| **AI Response Time** | <3s | ⏳ TBD |
| **Workflow Stages** | 13/13 operational | ⏳ TBD |
| **Pages Complete** | 7/7 functional | ⏳ TBD |
| **Smart Actions** | 10+ implemented | ⏳ TBD |

---

# 🎯 Daily Standup Questions

Use these questions each day:

1. **What did I complete yesterday?**
2. **What will I complete today?**
3. **Any blockers?**
4. **Any dependencies?**
5. **Any scope changes needed?**

---

# 📝 Change Log

| Date | Change | Reason |
|------|--------|--------|
| 2026-01-27 | Initial plan created | Project kickoff |
| | | |
| | | |

---

# 📊 D3.JS VISUALIZATION SPECIFICATIONS

## Complete D3.js Stack

| Component | D3.js Chart Type | Library | Use Case | Day Built |
|-----------|-----------------|---------|----------|-----------|
| **NetworkGraph** | Force-Directed Graph | d3-force v3+ | YTD account relationships | 16 (move) |
| **ProcessFlow** | Custom Process Flow | sap.suite ProcessFlow + D3 | 13-stage workflow display | 16 (move) |
| **LineageViewer** | DAG + Sankey + Network | d3-dag, d3-sankey | Data lineage 3-mode view | 16 |
| **WorkflowModelBuilder** | Drag-Drop Petri Net | d3-drag, d3-selection | Visual workflow editor | 16 |
| **YTD Sankey** | Sankey Flow Diagram | d3-sankey v0.12+ | Revenue/expense flows | 17 |

## D3.js Modules Required (Day 16 - Install)

```json
{
  "dependencies": {
    "d3": "^7.8.5",
    "d3-dag": "^0.11.5",
    "d3-sankey": "^0.12.3"
  }
}
```

**Core Modules Used:**
- `d3-selection` - DOM manipulation
- `d3-force` - Force-directed layouts
- `d3-drag` - Drag-and-drop interactions
- `d3-zoom` - Zoom/pan controls
- `d3-shape` - Path generators (curves, arcs)
- `d3-scale` - Color scales
- `d3-dag` - Directed acyclic graph layouts
- `d3-sankey` - Sankey diagram generator

## Visualization Details by Component

### 1. NetworkGraph (Force-Directed)
**Location:** `webapp/components/NetworkGraph/NetworkGraph.js`  
**Chart Type:** Force-Directed Graph  
**Use Cases:**
- YTD Analysis: Show account relationships
- Metadata Lineage: Alternative graph view

**Configuration:**
```javascript
{
  force: {
    charge: -500,           // Repulsion strength
    linkDistance: 100,      // Distance between nodes
    centerStrength: 0.1,    // Center force
    collisionRadius: 50     // Collision detection
  },
  nodes: {
    minRadius: 20,
    maxRadius: 60,
    sizeBy: "balance"       // Node size = account balance
  },
  links: {
    minWidth: 1,
    maxWidth: 10,
    widthBy: "strength"     // Link thickness = relationship
  }
}
```

**Data Format:**
```javascript
{
  nodes: [
    { id: "1000", label: "Cash", type: "Asset", balance: 1000000 }
  ],
  links: [
    { source: "1000", target: "2000", strength: 0.8 }
  ]
}
```

### 2. ProcessFlow (Workflow Stages)
**Location:** `webapp/components/ProcessFlow/ProcessFlow.js`  
**Chart Type:** Custom Process Flow (SAP + D3 hybrid)  
**Use Cases:**
- Checklist page: Display 13-stage workflow execution
- Metadata page: Mini process flow

**Features:**
- 13 workflow stages with status colors
- Current stage highlighting
- Completed stages marked green
- Failed stages marked red
- Lane-based layout (Maker/Checker lanes)

### 3. LineageViewer (DAG + Sankey + Network)
**Location:** `webapp/components/LineageViewer/LineageViewer.js`  
**Chart Types:** 3-mode visualization  

#### **Mode 1: DAG (Default)**
**Library:** d3-dag v0.11+  
**Layout:** Sugiyama (layered graph)  
**Use Case:** Show transformation chain

**Configuration:**
```javascript
{
  dag: {
    layering: "simplex",    // Layering algorithm
    decross: "opt",         // Minimize crossings
    coord: "center",        // Coordinate assignment
    nodeSize: [80, 40],     // Node dimensions
    layerGap: 100,          // Vertical spacing
    nodeGap: 150            // Horizontal spacing
  }
}
```

**Visual:**
```
[Extract] → [Validate] → [Transform] → [Aggregate] → [Calculate]
```

#### **Mode 2: Sankey**
**Library:** d3-sankey v0.12+  
**Use Case:** Show data volume flows through transformations

**Configuration:**
```javascript
{
  sankey: {
    nodeWidth: 15,
    nodePadding: 10,
    align: "justify",       // Node alignment
    iterations: 32          // Layout optimization
  },
  linkColor: "gradient"     // Source → Target gradient
}
```

**Visual:**
```
    1000 records
Raw Data ═══════════> Validated ════════> Final TB
    (100%)            (980 records)       (980 records)
                         (98%)              (98%)
```

#### **Mode 3: Network Graph**
**Reuses:** Existing NetworkGraph component  
**Use Case:** Interactive exploration of complex lineages

### 4. WorkflowModelBuilder (Petri Net Editor)
**Location:** `webapp/components/WorkflowModelBuilder/WorkflowModelBuilder.js`  
**Chart Type:** Interactive Petri Net Canvas  
**Use Case:** Visual workflow design/editing

**Features:**
- **Canvas:** 1200x800px SVG
- **Places:** Circles (40px ⌀) with token badges
- **Transitions:** Rectangles (60x30px) with labels
- **Arcs:** Curved bezier paths with arrows
- **Tools:** Add/Delete/Connect/Edit

**Petri Net Rules Enforced:**
- Places only connect to transitions
- Transitions only connect to places
- No self-loops
- Valid initial marking (token placement)

**Export Format:** workflow_config.json
```javascript
{
  places: [
    { id: "p1", name: "Start", tokens: 1 }
  ],
  transitions: [
    { id: "t1", name: "Extract", guard: "hasData" }
  ],
  arcs: [
    { from: "p1", to: "t1", weight: 1 }
  ]
}
```

### 5. YTD Sankey (Revenue/Expense Flow)
**Location:** Integrated in YTD Analysis page  
**Chart Type:** Sankey Flow Diagram  
**Use Case:** Show P&L flow from revenue to net income

**Configuration:**
```javascript
{
  flow: {
    source: "Revenue",
    intermediates: ["COGS", "Operating Expenses", "Interest", "Tax"],
    target: "Net Income"
  },
  linkColors: {
    revenue: "#107E3E",     // Green (positive)
    expense: "#B00",        // Red (negative)
    netIncome: "#6A6D70"    // Neutral
  }
}
```

**Visual:**
```
Revenue (10M) ════════════════════════════════╗
                                              ║
  ╠═══> COGS (6M) ═══════════════════════════╣
  ║                                           ║
  ╠═══> Op. Expenses (2M) ════════════════════╣═══> Net Income (1.5M)
  ║                                           ║
  ╚═══> Interest + Tax (0.5M) ════════════════╝
```

## D3.js + Fiori Color Mapping

All D3.js visualizations use Fiori semantic colors:

| Semantic Meaning | Fiori Color | Hex Code | D3.js Usage |
|-----------------|-------------|----------|-------------|
| Success/Positive | sapUiPositive | #107E3E | Completed stages, positive variances |
| Warning | sapUiCritical | #E9730C | In-progress, approaching threshold |
| Error/Negative | sapUiNegative | #B00 | Failed stages, negative variances |
| Neutral/Info | sapUiNeutral | #6A6D70 | Pending stages, neutral data |
| Primary | sapUiAccent1 | #0A6ED1 | Primary actions, highlights |

## Installation & Setup (Day 16, Task 16.0)

```bash
# Install D3.js v7 and specialized modules
npm install d3@^7.8.5 d3-dag@^0.11.5 d3-sankey@^0.12.3 --save

# Verify installation
npm list d3 d3-dag d3-sankey
```

---

# 🎨 SAP FIORI UI5 COMPONENT REFERENCE

## Component Matrix by Page

| Page | Primary Components | Secondary Components | Custom Components |
|------|-------------------|---------------------|------------------|
| **Home** | GenericTile, TileContent | NumericContent, FlexBox | SmartActionButton |
| **Overview** | NumericTile, ObjectHeader | Panel, Table, OverflowToolbar | SmartActionButton |
| **YTD Analysis** | AnalyticalTable, IconTabBar | SearchField, MultiComboBox | NetworkGraphControl, SmartActionButton |
| **Raw Data** | sap.ui.table.Table | SearchField, Toolbar | SmartActionButton |
| **BS Variance** | TreeTable | ObjectStatus, Text | AICommentaryGenerator, SmartActionButton |
| **Checklist** | List, CustomListItem | CheckBox, ObjectStatus | ProcessFlowControl, SmartActionButton |
| **Metadata** | IconTabBar, SimpleForm | ObjectAttribute, MessageStrip | LineageViewerControl, SmartActionButton |

## Detailed Component Specifications

### Navigation Components
- **sap.m.GenericTile** - Home page navigation tiles
  - Properties: header, subheader, press event
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.m.GenericTile

### Display Components
- **sap.suite.ui.commons.NumericTile** - KPI metrics display
  - Properties: value, scale, unit, state (Good/Error/Warning)
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.suite.ui.commons.NumericTile

- **sap.m.ObjectHeader** - Page headers with metadata
  - Properties: title, number, numberUnit, attributes
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.m.ObjectHeader

- **sap.m.ObjectStatus** - Status indicators
  - Properties: text, state (Success/Warning/Error)
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.m.ObjectStatus

### Table Components
- **sap.ui.table.Table** - High-performance table (1M+ rows)
  - Features: Virtual scrolling, column resizing, sorting
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.ui.table.Table

- **sap.ui.table.AnalyticalTable** - Table with grouping/aggregation
  - Features: Hierarchical grouping, sum rows, drill-down
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.ui.table.AnalyticalTable

- **sap.ui.table.TreeTable** - Hierarchical tree table
  - Features: Expand/collapse, tree structure
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.ui.table.TreeTable

- **sap.m.Table** - Responsive table (smaller datasets)
  - Features: Responsive columns, swipe actions
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.m.Table

### Input Components
- **sap.m.SearchField** - Search/filter input
  - Events: search, liveChange
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.m.SearchField

- **sap.m.MultiComboBox** - Multi-select dropdown
  - Features: Token display, selection change events
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.m.MultiComboBox

- **sap.m.CheckBox** - Checklist checkboxes
  - Events: select
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.m.CheckBox

### Layout Components
- **sap.m.FlexBox** - Flexible layout
  - Properties: direction, justifyContent, alignItems
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.m.FlexBox

- **sap.ui.layout.Grid** - Responsive grid
  - Properties: defaultSpan (L, M, S)
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.ui.layout.Grid

- **sap.m.VBox / HBox** - Vertical/Horizontal box
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.m.VBox

### Dialog Components
- **sap.m.Dialog** - Modal dialog
  - Features: Draggable, closable, customizable buttons
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.m.Dialog

- **sap.m.MessageBox** - Standard message dialogs
  - Methods: alert, confirm, error, information, warning
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.m.MessageBox

- **sap.m.BusyDialog** - Loading indicator
  - Properties: text, showCancelButton
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.m.BusyDialog

### Container Components
- **sap.m.Page** - Page container
  - Properties: title, showHeader, showFooter, content
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.m.Page

- **sap.m.Panel** - Collapsible panel
  - Properties: headerText, expandable, expanded
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.m.Panel

- **sap.m.IconTabBar** - Tab navigation
  - Features: Icons, counters, filters
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.m.IconTabBar

### Visualization Components
- **sap.suite.ui.commons.NetworkGraph** - Network visualization
  - Features: Nodes, edges, layouts, zoom
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.suite.ui.commons.NetworkGraph

- **sap.suite.ui.commons.MicroProcessFlow** - Mini process flow
  - Features: Nodes, status indicators
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.suite.ui.commons.MicroProcessFlow

### Toolbar Components
- **sap.m.OverflowToolbar** - Responsive toolbar
  - Features: Auto-overflow to menu, spacing
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.m.OverflowToolbar

- **sap.m.Toolbar** - Standard toolbar
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.m.Toolbar

- **sap.m.ToolbarSpacer** - Flexible spacing
  - Reference: https://sapui5.hana.ondemand.com/sdk/#/api/sap.m.ToolbarSpacer

### Custom Components (To Be Built)
- **SmartActionButton** - AI-powered action button
  - Extends: sap.m.Button
  - Features: Calls AI API, shows response dialog

- **AICommentaryGenerator** - Commentary generation UI
  - Uses: sap.m.Dialog, sap.m.TextArea, sap.m.Button
  - Features: Generate, edit, copy, regenerate

- **NetworkGraphControl** - Data lineage visualization
  - Uses: D3.js + SVG
  - Features: Interactive graph, zoom, pan

- **ProcessFlowControl** - Workflow stage visualization
  - Uses: sap.suite.ui.commons.ProcessFlow
  - Features: 13 stages, status indicators, progress

- **LineageViewerControl** - Lineage chain viewer
  - Uses: sap.suite.ui.commons.NetworkGraph or custom D3.js
  - Features: Transformation chain, data hashes

## SAP Fiori Design Principles

### Layout Guidelines
- Use **sap.m.Page** as root container for all pages
- Add **sap.m.OverflowToolbar** for responsive action buttons
- Use **sap.m.FlexBox** or **sap.ui.layout.Grid** for responsive layouts
- Standard spacing: 0.5rem (small), 1rem (medium), 2rem (large)

### Color Coding
- **Success/Positive:** Green (`sapUiPositive`, `Success`)
- **Warning:** Orange (`sapUiCritical`, `Warning`)
- **Error/Negative:** Red (`sapUiNegative`, `Error`)
- **Neutral:** Gray (`sapUiNeutral`, `None`)

### Icons
- Standard SAP icons: `sap-icon://[icon-name]`
- Common icons for Trial Balance:
  - `sap-icon://accounting-document-verification` - Verification
  - `sap-icon://wallet` - Financial
  - `sap-icon://alert` - Warnings
  - `sap-icon://sys-help` - Help/Info
  - `sap-icon://line-chart` - Analytics
  - `sap-icon://checklist` - Checklist

### Responsive Behavior
- Use `sapUiSizeCompact` CSS class for desktop
- Use responsive containers (FlexBox, Grid)
- Mobile-first approach for layouts
- Breakpoints: S (<600px), M (600-1024px), L (1024-1440px), XL (>1440px)

## Component Usage Patterns

### Pattern 1: Metric Tiles (Overview Page)
```xml
<NumericTile value="1,234,567" unit="USD" scale="M" state="Good">
  <TileContent>
    <NumericContent value="1.23" scale="M" valueColor="Good" />
  </TileContent>
</NumericTile>
```

### Pattern 2: Smart Action Button (All Pages)
```javascript
new SmartActionButton({
  actionId: "overview:explain_metrics",
  actionText: "💡 Explain Metrics",
  icon: "sap-icon://lightbulb",
  contextData: this.getView().getModel("data").getData()
})
```

### Pattern 3: AI Commentary Generator (Variance Page)
```javascript
new AICommentaryGenerator({
  varianceData: {
    account: "Revenue",
    variance: 125000,
    variance_pct: 15.5
  },
  commentaryGenerated: function(oEvent) {
    const commentary = oEvent.getParameter("commentary");
    // Save to model
  }
})
```

### Pattern 4: Data Table with Filtering (Raw Data)
```xml
<Page title="Raw Data">
  <content>
    <OverflowToolbar>
      <SearchField search="onSearch" width="20rem"/>
      <MultiComboBox items="{filters>/accountTypes}"/>
      <ToolbarSpacer/>
      <Button text="Export" icon="sap-icon://excel-attachment"/>
    </OverflowToolbar>
    <Table rows="{/rawData}" visibleRowCount="20">
      <columns>
        <Column><Label text="Account"/></Column>
        <Column><Label text="Amount"/></Column>
      </columns>
    </Table>
  </content>
</Page>
```

---

# 🚀 Next Steps

1. **Review this plan** - Ensure alignment with goals
2. **Bookmark Fiori Design System** - https://experience.sap.com/fiori-design-web/
3. **Bookmark UI5 SDK** - https://sapui5.hana.ondemand.com/sdk/
4. **Set up tracking** - Use checkboxes to track progress
5. **Start Day 1** - Begin with data models (backend first)
6. **Daily reviews** - Check progress against plan
7. **Weekly reviews** - Ensure layer completion

---

**Last Updated:** January 27, 2026 (Updated with SAP Fiori UI5 component specifications)  
**Status:** Ready to Begin  
**Start Date:** January 27, 2026  
**Target Completion:** February 21, 2026  
**UI Framework:** SAP UI5 v1.136+ with Fiori Design System
