"""
Application constants
"""

# API Endpoints
API_PREFIX = "/api/v1"
ORCHESTRATE_PREFIX = "/orchestrate"
A2UI_PREFIX = "/a2ui"

# Model Names
MODEL_CAMELBERT = "camelbert-dialect-financial"
MODEL_M2M100 = "m2m100-418M"

# Workflow Types
WORKFLOW_TYPE_INVOICE_PROCESSING = "invoice-processing"
WORKFLOW_TYPE_DEFAULT = "default-invoice-processing"

# Process Types
PROCESS_TYPE_OCR = "ocr"
PROCESS_TYPE_TRANSLATION = "translation"
PROCESS_TYPE_ANALYSIS = "analysis"
PROCESS_TYPE_VALIDATION = "validation"
PROCESS_TYPE_SUBMISSION = "submission"

# Node Types
NODE_TYPE_START = "start"
NODE_TYPE_END = "end"
NODE_TYPE_PROCESS = "process"
NODE_TYPE_DECISION = "decision"
NODE_TYPE_ACTION = "action"

# Confidence Thresholds
CONFIDENCE_THRESHOLD_HIGH = 0.85
CONFIDENCE_THRESHOLD_MEDIUM = 0.70
CONFIDENCE_THRESHOLD_LOW = 0.50

# Error Messages
ERROR_MODEL_NOT_LOADED = "Model not loaded"
ERROR_WORKFLOW_NOT_FOUND = "Workflow not found"
ERROR_ORCHESTRATION_UNAVAILABLE = "Orchestration service not available"
ERROR_A2UI_UNAVAILABLE = "A2UI service not available"
ERROR_INVALID_INPUT = "Invalid input provided"
ERROR_INTERNAL_SERVER = "Internal server error"

# HTTP Status Codes
HTTP_OK = 200
HTTP_CREATED = 201
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_INTERNAL_SERVER_ERROR = 500
HTTP_SERVICE_UNAVAILABLE = 503

