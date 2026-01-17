#!/bin/bash
# POST /odata/v4/research/ExportPresentation
curl -X POST http://localhost:8080/odata/v4/research/ExportPresentation \
  -H "Content-Type: application/json" \
  -d @test_output/odata_slides/export_request.json
