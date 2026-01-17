#!/bin/bash
# GET /odata/v4/research/Presentation?$filter=SourceId eq 'source_001'
curl -X GET "http://localhost:8080/odata/v4/research/Presentation?\$filter=SourceId%20eq%20'source_001'" \
  -H "Accept: application/json"
