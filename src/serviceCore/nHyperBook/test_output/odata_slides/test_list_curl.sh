#!/bin/bash
# GET /odata/v4/research/Presentation
curl -X GET http://localhost:8080/odata/v4/research/Presentation \
  -H "Accept: application/json"
