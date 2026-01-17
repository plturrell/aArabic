#!/bin/bash
# GET /odata/v4/research/Presentation('pres_20260116_200000')/Slides
curl -X GET "http://localhost:8080/odata/v4/research/Presentation('pres_20260116_200000')/Slides" \
  -H "Accept: application/json"
