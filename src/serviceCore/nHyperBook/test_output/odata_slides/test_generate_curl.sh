#!/bin/bash
# POST /odata/v4/research/GenerateSlides
curl -X POST http://localhost:8080/odata/v4/research/GenerateSlides \
  -H "Content-Type: application/json" \
  -d @test_output/odata_slides/generate_request.json
