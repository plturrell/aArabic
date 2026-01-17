#!/bin/bash
# DELETE /odata/v4/research/Presentation('pres_20260116_200000')
curl -X DELETE "http://localhost:8080/odata/v4/research/Presentation('pres_20260116_200000')" \
  -v
