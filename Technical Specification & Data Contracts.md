# Technical Specification: CAD Extraction & Verification

## 1. Data Schemas (The Handshake)

### 1.1 Geometry B-Rep Schema (Phase 2)
```json
{
  "page_number": 1,
  "dpi_reference": 300,
  "geometries": [
    {
      "primitive_id": "GEO_UUID",
      "type": "rectangle | circle | line",
      "coordinates": {"x1": 0, "y1": 0, "x2": 0, "y2": 0},
      "centroid": [x, y],
      "properties": { "width_px": 0, "height_px": 0 }
    }
  ]
}.2 Table & Text Schema (Phase 3)
JSON
{
  "tables": [
    {
      "table_id": "TBL_UUID",
      "rows": [
        {
          "row_index": 1,
          "cells": [
            {"column": "Mark", "text": "C1", "bbox": [x1, y1, x2, y2]},
            {"column": "Size", "text": "400x400", "bbox": [x1, y1, x2, y2]}
          ]
        }
      ]
    }
  ]
}