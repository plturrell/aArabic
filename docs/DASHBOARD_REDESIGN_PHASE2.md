# Dashboard Redesign - Phase 2: SAP Fiori Analytical Cards

## Overview
Redesigning the main Dashboard view to use SAP Fiori Analytical Cards with time-series data, model-specific metrics, and proper ordering.

## Current Issues
1. Using generic GenericTile instead of Analytical Cards
2. No time-series visualization
3. Metrics not model-specific
4. Prompt Testing buried at bottom (should be at top)
5. No user profile/default model selector
6. No current vs. historical comparison

## Redesign Structure

### 1. Header Section
```
[Breadcrumb: Home / Dashboard]                [Connected Status] [Model Selector ▼] [Model Configurator Button]
```

**New Model Selector:**
- Dropdown to select active model
- Shows: Model Name | Variant | Status
- Default from user profile
- Changes all dashboard metrics to selected model

### 2. Content Order (NEW)

#### A. Quick Prompt Testing (TOP - As Requested)
- **Compact prompt interface**
- Model already selected from header
- Text area for prompt
- Send button
- Response area with current vs. average metrics comparison
- **Show**: "This request: 45ms | Your average: 67ms (↓33% faster)"

#### B. Model Performance Analytics (SAP Fiori Analytical Cards)
**Grid of 6 Analytical Cards with time-series charts:**

1. **Latency Over Time** (Line Chart)
   - X-axis: Last 1 hour (timestamps)
   - Y-axis: Latency (ms)
   - Lines: P50, P95, P99
   - Current value highlighted
   - Threshold lines for targets

2. **Throughput Trends** (Area Chart)
   - Tokens per second over time
   - Shows capacity utilization
   - Peak indicators

3. **TTFT Performance** (Bar Chart)
   - Time to First Token
   - Compare current hour vs. previous hours
   - Target line at 100ms

4. **Cache Hit Rate** (Line Chart with Threshold)
   - Cache efficiency over time
   - Color zones: >70% green, 50-70% yellow, <50% red
   - Shows cache warming effect

5. **Queue Depth** (Area Chart)
   - Pending requests over time
   - Identifies congestion patterns
   - Capacity planning insights

6. **Token Distribution** (Stacked Bar Chart)
   - Input vs. Output tokens
   - Shows workload characteristics
   - Cost estimation

#### C. Tier Statistics (Moved Down)
- Keep current visualization
- Add sparklines showing tier usage trends
- Model-specific tier allocation

#### D. Model Comparison Table
- Table showing all available models
- Columns: Model, Status, Avg Latency, Throughput, Last Used
- Click to switch active model
- Highlight current selection

## SAP Fiori Analytical Card Structure

### Example: Latency Analytical Card
```xml
<f:Card class="sapUiMediumMargin">
    <f:header>
        <card:Header
            title="Latency Trends"
            subtitle="Last 1 Hour - LFM2.5-1.2B-Q4_0"/>
    </f:header>
    <f:content>
        <viz:VizFrame
            id="latencyChart"
            uiConfig="{applicationSet:'fiori'}"
            height="200px"
            width="100%"
            vizType='timeseries_line'>
            <viz:dataset>
                <viz.data:FlattenedDataset data="{metrics>/latencyHistory}">
                    <viz.data:dimensions>
                        <viz.data:DimensionDefinition name="Timestamp" value="{timestamp}"/>
                    </viz.data:dimensions>
                    <viz.data:measures>
                        <viz.data:MeasureDefinition name="P50" value="{p50}"/>
                        <viz.data:MeasureDefinition name="P95" value="{p95}"/>
                        <viz.data:MeasureDefinition name="P99" value="{p99}"/>
                    </viz.data:measures>
                </viz.data:FlattenedDataset>
            </viz:dataset>
            <viz:feeds>
                <viz.feeds:FeedItem uid="valueAxis" type="Measure" values="P50,P95,P99"/>
                <viz.feeds:FeedItem uid="timeAxis" type="Dimension" values="Timestamp"/>
            </viz:feeds>
        </viz:VizFrame>
    </f:content>
</f:Card>
```

## Data Model Requirements

### metrics Model Structure (Updated)
```javascript
{
  selectedModel: "lfm2.5-1.2b-q4_0",  // User's active model
  models: [
    {
      id: "lfm2.5-1.2b-q4_0",
      display_name: "LFM2.5 1.2B Q4_0",
      architecture: "lfm2",
      health: "healthy",
      // Current metrics
      currentMetrics: {
        latency: 45,
        ttft: 23,
        tps: 67,
        queueDepth: 3
      },
      // Historical averages (for comparison)
      historicalAverages: {
        latency: 67,
        ttft: 35,
        tps: 54
      },
      // Time-series data (last 1 hour, 1-minute intervals = 60 points)
      latencyHistory: [
        { timestamp: "2026-01-20T13:47:00Z", p50: 45, p95: 89, p99: 156 },
        { timestamp: "2026-01-20T13:48:00Z", p50: 43, p95: 87, p99: 151 },
        // ... 60 points total
      ],
      throughputHistory: [
        { timestamp: "2026-01-20T13:47:00Z", tps: 67 },
        // ...
      ],
      cacheHitHistory: [
        { timestamp: "2026-01-20T13:47:00Z", hitRate: 0.82 },
        // ...
      ]
    }
  ],
  // User profile
  userProfile: {
    defaultModel: "lfm2.5-1.2b-q4_0",
    preferredTemperature: 0.7,
    preferredMaxTokens: 512
  }
}
```

## Implementation Steps

### Step 1: Add Model Selector to Header
- Update breadcrumb bar
- Add Select control with available models
- Bind to metrics>/selectedModel
- On change: update all metrics displays

### Step 2: Move Prompt Testing to Top
- Extract Prompt Testing panel
- Place immediately after header
- Make it compact (collapsible)
- Add comparison metrics overlay

### Step 3: Create Analytical Cards
- Add sap.f and sap.viz libraries to manifest
- Create 6 analytical cards with VizFrames
- Implement time-series line/area/bar charts
- Add proper data binding

### Step 4: Update Controller
- Add model selection handler
- Fetch time-series data for selected model
- Update chart data on model change
- Implement current vs. historical comparison

### Step 5: Rearrange Sections
1. Prompt Testing (compact)
2. Model Performance Analytics (6 cards)
3. Tier Statistics (existing, moved down)
4. Model Comparison Table

## Required Dependencies

### manifest.json additions:
```json
"sap.ui5": {
  "dependencies": {
    "libs": {
      "sap.f": {},
      "sap.viz": {}
    }
  }
}
```

## Next Actions
1. Update manifest.json with required libraries
2. Create new Main.view.xml with Analytical Cards
3. Update Main.controller.js with chart logic
4. Add mock time-series data for testing
5. Later: Connect to SAP HANA for real data

## Benefits
- ✅ Time-series visualization shows trends
- ✅ Model-specific metrics (per variant)
- ✅ Quick prompt testing at top
- ✅ Current vs. historical comparison
- ✅ SAP Fiori design patterns
- ✅ Better insights for performance tuning
- ✅ User-selected default model
