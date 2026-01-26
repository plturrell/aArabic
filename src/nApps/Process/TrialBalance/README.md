# Trial Balance Application

A comprehensive Trial Balance application built with UI5 Freestyle, n-c-sdk (Zig) backend, and SAP BTP integration.

## Overview

This application provides a complete Trial Balance management system with:
- Real-time balance monitoring and analysis
- Maker-Checker-Manager approval workflow
- AI-powered narrative generation and inference
- HANA Cloud integration for data persistence
- Integration with nServices ecosystem via SAP Destinations

## Architecture

### Frontend (webapp/)
- **Framework**: SAPUI5 Freestyle
- **Components**: Custom components from nLocalModels (Charts, NetworkGraph, ProcessFlow, WorkflowBuilder)
- **Deployment**: SAP BTP Cloud Foundry

### Backend (backend/)
- **Language**: Zig (using n-c-sdk standards)
- **Features**: High-performance balance calculations, workflow engine, analytics
- **Database**: SAP HANA Cloud

### Integration (integrations/)
- **Method**: SAP Destination services
- **Services**: nAgentFlow, nAgentMeta, nLocalModels, nGrounding

## User Roles

1. **Maker**: Creates and modifies trial balance entries
2. **Checker**: Reviews and validates entries
3. **Manager**: Final approval and sign-off

## Project Structure

```
TrialBalance/
├── webapp/              # UI5 Freestyle application
├── backend/             # Zig backend services
├── integrations/        # nServices API integration layer
├── config/              # Configuration files
├── BusDocs/             # Business documentation and sample data
├── mta.yaml            # Multi-Target Application descriptor
├── xs-security.json    # Security configuration
└── package.json        # Node.js dependencies
```

## Prerequisites

- Node.js >= 18.x
- Zig >= 0.13.0
- SAP BTP account with HANA Cloud
- UI5 CLI: `npm install -g @ui5/cli`

## Setup

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure Destinations

Update `config/destinations/destinations.json` with your BTP destination configurations.

### 3. Deploy HANA Artifacts

```bash
# Deploy HANA schema and views
cf deploy config/hana/
```

### 4. Build Backend

```bash
cd backend
zig build
```

### 5. Run Locally

```bash
# Start backend server
cd backend && zig build run

# In another terminal, start UI5 app
cd webapp && ui5 serve
```

## Deployment

### Deploy to SAP BTP

```bash
# Build MTA archive
mbt build

# Deploy to Cloud Foundry
cf deploy mta_archives/trial-balance_1.0.0.mtar
```

## Development

### Frontend Development

```bash
cd webapp
npm run start
```

### Backend Development

```bash
cd backend
zig build test
zig build run
```

## Features

### Core Functionality
- Trial Balance Dashboard with account hierarchy
- Real-time balance calculations (Debit/Credit)
- Period-over-period comparison
- Account drill-down capabilities

### Workflow Management
- Three-tier approval process (Maker-Checker-Manager)
- Audit trail for all actions
- Email notifications
- Role-based access control

### Analytics & Reporting
- Variance analysis
- Trend visualization
- AI-generated narrative summaries
- Export to Excel/PDF

### Integration Features
- Real-time HANA Cloud synchronization
- nAgentFlow workflow orchestration
- nLocalModels AI inference
- nGrounding context enrichment

## Configuration

### Environment Variables

```bash
# Backend
HANA_HOST=<hana-cloud-host>
HANA_PORT=443
HANA_USER=<username>
HANA_PASSWORD=<password>

# Services
AGENT_FLOW_URL=<agent-flow-url>
LOCAL_MODELS_URL=<local-models-url>
```

### Destinations (BTP Cockpit)

Configure the following destinations:
- `HANA_CLOUD` - HANA Cloud database
- `AGENT_FLOW` - nAgentFlow service
- `AGENT_META` - nAgentMeta service
- `LOCAL_MODELS` - nLocalModels service
- `GROUNDING` - nGrounding service

## Testing

```bash
# Frontend tests
cd webapp
npm test

# Backend tests
cd backend
zig build test

# Integration tests
npm run test:integration
```

## Documentation

- [Business Requirements](./BusDocs/README.md)
- [API Documentation](./backend/docs/API.md)
- [Integration Guide](./integrations/docs/API_INTEGRATION.md)
- [Workflow Configuration](./config/workflow/README.md)

## Contributing

Please refer to the main project's contributing guidelines.

## License

Proprietary - Internal Use Only

## Support

For issues and support, contact the development team.