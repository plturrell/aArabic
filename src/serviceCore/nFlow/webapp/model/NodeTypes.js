sap.ui.define([], function () {
    "use strict";

    /**
     * Node Types definition for the Workflow Editor
     * Defines all available node types with their properties, ports, and configuration
     */
    return {
        categories: [
            { id: "flow", name: "Flow Control", icon: "sap-icon://process" },
            { id: "action", name: "Actions", icon: "sap-icon://action" },
            { id: "ai", name: "AI & ML", icon: "sap-icon://machine-learning" },
            { id: "data", name: "Data", icon: "sap-icon://database" }
        ],

        nodes: {
            start: {
                id: "start",
                name: "Start",
                icon: "sap-icon://begin",
                category: "flow",
                color: "#2ECC71",
                description: "Workflow entry point",
                inputs: [],
                outputs: [{ id: "out", name: "Output", type: "flow" }],
                properties: [
                    { id: "triggerType", name: "Trigger Type", type: "select", 
                      options: ["manual", "scheduled", "webhook", "event"], default: "manual" },
                    { id: "schedule", name: "Schedule (Cron)", type: "string", default: "", 
                      visible: { field: "triggerType", value: "scheduled" } }
                ],
                maxInstances: 1
            },
            end: {
                id: "end",
                name: "End",
                icon: "sap-icon://stop",
                category: "flow",
                color: "#E74C3C",
                description: "Workflow exit point",
                inputs: [{ id: "in", name: "Input", type: "flow" }],
                outputs: [],
                properties: [
                    { id: "status", name: "Exit Status", type: "select", 
                      options: ["success", "failure", "cancelled"], default: "success" }
                ],
                maxInstances: 1
            },
            task: {
                id: "task",
                name: "Task",
                icon: "sap-icon://task",
                category: "action",
                color: "#3498DB",
                description: "Generic task node",
                inputs: [{ id: "in", name: "Input", type: "flow" }],
                outputs: [{ id: "out", name: "Output", type: "flow" }],
                properties: [
                    { id: "name", name: "Task Name", type: "string", default: "New Task" },
                    { id: "description", name: "Description", type: "text", default: "" },
                    { id: "assignee", name: "Assignee", type: "string", default: "" },
                    { id: "timeout", name: "Timeout (seconds)", type: "number", default: 3600 }
                ]
            },
            decision: {
                id: "decision",
                name: "Decision",
                icon: "sap-icon://decision",
                category: "flow",
                color: "#F39C12",
                description: "Conditional branching",
                inputs: [{ id: "in", name: "Input", type: "flow" }],
                outputs: [
                    { id: "true", name: "True", type: "flow" },
                    { id: "false", name: "False", type: "flow" }
                ],
                properties: [
                    { id: "condition", name: "Condition", type: "expression", default: "" },
                    { id: "conditionType", name: "Condition Type", type: "select",
                      options: ["expression", "script", "rule"], default: "expression" }
                ]
            },
            llm: {
                id: "llm",
                name: "LLM",
                icon: "sap-icon://ai",
                category: "ai",
                color: "#9B59B6",
                description: "Large Language Model processing",
                inputs: [{ id: "in", name: "Input", type: "flow" }],
                outputs: [{ id: "out", name: "Output", type: "flow" }],
                properties: [
                    { id: "model", name: "Model", type: "select",
                      options: ["gpt-4", "gpt-3.5-turbo", "claude-3", "llama-2"], default: "gpt-4" },
                    { id: "systemPrompt", name: "System Prompt", type: "text", default: "" },
                    { id: "userPrompt", name: "User Prompt Template", type: "text", default: "" },
                    { id: "temperature", name: "Temperature", type: "number", default: 0.7, min: 0, max: 2 },
                    { id: "maxTokens", name: "Max Tokens", type: "number", default: 1000 }
                ]
            },
            http: {
                id: "http",
                name: "HTTP",
                icon: "sap-icon://world",
                category: "action",
                color: "#1ABC9C",
                description: "HTTP request node",
                inputs: [{ id: "in", name: "Input", type: "flow" }],
                outputs: [
                    { id: "success", name: "Success", type: "flow" },
                    { id: "error", name: "Error", type: "flow" }
                ],
                properties: [
                    { id: "method", name: "Method", type: "select",
                      options: ["GET", "POST", "PUT", "PATCH", "DELETE"], default: "GET" },
                    { id: "url", name: "URL", type: "string", default: "" },
                    { id: "headers", name: "Headers", type: "keyvalue", default: [] },
                    { id: "body", name: "Body", type: "code", default: "", language: "json" },
                    { id: "timeout", name: "Timeout (ms)", type: "number", default: 30000 }
                ]
            },
            database: {
                id: "database",
                name: "Database",
                icon: "sap-icon://database",
                category: "data",
                color: "#34495E",
                description: "Database operation node",
                inputs: [{ id: "in", name: "Input", type: "flow" }],
                outputs: [{ id: "out", name: "Output", type: "flow" }],
                properties: [
                    { id: "connection", name: "Connection", type: "string", default: "" },
                    { id: "operation", name: "Operation", type: "select",
                      options: ["query", "insert", "update", "delete"], default: "query" },
                    { id: "query", name: "Query/Table", type: "code", default: "", language: "sql" }
                ]
            },
            transform: {
                id: "transform",
                name: "Transform",
                icon: "sap-icon://workflow-transform",
                category: "data",
                color: "#E67E22",
                description: "Data transformation node",
                inputs: [{ id: "in", name: "Input", type: "flow" }],
                outputs: [{ id: "out", name: "Output", type: "flow" }],
                properties: [
                    { id: "transformType", name: "Transform Type", type: "select",
                      options: ["mapping", "script", "template"], default: "mapping" },
                    { id: "script", name: "Transform Script", type: "code", default: "", language: "javascript" }
                ]
            },
            filter: {
                id: "filter",
                name: "Filter",
                icon: "sap-icon://filter",
                category: "data",
                color: "#16A085",
                description: "Filter data node",
                inputs: [{ id: "in", name: "Input", type: "flow" }],
                outputs: [
                    { id: "match", name: "Match", type: "flow" },
                    { id: "nomatch", name: "No Match", type: "flow" }
                ],
                properties: [
                    { id: "filterExpression", name: "Filter Expression", type: "expression", default: "" }
                ]
            },
            aggregate: {
                id: "aggregate",
                name: "Aggregate",
                icon: "sap-icon://sum",
                category: "data",
                color: "#8E44AD",
                description: "Aggregate data node",
                inputs: [{ id: "in", name: "Input", type: "flow" }],
                outputs: [{ id: "out", name: "Output", type: "flow" }],
                properties: [
                    { id: "groupBy", name: "Group By", type: "string", default: "" },
                    { id: "aggregations", name: "Aggregations", type: "keyvalue", default: [] }
                ]
            }
        },

        /**
         * Get node type by ID
         */
        getNodeType: function (sId) {
            return this.nodes[sId] || null;
        },

        /**
         * Get all node types as array
         */
        getAllNodeTypes: function () {
            return Object.values(this.nodes);
        },

        /**
         * Get node types by category
         */
        getNodesByCategory: function (sCategoryId) {
            return Object.values(this.nodes).filter(function (node) {
                return node.category === sCategoryId;
            });
        }
    };
});

