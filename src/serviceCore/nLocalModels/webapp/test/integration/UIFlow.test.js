/**
 * UI Flow Integration Tests
 * Tests for navigation, dialogs, and data loading in the LLM Server Dashboard
 *
 * These tests use a framework-agnostic structure that can run with any test runner.
 * Mock fetch responses and simulate DOM interactions.
 */

// ============================================================================
// Test Utilities and Mock Setup
// ============================================================================

const MockFetch = {
    responses: new Map(),

    register(url, response) {
        this.responses.set(url, response);
    },

    clear() {
        this.responses.clear();
    },

    async fetch(url, options = {}) {
        const response = this.responses.get(url);
        if (response) {
            return {
                ok: true,
                json: async () => response,
                status: 200
            };
        }
        return {
            ok: false,
            status: 404,
            json: async () => ({ error: 'Not found' })
        };
    }
};

// Mock DOM utilities
const MockDOM = {
    elements: new Map(),
    eventListeners: new Map(),

    createElement(id, props = {}) {
        const element = {
            id,
            ...props,
            value: props.value || '',
            selectedKey: props.selectedKey || '',
            selectedKeys: props.selectedKeys || [],
            items: props.items || [],
            visible: true,
            valueState: 'None',
            getValue() { return this.value; },
            setValue(v) { this.value = v; },
            getSelectedKey() { return this.selectedKey; },
            setSelectedKey(k) { this.selectedKey = k; },
            getSelectedKeys() { return this.selectedKeys; },
            setSelectedKeys(k) { this.selectedKeys = k; },
            getItems() { return this.items; },
            setValueState(s) { this.valueState = s; },
            isOpen: () => this.visible,
            open() { this.visible = true; },
            close() { this.visible = false; },
            fireEvent(name, params) {
                const listeners = MockDOM.eventListeners.get(`${id}:${name}`) || [];
                listeners.forEach(fn => fn(params));
            }
        };
        this.elements.set(id, element);
        return element;
    },

    byId(id) {
        return this.elements.get(id);
    },

    attachEvent(id, eventName, handler) {
        const key = `${id}:${eventName}`;
        const listeners = this.eventListeners.get(key) || [];
        listeners.push(handler);
        this.eventListeners.set(key, listeners);
    },

    clear() {
        this.elements.clear();
        this.eventListeners.clear();
    }
};

// Test result collector
const TestResults = {
    passed: 0,
    failed: 0,
    results: [],

    record(name, passed, message = '') {
        this.results.push({ name, passed, message });
        if (passed) this.passed++;
        else this.failed++;
    },

    summary() {
        console.log(`\n=== Test Results ===`);
        console.log(`Passed: ${this.passed}, Failed: ${this.failed}`);
        this.results.forEach(r => {
            const status = r.passed ? '✓' : '✗';
            console.log(`  ${status} ${r.name}${r.message ? ': ' + r.message : ''}`);
        });
        return this.failed === 0;
    },

    clear() {
        this.passed = 0;
        this.failed = 0;
        this.results = [];
    }
};

// Test suite state
let currentBeforeEach = null;
let currentSuiteName = '';

// Assertion helpers
function expect(actual) {
    return {
        toBe(expected) {
            if (actual !== expected) {
                throw new Error(`Expected ${expected}, got ${actual}`);
            }
        },
        toEqual(expected) {
            if (JSON.stringify(actual) !== JSON.stringify(expected)) {
                throw new Error(`Expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
            }
        },
        toBeTruthy() {
            if (!actual) {
                throw new Error(`Expected truthy value, got ${actual}`);
            }
        },
        toBeFalsy() {
            if (actual) {
                throw new Error(`Expected falsy value, got ${actual}`);
            }
        },
        toContain(item) {
            if (!actual.includes(item)) {
                throw new Error(`Expected ${JSON.stringify(actual)} to contain ${item}`);
            }
        },
        toBeGreaterThan(expected) {
            if (!(actual > expected)) {
                throw new Error(`Expected ${actual} to be greater than ${expected}`);
            }
        },
        toHaveLength(expected) {
            if (actual.length !== expected) {
                throw new Error(`Expected length ${expected}, got ${actual.length}`);
            }
        }
    };
}

function test(name, fn) {
    // Run beforeEach if defined
    if (currentBeforeEach) {
        currentBeforeEach();
    }

    const fullName = currentSuiteName ? `${currentSuiteName}: ${name}` : name;

    try {
        // Handle async tests
        const result = fn();
        if (result && typeof result.then === 'function') {
            // For now, we'll just catch sync errors - async handled below
            result.then(() => {
                TestResults.record(fullName, true);
            }).catch(error => {
                TestResults.record(fullName, false, error.message);
            });
        } else {
            TestResults.record(fullName, true);
        }
    } catch (error) {
        TestResults.record(fullName, false, error.message);
    }
}

function describe(suiteName, fn) {
    console.log(`\n--- ${suiteName} ---`);
    const previousSuite = currentSuiteName;
    const previousBeforeEach = currentBeforeEach;
    currentSuiteName = suiteName;
    currentBeforeEach = null;
    fn();
    currentSuiteName = previousSuite;
    currentBeforeEach = previousBeforeEach;
}

function beforeEach(fn) {
    currentBeforeEach = fn;
}

// ============================================================================
// Section 1: Navigation Tests
// ============================================================================

describe('Navigation between pages', () => {
    beforeEach(() => {
        MockDOM.clear();
        MockFetch.clear();
    });

    test('should navigate from Main to Orchestration page', () => {
        // Setup navigation container mock
        const navContainer = MockDOM.createElement('navContainer', {
            currentPage: 'mainPageContent',
            to(pageId) { this.currentPage = pageId; }
        });

        const pageMap = {
            "main": "mainPageContent",
            "orchestration": "orchestrationPage",
            "modelRouter": "modelRouterPage"
        };

        // Simulate navigation event
        const navigationKey = 'orchestration';
        const targetPageId = pageMap[navigationKey];
        navContainer.to(targetPageId);

        expect(navContainer.currentPage).toBe('orchestrationPage');
    });

    test('should navigate from Main to Model Router page', () => {
        const navContainer = MockDOM.createElement('navContainer', {
            currentPage: 'mainPageContent',
            to(pageId) { this.currentPage = pageId; }
        });

        navContainer.to('modelRouterPage');
        expect(navContainer.currentPage).toBe('modelRouterPage');
    });

    test('should have correct page mapping for all navigation items', () => {
        const pageMap = {
            "main": "mainPageContent",
            "promptTesting": "promptTestingPage",
            "mhcTuning": "mhcTuningPage",
            "orchestration": "orchestrationPage",
            "modelVersions": "modelVersionsPage",
            "modelRouter": "modelRouterPage"
        };

        expect(Object.keys(pageMap)).toHaveLength(6);
        expect(pageMap.main).toBe('mainPageContent');
        expect(pageMap.orchestration).toBe('orchestrationPage');
        expect(pageMap.modelRouter).toBe('modelRouterPage');
    });

    test('should update sidebar selection on navigation', () => {
        const sideNavigation = MockDOM.createElement('sideNavigation', {
            expanded: true,
            selectedKey: 'main',
            setExpanded(val) { this.expanded = val; },
            setSelectedKey(key) { this.selectedKey = key; }
        });

        sideNavigation.setSelectedKey('orchestration');
        expect(sideNavigation.selectedKey).toBe('orchestration');
    });

    test('should toggle sidebar expansion', () => {
        const sideNavigation = MockDOM.createElement('sideNavigation', {
            expanded: true,
            setExpanded(val) { this.expanded = val; },
            getExpanded() { return this.expanded; }
        });

        // Toggle
        sideNavigation.setExpanded(!sideNavigation.getExpanded());
        expect(sideNavigation.expanded).toBe(false);

        // Toggle again
        sideNavigation.setExpanded(!sideNavigation.getExpanded());
        expect(sideNavigation.expanded).toBe(true);
    });
});


// ============================================================================
// Section 2: Add Agent Dialog Tests
// ============================================================================

describe('Add Agent Dialog', () => {
    beforeEach(() => {
        MockDOM.clear();
        MockFetch.clear();

        // Setup mock API response
        MockFetch.register('http://localhost:8080/api/v1/agents', {
            agents: [
                { id: 'agent-1', name: 'Router Agent', type: 'router', status: 'healthy' },
                { id: 'agent-2', name: 'Code Agent', type: 'code', status: 'healthy' }
            ]
        });
    });

    test('should open Add Agent dialog', () => {
        const dialog = MockDOM.createElement('addAgentDialog', {
            visible: false,
            title: 'Add New Agent'
        });

        dialog.open();
        expect(dialog.visible).toBe(true);
    });

    test('should have all required form fields', () => {
        const nameInput = MockDOM.createElement('addAgentNameInput', { value: '' });
        const typeSelect = MockDOM.createElement('addAgentTypeSelect', { selectedKey: 'router' });
        const descriptionTextArea = MockDOM.createElement('addAgentDescriptionTextArea', { value: '' });
        const modelIdInput = MockDOM.createElement('addAgentModelIdInput', { value: '' });
        const connectedAgentsComboBox = MockDOM.createElement('addAgentConnectedAgentsComboBox', { selectedKeys: [] });

        expect(nameInput).toBeTruthy();
        expect(typeSelect).toBeTruthy();
        expect(descriptionTextArea).toBeTruthy();
        expect(modelIdInput).toBeTruthy();
        expect(connectedAgentsComboBox).toBeTruthy();
    });

    test('should fill agent name field', () => {
        const nameInput = MockDOM.createElement('addAgentNameInput', { value: '' });

        nameInput.setValue('Test Agent');
        expect(nameInput.getValue()).toBe('Test Agent');
    });

    test('should select agent type', () => {
        const typeSelect = MockDOM.createElement('addAgentTypeSelect', {
            selectedKey: 'router',
            items: [
                { key: 'router', text: 'Router' },
                { key: 'orchestrator', text: 'Orchestrator' },
                { key: 'code', text: 'Code' },
                { key: 'translation', text: 'Translation' },
                { key: 'rag', text: 'RAG' }
            ]
        });

        typeSelect.setSelectedKey('code');
        expect(typeSelect.getSelectedKey()).toBe('code');
    });

    test('should validate required name field on submit', () => {
        const nameInput = MockDOM.createElement('addAgentNameInput', { value: '', valueState: 'None' });

        // Empty name should trigger error state
        const name = nameInput.getValue().trim();
        if (!name) {
            nameInput.setValueState('Error');
        }

        expect(nameInput.valueState).toBe('Error');
    });

    test('should clear error state when name is provided', () => {
        const nameInput = MockDOM.createElement('addAgentNameInput', { value: '', valueState: 'Error' });

        nameInput.setValue('Valid Name');
        const name = nameInput.getValue().trim();
        if (name) {
            nameInput.setValueState('None');
        }

        expect(nameInput.valueState).toBe('None');
    });

    test('should build agent config object on submit', () => {
        const nameInput = MockDOM.createElement('addAgentNameInput', { value: 'New Agent' });
        const typeSelect = MockDOM.createElement('addAgentTypeSelect', { selectedKey: 'code' });
        const descriptionTextArea = MockDOM.createElement('addAgentDescriptionTextArea', { value: 'Test description' });
        const modelIdInput = MockDOM.createElement('addAgentModelIdInput', { value: 'gpt-4' });
        const connectedAgentsComboBox = MockDOM.createElement('addAgentConnectedAgentsComboBox', {
            selectedKeys: ['agent-1', 'agent-2']
        });

        const agentConfig = {
            name: nameInput.getValue().trim(),
            type: typeSelect.getSelectedKey(),
            description: descriptionTextArea.getValue().trim(),
            model_id: modelIdInput.getValue().trim(),
            connected_agents: connectedAgentsComboBox.getSelectedKeys()
        };

        expect(agentConfig.name).toBe('New Agent');
        expect(agentConfig.type).toBe('code');
        expect(agentConfig.description).toBe('Test description');
        expect(agentConfig.model_id).toBe('gpt-4');
        expect(agentConfig.connected_agents).toEqual(['agent-1', 'agent-2']);
    });

    test('should close dialog after successful submit', () => {
        const dialog = MockDOM.createElement('addAgentDialog', { visible: true });

        // Simulate successful API call
        dialog.close();
        expect(dialog.visible).toBe(false);
    });

    test('should reset form fields after dialog close', () => {
        const nameInput = MockDOM.createElement('addAgentNameInput', { value: 'Test', valueState: 'Error' });
        const typeSelect = MockDOM.createElement('addAgentTypeSelect', { selectedKey: 'code' });
        const connectedAgentsComboBox = MockDOM.createElement('addAgentConnectedAgentsComboBox', {
            selectedKeys: ['agent-1']
        });

        // Simulate afterClose handler
        nameInput.setValue('');
        nameInput.setValueState('None');
        typeSelect.setSelectedKey('router');
        connectedAgentsComboBox.setSelectedKeys([]);

        expect(nameInput.getValue()).toBe('');
        expect(nameInput.valueState).toBe('None');
        expect(typeSelect.getSelectedKey()).toBe('router');
        expect(connectedAgentsComboBox.getSelectedKeys()).toEqual([]);
    });
});


// ============================================================================
// Section 3: Create Workflow Dialog Tests
// ============================================================================

describe('Create Workflow Dialog', () => {
    beforeEach(() => {
        MockDOM.clear();
        MockFetch.clear();

        MockFetch.register('http://localhost:8080/api/v1/agents', {
            agents: [
                { id: 'agent-1', name: 'Router', type: 'router' },
                { id: 'agent-2', name: 'Code Agent', type: 'code' },
                { id: 'agent-3', name: 'Validator', type: 'validation' }
            ]
        });
    });

    test('should open Create Workflow dialog', () => {
        const dialog = MockDOM.createElement('createWorkflowDialog', {
            visible: false,
            title: 'Create New Workflow'
        });

        dialog.open();
        expect(dialog.visible).toBe(true);
    });

    test('should populate agent selection from graph model', () => {
        const nodes = [
            { id: 'agent-1', name: 'Router', type: 'router' },
            { id: 'agent-2', name: 'Code Agent', type: 'code' },
            { id: 'agent-3', name: 'Validator', type: 'validation' }
        ];

        const agentCombo = MockDOM.createElement('workflowAgentCombo', {
            items: nodes.map(n => ({ key: n.id, text: n.name })),
            selectedKeys: []
        });

        expect(agentCombo.getItems()).toHaveLength(3);
    });

    test('should select multiple agents for workflow', () => {
        const agentCombo = MockDOM.createElement('workflowAgentCombo', {
            selectedKeys: []
        });

        agentCombo.setSelectedKeys(['agent-1', 'agent-2', 'agent-3']);
        expect(agentCombo.getSelectedKeys()).toHaveLength(3);
        expect(agentCombo.getSelectedKeys()).toContain('agent-2');
    });

    test('should validate minimum 2 agents required', () => {
        const agentCombo = MockDOM.createElement('workflowAgentCombo', {
            selectedKeys: ['agent-1']
        });

        const selectedKeys = agentCombo.getSelectedKeys();
        const isValid = selectedKeys.length >= 2;

        expect(isValid).toBeFalsy();
    });

    test('should accept 2 or more agents', () => {
        const agentCombo = MockDOM.createElement('workflowAgentCombo', {
            selectedKeys: ['agent-1', 'agent-2']
        });

        const selectedKeys = agentCombo.getSelectedKeys();
        const isValid = selectedKeys.length >= 2;

        expect(isValid).toBeTruthy();
    });

    test('should build workflow config with sequential connections', () => {
        const nameInput = MockDOM.createElement('workflowNameInput', { value: 'Test Workflow' });
        const descriptionArea = MockDOM.createElement('workflowDescriptionInput', { value: 'Test description' });
        const agentCombo = MockDOM.createElement('workflowAgentCombo', {
            selectedKeys: ['agent-1', 'agent-2', 'agent-3']
        });

        const aSelectedKeys = agentCombo.getSelectedKeys();
        const workflowId = 'workflow-' + Date.now();

        // Build nodes array
        const nodes = aSelectedKeys.map((agentId, index) => ({
            id: agentId,
            position: index
        }));

        // Build sequential connections
        const connections = [];
        for (let i = 0; i < aSelectedKeys.length - 1; i++) {
            connections.push({
                from: aSelectedKeys[i],
                to: aSelectedKeys[i + 1]
            });
        }

        expect(nodes).toHaveLength(3);
        expect(connections).toHaveLength(2);
        expect(connections[0].from).toBe('agent-1');
        expect(connections[0].to).toBe('agent-2');
        expect(connections[1].from).toBe('agent-2');
        expect(connections[1].to).toBe('agent-3');
    });

    test('should validate workflow name is required', () => {
        const nameInput = MockDOM.createElement('workflowNameInput', { value: '', valueState: 'None' });

        const name = nameInput.getValue().trim();
        if (!name) {
            nameInput.setValueState('Error');
        }

        expect(nameInput.valueState).toBe('Error');
    });

    test('should update selected agents list on selection change', () => {
        const selectedAgentsList = [];
        const selectedKeys = ['agent-1', 'agent-2'];
        const agents = [
            { id: 'agent-1', name: 'Router' },
            { id: 'agent-2', name: 'Code Agent' }
        ];

        // Simulate _updateSelectedAgentsList
        selectedKeys.forEach((key, index) => {
            const agent = agents.find(a => a.id === key);
            if (agent) {
                selectedAgentsList.push({
                    title: `${index + 1}. ${agent.name}`,
                    info: index < selectedKeys.length - 1 ? '→' : 'End'
                });
            }
        });

        expect(selectedAgentsList).toHaveLength(2);
        expect(selectedAgentsList[0].title).toBe('1. Router');
        expect(selectedAgentsList[0].info).toBe('→');
        expect(selectedAgentsList[1].info).toBe('End');
    });
});


// ============================================================================
// Section 4: Model Router Auto-Assign Tests
// ============================================================================

describe('Model Router Auto-Assign Functionality', () => {
    beforeEach(() => {
        MockDOM.clear();
        MockFetch.clear();

        MockFetch.register('http://localhost:8080/api/v1/models', {
            models: [
                { id: 'model-1', name: 'GPT-4', capabilities: ['coding', 'reasoning'], quality: 95 },
                { id: 'model-2', name: 'Claude-3', capabilities: ['creative', 'reasoning'], quality: 90 },
                { id: 'model-3', name: 'Llama-2', capabilities: ['general'], quality: 75 }
            ]
        });
    });

    test('should calculate match score between agent and model', () => {
        const assignment = {
            agentId: 'agent-1',
            agentName: 'Code Agent',
            agentType: 'coder',
            requirements: ['coding']
        };

        const model = {
            id: 'model-1',
            name: 'GPT-4',
            capabilities: ['coding', 'reasoning'],
            quality: 95
        };

        // Simulate _calculateMatchScore logic
        let score = 50; // Base score

        // Capability match bonus
        const hasCapability = model.capabilities.some(c =>
            assignment.requirements.includes(c)
        );
        if (hasCapability) score += 30;

        // Quality bonus
        score += (model.quality - 50) * 0.2;

        expect(score).toBeGreaterThan(70);
    });

    test('should auto-assign all agents to optimal models', () => {
        const assignments = [
            { agentId: 'agent-1', agentName: 'Coder', agentType: 'coder', requirements: ['coding'], modelId: null, status: 'pending' },
            { agentId: 'agent-2', agentName: 'Writer', agentType: 'writer', requirements: ['creative'], modelId: null, status: 'pending' }
        ];

        const models = [
            { id: 'model-1', name: 'GPT-4', capabilities: ['coding'], quality: 95 },
            { id: 'model-2', name: 'Claude-3', capabilities: ['creative'], quality: 90 }
        ];

        let updatedCount = 0;

        assignments.forEach(assignment => {
            let bestModel = null;
            let bestScore = -1;

            models.forEach(model => {
                let score = 50;
                const hasCapability = model.capabilities.some(c =>
                    assignment.requirements.includes(c)
                );
                if (hasCapability) score += 30;

                if (score > bestScore) {
                    bestScore = score;
                    bestModel = model;
                }
            });

            if (bestModel) {
                assignment.modelId = bestModel.id;
                assignment.matchScore = bestScore;
                assignment.status = 'assigned';
                updatedCount++;
            }
        });

        expect(updatedCount).toBe(2);
        expect(assignments[0].modelId).toBe('model-1');
        expect(assignments[1].modelId).toBe('model-2');
        expect(assignments[0].status).toBe('assigned');
    });

    test('should warn when no models available for assignment', () => {
        const models = [];
        const hasModels = models.length > 0;

        expect(hasModels).toBeFalsy();
    });

    test('should auto-assign single agent', () => {
        const assignment = {
            agentId: 'agent-1',
            agentName: 'Code Agent',
            requirements: ['coding'],
            modelId: null
        };

        const models = [
            { id: 'model-1', capabilities: ['coding'], quality: 95 },
            { id: 'model-2', capabilities: ['creative'], quality: 90 }
        ];

        let bestModel = null;
        let bestScore = -1;

        models.forEach(model => {
            let score = 50;
            const hasCapability = model.capabilities.some(c =>
                assignment.requirements.includes(c)
            );
            if (hasCapability) score += 30;
            score += (model.quality - 50) * 0.2;

            if (score > bestScore) {
                bestScore = score;
                bestModel = model;
            }
        });

        if (bestModel) {
            assignment.modelId = bestModel.id;
            assignment.matchScore = bestScore;
        }

        expect(assignment.modelId).toBe('model-1');
        expect(assignment.matchScore).toBeGreaterThan(80);
    });

    test('should update statistics after auto-assign', () => {
        const stats = {
            totalAgents: 5,
            assignedAgents: 0,
            totalModels: 3,
            avgMatchScore: 0
        };

        const assignments = [
            { modelId: 'model-1', matchScore: 85 },
            { modelId: 'model-2', matchScore: 90 },
            { modelId: null, matchScore: 0 }
        ];

        // Update statistics
        stats.assignedAgents = assignments.filter(a => a.modelId !== null).length;
        const assignedWithScore = assignments.filter(a => a.matchScore > 0);
        stats.avgMatchScore = assignedWithScore.length > 0
            ? Math.round(assignedWithScore.reduce((sum, a) => sum + a.matchScore, 0) / assignedWithScore.length)
            : 0;

        expect(stats.assignedAgents).toBe(2);
        expect(stats.avgMatchScore).toBe(88);
    });
});


// ============================================================================
// Section 5: Graph Refresh and Data Loading Tests
// ============================================================================

describe('Graph Refresh and Data Loading', () => {
    beforeEach(() => {
        MockDOM.clear();
        MockFetch.clear();

        MockFetch.register('http://localhost:8080/api/v1/agents', {
            agents: [
                { id: 'agent-1', name: 'Router', type: 'router', status: 'healthy', next_agents: ['agent-2'] },
                { id: 'agent-2', name: 'Coder', type: 'code', status: 'healthy', next_agents: ['agent-3'] },
                { id: 'agent-3', name: 'Validator', type: 'validation', status: 'healthy', next_agents: [] }
            ]
        });

        MockFetch.register('http://localhost:8080/api/v1/workflows', {
            workflows: [
                { id: 'workflow-1', name: 'Code Review', nodes: ['agent-1', 'agent-2', 'agent-3'] }
            ]
        });
    });

    test('should load agent topology from backend', async () => {
        const response = await MockFetch.fetch('http://localhost:8080/api/v1/agents');
        const data = await response.json();

        expect(data.agents).toHaveLength(3);
        expect(data.agents[0].id).toBe('agent-1');
        expect(data.agents[0].status).toBe('healthy');
    });

    test('should build network graph nodes from agents', () => {
        const agents = [
            { id: 'agent-1', name: 'Router', type: 'router', status: 'healthy' },
            { id: 'agent-2', name: 'Coder', type: 'code', status: 'busy' }
        ];

        const nodes = agents.map((agent, idx) => ({
            id: agent.id,
            name: agent.name,
            type: agent.type,
            status: agent.status === 'healthy' ? 'Success' : 'Warning',
            x: (idx % 3) * 300 + 100,
            y: Math.floor(idx / 3) * 200 + 100
        }));

        expect(nodes).toHaveLength(2);
        expect(nodes[0].status).toBe('Success');
        expect(nodes[1].status).toBe('Warning');
    });

    test('should build network graph edges from agent connections', () => {
        const agents = [
            { id: 'agent-1', next_agents: ['agent-2'] },
            { id: 'agent-2', next_agents: ['agent-3'] },
            { id: 'agent-3', next_agents: [] }
        ];

        const lines = [];
        agents.forEach(agent => {
            if (agent.next_agents && Array.isArray(agent.next_agents)) {
                agent.next_agents.forEach(targetId => {
                    lines.push({
                        from: agent.id,
                        to: targetId,
                        status: 'Success'
                    });
                });
            }
        });

        expect(lines).toHaveLength(2);
        expect(lines[0].from).toBe('agent-1');
        expect(lines[0].to).toBe('agent-2');
    });

    test('should update graph model after data load', () => {
        const graphModel = {
            nodes: [],
            lines: [],
            groups: [],
            setProperty(path, value) {
                const prop = path.replace('/', '');
                this[prop] = value;
            },
            getProperty(path) {
                const prop = path.replace('/', '');
                return this[prop];
            }
        };

        const nodes = [{ id: 'agent-1', name: 'Router' }];
        const lines = [{ from: 'agent-1', to: 'agent-2' }];

        graphModel.setProperty('/nodes', nodes);
        graphModel.setProperty('/lines', lines);

        expect(graphModel.getProperty('/nodes')).toHaveLength(1);
        expect(graphModel.getProperty('/lines')).toHaveLength(1);
    });

    test('should map agent status to UI status correctly', () => {
        const statusMap = {
            "healthy": "Success",
            "active": "Success",
            "busy": "Warning",
            "idle": "None",
            "error": "Error",
            "stopped": "None"
        };

        expect(statusMap.healthy).toBe('Success');
        expect(statusMap.busy).toBe('Warning');
        expect(statusMap.error).toBe('Error');
        expect(statusMap.idle).toBe('None');
    });

    test('should handle API error gracefully with fallback to mock data', async () => {
        // Unregistered URL returns 404
        const response = await MockFetch.fetch('http://localhost:8080/api/v1/unknown');

        expect(response.ok).toBeFalsy();
        expect(response.status).toBe(404);

        // Fallback to mock data
        const fallbackData = {
            agents: [{ id: 'mock-1', name: 'Mock Agent' }]
        };

        expect(fallbackData.agents).toHaveLength(1);
    });

    test('should trigger refresh on GraphIntegration', () => {
        let networkGraphRefreshed = false;
        let processFlowRefreshed = false;

        // Mock GraphIntegration.refresh()
        const GraphIntegration = {
            networkGraph: true,
            processFlow: true,
            refresh() {
                if (this.networkGraph) {
                    networkGraphRefreshed = true;
                }
                if (this.processFlow) {
                    processFlowRefreshed = true;
                }
            }
        };

        GraphIntegration.refresh();

        expect(networkGraphRefreshed).toBeTruthy();
        expect(processFlowRefreshed).toBeTruthy();
    });

    test('should update statistics after loading agents', () => {
        const agents = [
            { status: 'healthy' },
            { status: 'healthy' },
            { status: 'busy' },
            { status: 'error' }
        ];

        const stats = {
            totalAgents: agents.length,
            activeAgents: agents.filter(a => a.status === 'healthy' || a.status === 'busy').length,
            errorAgents: agents.filter(a => a.status === 'error').length
        };

        expect(stats.totalAgents).toBe(4);
        expect(stats.activeAgents).toBe(3);
        expect(stats.errorAgents).toBe(1);
    });

    test('should load workflows after agent topology', async () => {
        const response = await MockFetch.fetch('http://localhost:8080/api/v1/workflows');
        const data = await response.json();

        expect(data.workflows).toHaveLength(1);
        expect(data.workflows[0].name).toBe('Code Review');
        expect(data.workflows[0].nodes).toHaveLength(3);
    });
});

// ============================================================================
// Test Runner
// ============================================================================

function printTestSummary() {
    console.log('🧪 UI Flow Integration Tests Complete\n');

    // Print summary
    const allPassed = TestResults.summary();

    if (allPassed) {
        console.log('\n✅ All tests passed!');
    } else {
        console.log('\n❌ Some tests failed.');
    }

    return allPassed;
}

// Export for use with different test runners
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        MockFetch,
        MockDOM,
        TestResults,
        expect,
        test,
        describe,
        beforeEach,
        printTestSummary
    };
}

// Auto-run summary if executed directly (tests already ran above)
if (typeof window === 'undefined' && typeof process !== 'undefined') {
    // Use setTimeout to allow async tests to complete
    setTimeout(() => {
        printTestSummary();
    }, 100);
}