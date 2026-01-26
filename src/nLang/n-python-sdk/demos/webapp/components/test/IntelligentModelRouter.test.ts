/**
 * IntelligentModelRouter Unit Tests
 * Tests for intelligent routing system for multi-model orchestration
 */

import { describe, test, expect, beforeEach } from 'bun:test';
import IntelligentModelRouter from '../ModelRouter/IntelligentModelRouter';
import { ModelProfile, ModelCapability, TaskCharacteristic, ModelRouterConfig } from '../ModelRouter/types';

// ============================================================================
// Test Fixtures
// ============================================================================

const createMockConfig = (overrides: Partial<ModelRouterConfig> = {}): ModelRouterConfig => ({
    strategy: 'balanced',
    enableFallback: true,
    maxFallbackAttempts: 3,
    enableLearning: false,
    learningRate: 0.1,
    ...overrides
});

const createMockModel = (overrides: Partial<ModelProfile> = {}): ModelProfile => ({
    id: 'test-model',
    name: 'Test Model',
    provider: 'test-provider',
    capabilities: [ModelCapability.REASONING],
    maxContextLength: 8000,
    costPerToken: 0.01,
    avgLatencyMs: 500,
    qualityScore: 80,
    ...overrides
});

// ============================================================================
// analyzeTask() Tests
// ============================================================================

describe('IntelligentModelRouter.analyzeTask()', () => {
    let router: IntelligentModelRouter;

    beforeEach(() => {
        router = new IntelligentModelRouter(createMockConfig());
    });

    test('identifies coding tasks correctly', () => {
        const task = router.analyzeTask('Please implement a function that sorts an array');
        expect(task.type).toBe('coding');
    });

    test('identifies math tasks correctly', () => {
        const task = router.analyzeTask('Calculate the derivative of x^2 + 3x');
        expect(task.type).toBe('math');
    });

    test('identifies reasoning tasks correctly', () => {
        const task = router.analyzeTask('Explain why climate change affects biodiversity');
        expect(task.type).toBe('reasoning');
    });

    test('identifies creative tasks correctly', () => {
        const task = router.analyzeTask('Write a short story about a robot');
        expect(task.type).toBe('creative');
    });

    test('identifies translation tasks correctly', () => {
        const task = router.analyzeTask('Translate this sentence to Spanish');
        expect(task.type).toBe('translation');
    });

    test('defaults to general for ambiguous tasks', () => {
        const task = router.analyzeTask('Hello, how are you today?');
        expect(task.type).toBe('general');
    });

    test('detects tool use requirements', () => {
        const task = router.analyzeTask('Search for the latest news about AI');
        expect(task.requiresToolUse).toBe(true);
    });

    test('detects vision requirements', () => {
        const task = router.analyzeTask('Look at this image and describe what you see');
        expect(task.requiresVision).toBe(true);
    });

    test('assigns complexity based on input length', () => {
        const shortTask = router.analyzeTask('Hi');
        const mediumTask = router.analyzeTask('a'.repeat(250));
        const longTask = router.analyzeTask('a'.repeat(600));

        expect(shortTask.complexity).toBe('simple');
        expect(mediumTask.complexity).toBe('moderate');
        expect(longTask.complexity).toBe('complex');
    });
});

// ============================================================================
// scoreModel() Tests
// ============================================================================

describe('IntelligentModelRouter.scoreModel()', () => {
    let router: IntelligentModelRouter;

    beforeEach(() => {
        router = new IntelligentModelRouter(createMockConfig());
    });

    test('gives higher score for capability match', () => {
        const codingModel = createMockModel({
            id: 'coder',
            capabilities: [ModelCapability.CODING]
        });
        const generalModel = createMockModel({
            id: 'general',
            capabilities: [ModelCapability.REASONING]
        });

        const codingTask: TaskCharacteristic = {
            type: 'coding',
            complexity: 'moderate',
            contextLength: 4000,
            requiresToolUse: false,
            requiresVision: false,
            latencyPriority: 'medium',
            costPriority: 'medium'
        };

        const codingScore = router.scoreModel(codingModel, codingTask);
        const generalScore = router.scoreModel(generalModel, codingTask);

        expect(codingScore).toBeGreaterThan(generalScore);
    });

    test('gives bonus for tool use capability when required', () => {
        const toolModel = createMockModel({
            id: 'tool-user',
            capabilities: [ModelCapability.REASONING, ModelCapability.TOOL_USE]
        });
        const noToolModel = createMockModel({
            id: 'no-tool',
            capabilities: [ModelCapability.REASONING]
        });

        const toolTask: TaskCharacteristic = {
            type: 'reasoning',
            complexity: 'moderate',
            contextLength: 4000,
            requiresToolUse: true,
            requiresVision: false,
            latencyPriority: 'medium',
            costPriority: 'medium'
        };

        const toolScore = router.scoreModel(toolModel, toolTask);
        const noToolScore = router.scoreModel(noToolModel, toolTask);

        expect(toolScore).toBeGreaterThan(noToolScore);
    });

    test('considers context length fit in scoring', () => {
        const largeContextModel = createMockModel({
            id: 'large-context',
            maxContextLength: 32000
        });
        const smallContextModel = createMockModel({
            id: 'small-context',
            maxContextLength: 2000
        });

        const largeTask: TaskCharacteristic = {
            type: 'general',
            complexity: 'complex',
            contextLength: 16000,
            requiresToolUse: false,
            requiresVision: false,
            latencyPriority: 'medium',
            costPriority: 'medium'
        };

        const largeScore = router.scoreModel(largeContextModel, largeTask);
        const smallScore = router.scoreModel(smallContextModel, largeTask);

        expect(largeScore).toBeGreaterThan(smallScore);
    });

    test('prioritizes low latency when high priority', () => {
        const fastModel = createMockModel({
            id: 'fast',
            avgLatencyMs: 200
        });
        const slowModel = createMockModel({
            id: 'slow',
            avgLatencyMs: 3000
        });

        const latencyTask: TaskCharacteristic = {
            type: 'general',
            complexity: 'simple',
            contextLength: 1000,
            requiresToolUse: false,
            requiresVision: false,
            latencyPriority: 'high',
            costPriority: 'medium'
        };

        const fastScore = router.scoreModel(fastModel, latencyTask);
        const slowScore = router.scoreModel(slowModel, latencyTask);

        expect(fastScore).toBeGreaterThan(slowScore);
    });

    test('prioritizes low cost when high priority', () => {
        const cheapModel = createMockModel({
            id: 'cheap',
            costPerToken: 0.0001
        });
        const expensiveModel = createMockModel({
            id: 'expensive',
            costPerToken: 0.5
        });

        const costTask: TaskCharacteristic = {
            type: 'general',
            complexity: 'simple',
            contextLength: 1000,
            requiresToolUse: false,
            requiresVision: false,
            latencyPriority: 'medium',
            costPriority: 'high'
        };

        const cheapScore = router.scoreModel(cheapModel, costTask);
        const expensiveScore = router.scoreModel(expensiveModel, costTask);

        expect(cheapScore).toBeGreaterThan(expensiveScore);
    });
});

// ============================================================================
// selectModel() Tests
// ============================================================================

describe('IntelligentModelRouter.selectModel()', () => {
    let router: IntelligentModelRouter;

    beforeEach(() => {
        router = new IntelligentModelRouter(createMockConfig());
    });

    test('returns best model with highest score', () => {
        const codingModel = createMockModel({
            id: 'coder',
            name: 'Coder Model',
            capabilities: [ModelCapability.CODING],
            qualityScore: 95
        });
        const generalModel = createMockModel({
            id: 'general',
            name: 'General Model',
            capabilities: [ModelCapability.REASONING],
            qualityScore: 70
        });

        router.registerModel(codingModel);
        router.registerModel(generalModel);

        const codingTask: TaskCharacteristic = {
            type: 'coding',
            complexity: 'moderate',
            contextLength: 4000,
            requiresToolUse: false,
            requiresVision: false,
            latencyPriority: 'medium',
            costPriority: 'medium'
        };

        const decision = router.selectModel(codingTask);

        expect(decision.selectedModel.id).toBe('coder');
        expect(decision.score).toBeGreaterThan(0);
    });

    test('includes alternatives in decision', () => {
        router.registerModel(createMockModel({ id: 'model-1', name: 'Model 1' }));
        router.registerModel(createMockModel({ id: 'model-2', name: 'Model 2' }));
        router.registerModel(createMockModel({ id: 'model-3', name: 'Model 3' }));

        const task: TaskCharacteristic = {
            type: 'general',
            complexity: 'simple',
            contextLength: 1000,
            requiresToolUse: false,
            requiresVision: false,
            latencyPriority: 'medium',
            costPriority: 'medium'
        };

        const decision = router.selectModel(task);

        expect(decision.alternatives.length).toBeGreaterThan(0);
        expect(decision.fallbackChain.length).toBeGreaterThan(0);
    });

    test('includes reasoning in decision', () => {
        router.registerModel(createMockModel({ id: 'test-model' }));

        const task: TaskCharacteristic = {
            type: 'reasoning',
            complexity: 'complex',
            contextLength: 8000,
            requiresToolUse: true,
            requiresVision: false,
            latencyPriority: 'medium',
            costPriority: 'medium'
        };

        const decision = router.selectModel(task);

        expect(decision.reasoning.length).toBeGreaterThan(0);
        expect(decision.reasoning.some(r => r.includes('reasoning'))).toBe(true);
    });

    test('throws error when no models registered', () => {
        const task: TaskCharacteristic = {
            type: 'general',
            complexity: 'simple',
            contextLength: 1000,
            requiresToolUse: false,
            requiresVision: false,
            latencyPriority: 'medium',
            costPriority: 'medium'
        };

        expect(() => router.selectModel(task)).toThrow('No models registered for routing');
    });
});

// ============================================================================
// assignModelToAgent() Tests
// ============================================================================

describe('IntelligentModelRouter.assignModelToAgent()', () => {
    let router: IntelligentModelRouter;

    beforeEach(() => {
        router = new IntelligentModelRouter(createMockConfig());
        // Register models with different capabilities
        router.registerModel(createMockModel({
            id: 'coder-model',
            name: 'Coder',
            capabilities: [ModelCapability.CODING]
        }));
        router.registerModel(createMockModel({
            id: 'math-model',
            name: 'Mathematician',
            capabilities: [ModelCapability.MATH]
        }));
        router.registerModel(createMockModel({
            id: 'writer-model',
            name: 'Writer',
            capabilities: [ModelCapability.CREATIVE]
        }));
        router.registerModel(createMockModel({
            id: 'translator-model',
            name: 'Translator',
            capabilities: [ModelCapability.TRANSLATION]
        }));
        router.registerModel(createMockModel({
            id: 'analyst-model',
            name: 'Analyst',
            capabilities: [ModelCapability.REASONING, ModelCapability.TOOL_USE]
        }));
    });

    test('maps coder agent to coding task type', () => {
        const assignment = router.assignModelToAgent('agent-1', 'coder');
        expect(assignment.agentType).toBe('coder');
        expect(assignment.assignedModel.capabilities).toContain(ModelCapability.CODING);
    });

    test('maps mathematician agent to math task type', () => {
        const assignment = router.assignModelToAgent('agent-2', 'mathematician');
        expect(assignment.agentType).toBe('mathematician');
        expect(assignment.assignedModel.capabilities).toContain(ModelCapability.MATH);
    });

    test('maps writer agent to creative task type', () => {
        const assignment = router.assignModelToAgent('agent-3', 'writer');
        expect(assignment.agentType).toBe('writer');
        expect(assignment.assignedModel.capabilities).toContain(ModelCapability.CREATIVE);
    });

    test('maps translator agent to translation task type', () => {
        const assignment = router.assignModelToAgent('agent-4', 'translator');
        expect(assignment.agentType).toBe('translator');
        expect(assignment.assignedModel.capabilities).toContain(ModelCapability.TRANSLATION);
    });

    test('maps analyst agent with tool use requirement', () => {
        const assignment = router.assignModelToAgent('agent-5', 'analyst');
        expect(assignment.agentType).toBe('analyst');
        expect(assignment.assignedModel.capabilities).toContain(ModelCapability.TOOL_USE);
    });

    test('returns auto-assigned flag as true', () => {
        const assignment = router.assignModelToAgent('agent-1', 'coder');
        expect(assignment.autoAssigned).toBe(true);
    });

    test('initializes performance metrics correctly', () => {
        const assignment = router.assignModelToAgent('agent-1', 'coder');
        expect(assignment.performanceMetrics.successRate).toBe(0);
        expect(assignment.performanceMetrics.avgLatency).toBe(0);
        expect(assignment.performanceMetrics.totalRequests).toBe(0);
    });

    test('sets lastUpdated to current date', () => {
        const before = new Date();
        const assignment = router.assignModelToAgent('agent-1', 'coder');
        const after = new Date();

        expect(assignment.lastUpdated.getTime()).toBeGreaterThanOrEqual(before.getTime());
        expect(assignment.lastUpdated.getTime()).toBeLessThanOrEqual(after.getTime());
    });
});

// ============================================================================
// recordOutcome() Tests
// ============================================================================

describe('IntelligentModelRouter.recordOutcome()', () => {
    let router: IntelligentModelRouter;

    beforeEach(() => {
        router = new IntelligentModelRouter(createMockConfig());
        router.registerModel(createMockModel({ id: 'test-model' }));
    });

    test('increments successful routings on success', () => {
        const task: TaskCharacteristic = {
            type: 'general',
            complexity: 'simple',
            contextLength: 1000,
            requiresToolUse: false,
            requiresVision: false,
            latencyPriority: 'medium',
            costPriority: 'medium'
        };

        const decision = router.selectModel(task);
        router.recordOutcome(decision, true, 100);

        const stats = router.getRoutingStats();
        expect(stats.successfulRoutings).toBe(1);
        expect(stats.failedRoutings).toBe(0);
    });

    test('increments failed routings on failure', () => {
        const task: TaskCharacteristic = {
            type: 'general',
            complexity: 'simple',
            contextLength: 1000,
            requiresToolUse: false,
            requiresVision: false,
            latencyPriority: 'medium',
            costPriority: 'medium'
        };

        const decision = router.selectModel(task);
        router.recordOutcome(decision, false, 200);

        const stats = router.getRoutingStats();
        expect(stats.successfulRoutings).toBe(0);
        expect(stats.failedRoutings).toBe(1);
    });

    test('updates average latency correctly', () => {
        const task: TaskCharacteristic = {
            type: 'general',
            complexity: 'simple',
            contextLength: 1000,
            requiresToolUse: false,
            requiresVision: false,
            latencyPriority: 'medium',
            costPriority: 'medium'
        };

        const decision = router.selectModel(task);
        router.recordOutcome(decision, true, 100);
        router.recordOutcome(decision, true, 200);
        router.recordOutcome(decision, true, 300);

        const stats = router.getRoutingStats();
        expect(stats.averageLatencyMs).toBe(200); // (100 + 200 + 300) / 3
    });

    test('tracks multiple outcomes correctly', () => {
        const task: TaskCharacteristic = {
            type: 'general',
            complexity: 'simple',
            contextLength: 1000,
            requiresToolUse: false,
            requiresVision: false,
            latencyPriority: 'medium',
            costPriority: 'medium'
        };

        const decision = router.selectModel(task);
        router.recordOutcome(decision, true, 100);
        router.recordOutcome(decision, true, 150);
        router.recordOutcome(decision, false, 500);
        router.recordOutcome(decision, true, 200);

        const stats = router.getRoutingStats();
        expect(stats.successfulRoutings).toBe(3);
        expect(stats.failedRoutings).toBe(1);
    });

    test('tracks model usage in stats', () => {
        const task: TaskCharacteristic = {
            type: 'general',
            complexity: 'simple',
            contextLength: 1000,
            requiresToolUse: false,
            requiresVision: false,
            latencyPriority: 'medium',
            costPriority: 'medium'
        };

        router.selectModel(task);
        router.selectModel(task);
        router.selectModel(task);

        const stats = router.getRoutingStats();
        expect(stats.totalRoutingDecisions).toBe(3);
        expect(stats.modelUsage.get('test-model')).toBe(3);
    });
});
