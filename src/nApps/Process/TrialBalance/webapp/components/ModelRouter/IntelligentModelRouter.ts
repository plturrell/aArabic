/**
 * IntelligentModelRouter
 * Intelligent routing system for multi-model orchestration
 */

import {
    ModelRouterConfig,
    ModelProfile,
    TaskCharacteristic,
    RoutingDecision,
    AgentModelAssignment,
    ModelCapability
} from './types';

interface OutcomeRecord {
    decision: RoutingDecision;
    success: boolean;
    latency: number;
    timestamp: number;
}

interface RoutingStats {
    totalRoutingDecisions: number;
    successfulRoutings: number;
    failedRoutings: number;
    averageLatencyMs: number;
    modelUsage: Map<string, number>;
    fallbacksUsed: number;
}

class IntelligentModelRouter {
    private config: ModelRouterConfig;
    private models: Map<string, ModelProfile> = new Map();
    private outcomes: OutcomeRecord[] = [];
    private stats: RoutingStats;

    constructor(config: ModelRouterConfig) {
        this.config = config;
        this.stats = {
            totalRoutingDecisions: 0,
            successfulRoutings: 0,
            failedRoutings: 0,
            averageLatencyMs: 0,
            modelUsage: new Map(),
            fallbacksUsed: 0
        };
    }

    registerModel(model: ModelProfile): void {
        this.models.set(model.id, model);
    }

    analyzeTask(input: string, context?: any): TaskCharacteristic {
        const lowerInput = input.toLowerCase();

        // Detect task types
        const isCoding = /\b(code|function|class|implement|debug|fix bug)\b/.test(lowerInput);
        const isMath = /\b(calculate|solve|equation|math|prove)\b/.test(lowerInput);
        const isReasoning = /\b(explain|why|analyze|compare|evaluate)\b/.test(lowerInput);
        const isCreative = /\b(write|story|poem|creative|imagine)\b/.test(lowerInput);
        const isTranslation = /\b(translate|language|convert to)\b/.test(lowerInput);
        const requiresToolUse = /\b(search|browse|execute|run)\b/.test(lowerInput);
        const requiresVision = /\b(image|picture|photo|screenshot|look at)\b/.test(lowerInput);

        // Determine primary type
        let type: TaskCharacteristic['type'] = 'general';
        if (isCoding) type = 'coding';
        else if (isMath) type = 'math';
        else if (isReasoning) type = 'reasoning';
        else if (isCreative) type = 'creative';
        else if (isTranslation) type = 'translation';

        // Estimate complexity based on input length and structure
        let complexity: TaskCharacteristic['complexity'] = 'simple';
        if (input.length > 500 || input.includes('\n\n')) {
            complexity = 'complex';
        } else if (input.length > 200) {
            complexity = 'moderate';
        }

        // Estimate context length needed
        const contextLength = Math.max(input.length * 4, context?.estimatedTokens || 1000);

        return {
            type,
            complexity,
            contextLength,
            requiresToolUse,
            requiresVision,
            latencyPriority: context?.latencyPriority || 'medium',
            costPriority: context?.costPriority || 'medium'
        };
    }

    scoreModel(model: ModelProfile, task: TaskCharacteristic): number {
        let score = 0;

        // +30 for capability match
        const capabilityMap: Record<string, ModelCapability> = {
            coding: ModelCapability.CODING,
            math: ModelCapability.MATH,
            reasoning: ModelCapability.REASONING,
            creative: ModelCapability.CREATIVE,
            translation: ModelCapability.TRANSLATION,
            rag: ModelCapability.RAG
        };
        const requiredCapability = capabilityMap[task.type];
        if (requiredCapability && model.capabilities.includes(requiredCapability)) {
            score += 30;
        }
        if (task.requiresToolUse && model.capabilities.includes(ModelCapability.TOOL_USE)) {
            score += 10;
        }

        // +20 for context length fit
        if (model.maxContextLength >= task.contextLength) {
            score += 20;
        } else {
            score += Math.max(0, 20 - ((task.contextLength - model.maxContextLength) / 1000));
        }

        // +20 for latency match (if high priority)
        if (task.latencyPriority === 'high') {
            if (model.avgLatencyMs < 500) score += 20;
            else if (model.avgLatencyMs < 1000) score += 15;
            else if (model.avgLatencyMs < 2000) score += 10;
        } else {
            score += 10; // Base latency score for non-priority
        }

        // +20 for cost match (if high priority)
        if (task.costPriority === 'high') {
            if (model.costPerToken < 0.001) score += 20;
            else if (model.costPerToken < 0.01) score += 15;
            else if (model.costPerToken < 0.1) score += 10;
        } else {
            score += 10; // Base cost score for non-priority
        }

        // +10 for quality score
        score += (model.qualityScore / 100) * 10;

        return Math.min(100, Math.max(0, score));
    }

    selectModel(task: TaskCharacteristic): RoutingDecision {
        const scoredModels: { model: ModelProfile; score: number }[] = [];

        this.models.forEach((model) => {
            const score = this.scoreModel(model, task);
            scoredModels.push({ model, score });
        });

        // Sort by score descending
        scoredModels.sort((a, b) => b.score - a.score);

        if (scoredModels.length === 0) {
            throw new Error('No models registered for routing');
        }

        const selected = scoredModels[0];
        const fallbackChain = scoredModels.slice(1, 4).map(s => s.model.id);

        // Build reasoning
        const reasoning: string[] = [];
        reasoning.push(`Selected ${selected.model.name} with score ${selected.score.toFixed(1)}`);
        reasoning.push(`Task type: ${task.type}, complexity: ${task.complexity}`);
        if (task.requiresToolUse) {
            reasoning.push('Task requires tool use capability');
        }

        this.stats.totalRoutingDecisions++;
        const usage = this.stats.modelUsage.get(selected.model.id) || 0;
        this.stats.modelUsage.set(selected.model.id, usage + 1);

        return {
            selectedModel: selected.model,
            score: selected.score,
            reasoning,
            alternatives: scoredModels.slice(1, 4),
            fallbackChain
        };
    }

    assignModelToAgent(agentId: string, agentType: string): AgentModelAssignment {
        // Create a task characteristic based on agent type
        const agentTypeToTask: Record<string, TaskCharacteristic['type']> = {
            coder: 'coding',
            mathematician: 'math',
            analyst: 'reasoning',
            writer: 'creative',
            translator: 'translation',
            researcher: 'rag'
        };

        const taskType = agentTypeToTask[agentType.toLowerCase()] || 'general';

        const task: TaskCharacteristic = {
            type: taskType,
            complexity: 'moderate',
            contextLength: 4000,
            requiresToolUse: ['researcher', 'analyst'].includes(agentType.toLowerCase()),
            requiresVision: false,
            latencyPriority: 'medium',
            costPriority: 'medium'
        };

        const decision = this.selectModel(task);

        return {
            agentId,
            agentName: agentId,
            agentType,
            assignedModel: decision.selectedModel,
            autoAssigned: true,
            lastUpdated: new Date(),
            performanceMetrics: {
                successRate: 0,
                avgLatency: 0,
                totalRequests: 0
            }
        };
    }

    recordOutcome(decision: RoutingDecision, success: boolean, latency: number): void {
        const record: OutcomeRecord = {
            decision,
            success,
            latency,
            timestamp: Date.now()
        };
        this.outcomes.push(record);

        // Update stats
        if (success) {
            this.stats.successfulRoutings++;
        } else {
            this.stats.failedRoutings++;
        }

        // Update average latency
        const totalLatency = this.stats.averageLatencyMs * (this.outcomes.length - 1) + latency;
        this.stats.averageLatencyMs = totalLatency / this.outcomes.length;

        // RL learning: adjust model scores based on outcomes (placeholder for future)
        if (this.config.enableLearning) {
            // Future: Implement reinforcement learning updates here
        }
    }

    getRoutingStats(): RoutingStats {
        return { ...this.stats };
    }
}

export default IntelligentModelRouter;

