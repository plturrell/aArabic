/**
 * ModelRouter Type Definitions
 * Intelligent model routing type system for dynamic model selection
 */

// ============================================================================
// Enums
// ============================================================================

/**
 * Capabilities that a model can support
 */
export enum ModelCapability {
    CODING = 'CODING',
    MATH = 'MATH',
    REASONING = 'REASONING',
    CREATIVE = 'CREATIVE',
    TRANSLATION = 'TRANSLATION',
    RAG = 'RAG',
    VISION = 'VISION',
    AUDIO = 'AUDIO',
    TOOL_USE = 'TOOL_USE',
    LONG_CONTEXT = 'LONG_CONTEXT',
    FAST_INFERENCE = 'FAST_INFERENCE'
}

// ============================================================================
// Core Types
// ============================================================================

/**
 * Routing strategy options for model selection
 */
export type RoutingStrategy =
    | 'capability_match'
    | 'cost_optimized'
    | 'latency_optimized'
    | 'quality_optimized'
    | 'balanced'
    | 'rl_learned';

// ============================================================================
// Interfaces
// ============================================================================

/**
 * Characteristics of a task to be routed
 */
export interface TaskCharacteristic {
    type: 'coding' | 'math' | 'reasoning' | 'creative' | 'translation' | 'rag' | 'general';
    complexity: 'simple' | 'moderate' | 'complex';
    contextLength: number;
    requiresToolUse: boolean;
    requiresVision: boolean;
    latencyPriority: 'low' | 'medium' | 'high';
    costPriority: 'low' | 'medium' | 'high';
}

/**
 * Profile describing a model's characteristics and capabilities
 */
export interface ModelProfile {
    id: string;
    name: string;
    provider: string;
    capabilities: ModelCapability[];
    maxContextLength: number;
    costPerToken: number;
    avgLatencyMs: number;
    /** Quality score from 0-100 */
    qualityScore: number;
    quantization?: string;
    architecture?: string;
}

/**
 * Result of a routing decision
 */
export interface RoutingDecision {
    selectedModel: ModelProfile;
    score: number;
    reasoning: string[];
    alternatives: { model: ModelProfile; score: number }[];
    fallbackChain: string[];
}

/**
 * Model assignment for a specific agent
 */
export interface AgentModelAssignment {
    agentId: string;
    agentName: string;
    agentType: string;
    assignedModel: ModelProfile;
    autoAssigned: boolean;
    overrideReason?: string;
    lastUpdated: Date;
    performanceMetrics: {
        successRate: number;
        avgLatency: number;
        totalRequests: number;
    };
}

/**
 * Configuration for the model router
 */
export interface ModelRouterConfig {
    strategy: RoutingStrategy;
    enableFallback: boolean;
    maxFallbackAttempts: number;
    enableLearning: boolean;
    learningRate: number;
}

