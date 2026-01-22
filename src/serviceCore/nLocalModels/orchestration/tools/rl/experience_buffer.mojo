"""
Experience Buffer for KTO Learning
Manages desirable and undesirable experiences for balanced training

Features:
- Separate buffers for successful/failed executions
- Balanced sampling for KTO
- Efficient circular buffer
- Experience replay
"""

from collections import List
from ..state import OrchestrationState, StateTransition
from .kto_policy import ToolAction


# ============================================================================
# Experience Types
# ============================================================================

@value
struct Experience:
    """
    Single experience tuple for RL learning
    
    Represents one workflow execution step with outcome
    """
    var state: OrchestrationState
    var action: ToolAction
    var next_state: OrchestrationState
    var reward: Float32
    var done: Bool  # Whether workflow completed
    var success: Bool  # Whether action succeeded (desirable/undesirable)
    var timestamp: Float64
    
    fn __init__(
        inout self,
        state: OrchestrationState,
        action: ToolAction,
        next_state: OrchestrationState,
        reward: Float32,
        done: Bool,
        success: Bool,
        timestamp: Float64 = 0.0
    ):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.done = done
        self.success = success
        self.timestamp = timestamp


# ============================================================================
# Experience Buffer
# ============================================================================

struct ExperienceBuffer:
    """
    Replay buffer with separate desirable/undesirable experiences
    
    Key for KTO:
    - Desirable: Successful tool executions
    - Undesirable: Failed tool executions
    - Balanced sampling for training
    
    Features:
    - Efficient circular buffers
    - Automatic categorization
    - Balanced batch sampling
    - Memory-efficient storage
    """
    var desirable_buffer: List[Experience]
    var undesirable_buffer: List[Experience]
    var max_size_per_category: Int
    var desirable_index: Int  # Circular buffer index
    var undesirable_index: Int
    var total_experiences: Int
    var desirable_count: Int
    var undesirable_count: Int
    
    fn __init__(
        inout self,
        max_size_per_category: Int = 10000
    ):
        """
        Initialize experience buffer
        
        Args:
            max_size_per_category: Max experiences per category (desirable/undesirable)
                                  Total capacity = 2 * max_size_per_category
        """
        self.desirable_buffer = List[Experience]()
        self.undesirable_buffer = List[Experience]()
        self.max_size_per_category = max_size_per_category
        self.desirable_index = 0
        self.undesirable_index = 0
        self.total_experiences = 0
        self.desirable_count = 0
        self.undesirable_count = 0
    
    
    # ========================================================================
    # Add Experiences
    # ========================================================================
    
    fn add(inout self, experience: Experience):
        """
        Add experience to appropriate buffer
        
        Automatically categorizes as desirable/undesirable based on success
        Uses circular buffer to maintain max_size
        """
        if experience.success:
            self._add_desirable(experience)
        else:
            self._add_undesirable(experience)
        
        self.total_experiences += 1
    
    fn _add_desirable(inout self, experience: Experience):
        """Add desirable experience (successful execution)"""
        if len(self.desirable_buffer) < self.max_size_per_category:
            # Buffer not full - append
            self.desirable_buffer.append(experience)
        else:
            # Buffer full - circular overwrite
            self.desirable_buffer[self.desirable_index % self.max_size_per_category] = experience
        
        self.desirable_index += 1
        self.desirable_count += 1
    
    fn _add_undesirable(inout self, experience: Experience):
        """Add undesirable experience (failed execution)"""
        if len(self.undesirable_buffer) < self.max_size_per_category:
            self.undesirable_buffer.append(experience)
        else:
            self.undesirable_buffer[self.undesirable_index % self.max_size_per_category] = experience
        
        self.undesirable_index += 1
        self.undesirable_count += 1
    
    fn add_batch(inout self, experiences: List[Experience]):
        """Add multiple experiences at once"""
        for i in range(len(experiences)):
            self.add(experiences[i])
    
    
    # ========================================================================
    # Sampling Methods
    # ========================================================================
    
    fn sample_balanced(
        self,
        batch_size: Int = 32
    ) -> BalancedBatch:
        """
        Sample balanced batch for KTO training
        
        Returns batch_size/2 desirable + batch_size/2 undesirable experiences
        This ensures balanced learning as required by KTO
        
        Args:
            batch_size: Total batch size (must be even)
            
        Returns:
            BalancedBatch with equal desirable/undesirable samples
        """
        let n_per_category = batch_size // 2
        
        # Sample desirable experiences
        let desirable_samples = self._sample_from_buffer(
            self.desirable_buffer,
            n_per_category
        )
        
        # Sample undesirable experiences
        let undesirable_samples = self._sample_from_buffer(
            self.undesirable_buffer,
            n_per_category
        )
        
        return BalancedBatch(
            desirable=desirable_samples,
            undesirable=undesirable_samples
        )
    
    fn _sample_from_buffer(
        self,
        buffer: List[Experience],
        n_samples: Int
    ) -> List[Experience]:
        """
        Sample n_samples from buffer
        
        Uses random sampling without replacement
        In production: proper random sampling
        For now: take first n samples (simplified)
        """
        var samples = List[Experience]()
        let buffer_size = len(buffer)
        
        if buffer_size == 0:
            return samples
        
        # Take min(n_samples, buffer_size) samples
        let actual_n = min(n_samples, buffer_size)
        
        for i in range(actual_n):
            # Simplified: sequential sampling
            # Production: random indices without replacement
            samples.append(buffer[i])
        
        return samples
    
    fn sample_desirable(self, n: Int) -> List[Experience]:
        """Sample only desirable experiences"""
        return self._sample_from_buffer(self.desirable_buffer, n)
    
    fn sample_undesirable(self, n: Int) -> List[Experience]:
        """Sample only undesirable experiences"""
        return self._sample_from_buffer(self.undesirable_buffer, n)
    
    
    # ========================================================================
    # Query Methods
    # ========================================================================
    
    fn size(self) -> Int:
        """Get total number of stored experiences"""
        return len(self.desirable_buffer) + len(self.undesirable_buffer)
    
    fn desirable_size(self) -> Int:
        """Get number of desirable experiences"""
        return len(self.desirable_buffer)
    
    fn undesirable_size(self) -> Int:
        """Get number of undesirable experiences"""
        return len(self.undesirable_buffer)
    
    fn is_empty(self) -> Bool:
        """Check if buffer is empty"""
        return self.size() == 0
    
    fn can_sample(self, batch_size: Int) -> Bool:
        """
        Check if we can sample a balanced batch
        
        Requires at least batch_size/2 samples in each category
        """
        let required_per_category = batch_size // 2
        return (
            len(self.desirable_buffer) >= required_per_category and
            len(self.undesirable_buffer) >= required_per_category
        )
    
    fn get_balance_ratio(self) -> Float32:
        """
        Get ratio of desirable to undesirable experiences
        
        Ideal ratio is 1.0 (balanced)
        """
        let n_des = len(self.desirable_buffer)
        let n_undes = len(self.undesirable_buffer)
        
        if n_undes == 0:
            return Float32.MAX if n_des > 0 else 1.0
        
        return Float32(n_des) / Float32(n_undes)
    
    fn get_statistics(self) -> BufferStatistics:
        """Get comprehensive buffer statistics"""
        return BufferStatistics(
            total_experiences=self.total_experiences,
            desirable_count=self.desirable_count,
            undesirable_count=self.undesirable_count,
            current_desirable_size=len(self.desirable_buffer),
            current_undesirable_size=len(self.undesirable_buffer),
            balance_ratio=self.get_balance_ratio(),
            capacity_per_category=self.max_size_per_category
        )
    
    
    # ========================================================================
    # Maintenance Methods
    # ========================================================================
    
    fn clear(inout self):
        """Clear all experiences"""
        self.desirable_buffer = List[Experience]()
        self.undesirable_buffer = List[Experience]()
        self.desirable_index = 0
        self.undesirable_index = 0
        self.total_experiences = 0
        self.desirable_count = 0
        self.undesirable_count = 0
    
    fn clear_desirable(inout self):
        """Clear only desirable experiences"""
        self.desirable_buffer = List[Experience]()
        self.desirable_index = 0
    
    fn clear_undesirable(inout self):
        """Clear only undesirable experiences"""
        self.undesirable_buffer = List[Experience]()
        self.undesirable_index = 0


# ============================================================================
# Batch Types
# ============================================================================

@value
struct BalancedBatch:
    """
    Balanced batch of desirable and undesirable experiences
    
    Used for KTO training to ensure equal representation
    """
    var desirable: List[Experience]
    var undesirable: List[Experience]
    
    fn __init__(
        inout self,
        desirable: List[Experience],
        undesirable: List[Experience]
    ):
        self.desirable = desirable
        self.undesirable = undesirable
    
    fn total_size(self) -> Int:
        """Get total batch size"""
        return len(self.desirable) + len(self.undesirable)
    
    fn is_balanced(self) -> Bool:
        """Check if batch is perfectly balanced"""
        return len(self.desirable) == len(self.undesirable)


@value
struct BufferStatistics:
    """Statistics about experience buffer"""
    var total_experiences: Int  # Total experiences added (including overwrites)
    var desirable_count: Int  # Total desirable added
    var undesirable_count: Int  # Total undesirable added
    var current_desirable_size: Int  # Current desirable buffer size
    var current_undesirable_size: Int  # Current undesirable buffer size
    var balance_ratio: Float32  # Desirable/Undesirable ratio
    var capacity_per_category: Int  # Max capacity per category
    
    fn __init__(
        inout self,
        total_experiences: Int = 0,
        desirable_count: Int = 0,
        undesirable_count: Int = 0,
        current_desirable_size: Int = 0,
        current_undesirable_size: Int = 0,
        balance_ratio: Float32 = 1.0,
        capacity_per_category: Int = 10000
    ):
        self.total_experiences = total_experiences
        self.desirable_count = desirable_count
        self.undesirable_count = undesirable_count
        self.current_desirable_size = current_desirable_size
        self.current_undesirable_size = current_undesirable_size
        self.balance_ratio = balance_ratio
        self.capacity_per_category = capacity_per_category


# ============================================================================
# Utility Functions
# ============================================================================

fn min(a: Int, b: Int) -> Int:
    """Minimum of two integers"""
    return a if a < b else b
