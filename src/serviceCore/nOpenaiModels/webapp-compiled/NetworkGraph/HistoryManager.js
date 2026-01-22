/**
 * HistoryManager - Undo/Redo functionality
 * Command pattern for reversible operations
 */
export class HistoryManager {
    constructor() {
        this.undoStack = [];
        this.redoStack = [];
        this.maxHistorySize = 50;
        // Callbacks
        this.onHistoryChange = null;
    }
    // ========================================================================
    // Command Recording
    // ========================================================================
    recordNodeAdd(node, addFn, removeFn) {
        const command = {
            type: 'node_add',
            execute: addFn,
            undo: removeFn,
            data: { node }
        };
        this.executeAndRecord(command);
    }
    recordNodeRemove(node, removeFn, addFn) {
        const command = {
            type: 'node_remove',
            execute: removeFn,
            undo: addFn,
            data: { node }
        };
        this.executeAndRecord(command);
    }
    recordNodeMove(nodeId, oldPos, newPos, moveFn) {
        const command = {
            type: 'node_move',
            execute: () => moveFn(newPos),
            undo: () => moveFn(oldPos),
            data: { nodeId, oldPos, newPos }
        };
        this.executeAndRecord(command);
    }
    recordEdgeAdd(edge, addFn, removeFn) {
        const command = {
            type: 'edge_add',
            execute: addFn,
            undo: removeFn,
            data: { edge }
        };
        this.executeAndRecord(command);
    }
    recordEdgeRemove(edge, removeFn, addFn) {
        const command = {
            type: 'edge_remove',
            execute: removeFn,
            undo: addFn,
            data: { edge }
        };
        this.executeAndRecord(command);
    }
    recordStatusChange(nodeId, oldStatus, newStatus, changeFn) {
        const command = {
            type: 'status_change',
            execute: () => changeFn(newStatus),
            undo: () => changeFn(oldStatus),
            data: { nodeId, oldStatus, newStatus }
        };
        this.executeAndRecord(command);
    }
    recordBatchOperation(operations, reverseOperations) {
        const command = {
            type: 'batch',
            execute: () => operations.forEach(op => op()),
            undo: () => reverseOperations.reverse().forEach(op => op()),
            data: { count: operations.length }
        };
        this.executeAndRecord(command);
    }
    // ========================================================================
    // Execution
    // ========================================================================
    executeAndRecord(command) {
        // Execute the command
        command.execute();
        // Add to undo stack
        this.undoStack.push(command);
        // Clear redo stack
        this.redoStack = [];
        // Limit stack size
        if (this.undoStack.length > this.maxHistorySize) {
            this.undoStack.shift();
        }
        this.notifyChange();
    }
    // ========================================================================
    // Undo/Redo
    // ========================================================================
    undo() {
        if (this.undoStack.length === 0)
            return false;
        const command = this.undoStack.pop();
        command.undo();
        this.redoStack.push(command);
        this.notifyChange();
        return true;
    }
    redo() {
        if (this.redoStack.length === 0)
            return false;
        const command = this.redoStack.pop();
        command.execute();
        this.undoStack.push(command);
        this.notifyChange();
        return true;
    }
    canUndo() {
        return this.undoStack.length > 0;
    }
    canRedo() {
        return this.redoStack.length > 0;
    }
    // ========================================================================
    // Stack Management
    // ========================================================================
    clear() {
        this.undoStack = [];
        this.redoStack = [];
        this.notifyChange();
    }
    getUndoStack() {
        return [...this.undoStack];
    }
    getRedoStack() {
        return [...this.redoStack];
    }
    getUndoStackSize() {
        return this.undoStack.length;
    }
    getRedoStackSize() {
        return this.redoStack.length;
    }
    // ========================================================================
    // History Navigation
    // ========================================================================
    goToState(index) {
        // Undo to reach the target state
        while (this.undoStack.length > index) {
            if (!this.undo())
                break;
        }
        // Redo to reach the target state
        while (this.undoStack.length < index) {
            if (!this.redo())
                break;
        }
    }
    getStateCount() {
        return this.undoStack.length + this.redoStack.length + 1; // +1 for current state
    }
    // ========================================================================
    // Callbacks
    // ========================================================================
    onChange(callback) {
        this.onHistoryChange = callback;
    }
    notifyChange() {
        if (this.onHistoryChange) {
            this.onHistoryChange();
        }
    }
}
//# sourceMappingURL=HistoryManager.js.map