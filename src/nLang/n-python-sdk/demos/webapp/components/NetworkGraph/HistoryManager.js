/**
 * HistoryManager - Undo/Redo functionality
 * Command pattern for reversible operations
 */

import { NodeConfig, EdgeConfig } from './types';

interface Command {
    type: string;
    execute: () => void;
    undo: () => void;
    data: any;
}

export class HistoryManager {
    private undoStack: Command[] = [];
    private redoStack: Command[] = [];
    private maxHistorySize: number = 50;
    
    // Callbacks
    private onHistoryChange: (() => void) | null = null;
    
    // ========================================================================
    // Command Recording
    // ========================================================================
    
    public recordNodeAdd(node: NodeConfig, addFn: () => void, removeFn: () => void): void {
        const command: Command = {
            type: 'node_add',
            execute: addFn,
            undo: removeFn,
            data: { node }
        };
        
        this.executeAndRecord(command);
    }
    
    public recordNodeRemove(node: NodeConfig, removeFn: () => void, addFn: () => void): void {
        const command: Command = {
            type: 'node_remove',
            execute: removeFn,
            undo: addFn,
            data: { node }
        };
        
        this.executeAndRecord(command);
    }
    
    public recordNodeMove(
        nodeId: string,
        oldPos: { x: number; y: number },
        newPos: { x: number; y: number },
        moveFn: (pos: { x: number; y: number }) => void
    ): void {
        const command: Command = {
            type: 'node_move',
            execute: () => moveFn(newPos),
            undo: () => moveFn(oldPos),
            data: { nodeId, oldPos, newPos }
        };
        
        this.executeAndRecord(command);
    }
    
    public recordEdgeAdd(edge: EdgeConfig, addFn: () => void, removeFn: () => void): void {
        const command: Command = {
            type: 'edge_add',
            execute: addFn,
            undo: removeFn,
            data: { edge }
        };
        
        this.executeAndRecord(command);
    }
    
    public recordEdgeRemove(edge: EdgeConfig, removeFn: () => void, addFn: () => void): void {
        const command: Command = {
            type: 'edge_remove',
            execute: removeFn,
            undo: addFn,
            data: { edge }
        };
        
        this.executeAndRecord(command);
    }
    
    public recordStatusChange(
        nodeId: string,
        oldStatus: string,
        newStatus: string,
        changeFn: (status: string) => void
    ): void {
        const command: Command = {
            type: 'status_change',
            execute: () => changeFn(newStatus),
            undo: () => changeFn(oldStatus),
            data: { nodeId, oldStatus, newStatus }
        };
        
        this.executeAndRecord(command);
    }
    
    public recordBatchOperation(
        operations: Array<() => void>,
        reverseOperations: Array<() => void>
    ): void {
        const command: Command = {
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
    
    private executeAndRecord(command: Command): void {
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
    
    public undo(): boolean {
        if (this.undoStack.length === 0) return false;
        
        const command = this.undoStack.pop()!;
        command.undo();
        
        this.redoStack.push(command);
        this.notifyChange();
        
        return true;
    }
    
    public redo(): boolean {
        if (this.redoStack.length === 0) return false;
        
        const command = this.redoStack.pop()!;
        command.execute();
        
        this.undoStack.push(command);
        this.notifyChange();
        
        return true;
    }
    
    public canUndo(): boolean {
        return this.undoStack.length > 0;
    }
    
    public canRedo(): boolean {
        return this.redoStack.length > 0;
    }
    
    // ========================================================================
    // Stack Management
    // ========================================================================
    
    public clear(): void {
        this.undoStack = [];
        this.redoStack = [];
        this.notifyChange();
    }
    
    public getUndoStack(): Command[] {
        return [...this.undoStack];
    }
    
    public getRedoStack(): Command[] {
        return [...this.redoStack];
    }
    
    public getUndoStackSize(): number {
        return this.undoStack.length;
    }
    
    public getRedoStackSize(): number {
        return this.redoStack.length;
    }
    
    // ========================================================================
    // History Navigation
    // ========================================================================
    
    public goToState(index: number): void {
        // Undo to reach the target state
        while (this.undoStack.length > index) {
            if (!this.undo()) break;
        }
        
        // Redo to reach the target state
        while (this.undoStack.length < index) {
            if (!this.redo()) break;
        }
    }
    
    public getStateCount(): number {
        return this.undoStack.length + this.redoStack.length + 1;  // +1 for current state
    }
    
    // ========================================================================
    // Callbacks
    // ========================================================================
    
    public onChange(callback: () => void): void {
        this.onHistoryChange = callback;
    }
    
    private notifyChange(): void {
        if (this.onHistoryChange) {
            this.onHistoryChange();
        }
    }
}
