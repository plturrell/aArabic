import * as vscode from 'vscode';
import * as path from 'path';
import * as child_process from 'child_process';

export class MojoCommands {
    private outputChannel: vscode.OutputChannel;
    private statusBarItem: vscode.StatusBarItem;

    constructor(outputChannel: vscode.OutputChannel) {
        this.outputChannel = outputChannel;
        
        // Create status bar item
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Left,
            100
        );
        this.statusBarItem.text = "$(flame) Mojo";
        this.statusBarItem.tooltip = "Mojo Language Support";
        this.statusBarItem.show();
    }

    public dispose(): void {
        this.statusBarItem.dispose();
    }

    // ========================================================================
    // Build Command
    // ========================================================================

    public async buildProject(): Promise<void> {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            vscode.window.showErrorMessage('No workspace folder open');
            return;
        }

        const activeEditor = vscode.window.activeTextEditor;
        const filePath = activeEditor?.document.uri.fsPath;

        if (!filePath || !filePath.endsWith('.mojo')) {
            vscode.window.showErrorMessage('No Mojo file is currently open');
            return;
        }

        this.outputChannel.clear();
        this.outputChannel.show(true);
        this.outputChannel.appendLine('Building Mojo project...');
        this.outputChannel.appendLine(`File: ${filePath}\n`);

        this.statusBarItem.text = "$(sync~spin) Building...";

        try {
            await this.runCommand('mojo build', filePath, workspaceFolder.uri.fsPath);
            this.statusBarItem.text = "$(check) Build Success";
            vscode.window.showInformationMessage('Build completed successfully');
            
            setTimeout(() => {
                this.statusBarItem.text = "$(flame) Mojo";
            }, 3000);
        } catch (error) {
            this.statusBarItem.text = "$(error) Build Failed";
            vscode.window.showErrorMessage(`Build failed: ${error}`);
            
            setTimeout(() => {
                this.statusBarItem.text = "$(flame) Mojo";
            }, 3000);
        }
    }

    // ========================================================================
    // Run Command
    // ========================================================================

    public async runFile(): Promise<void> {
        const activeEditor = vscode.window.activeTextEditor;
        const filePath = activeEditor?.document.uri.fsPath;

        if (!filePath || !filePath.endsWith('.mojo')) {
            vscode.window.showErrorMessage('No Mojo file is currently open');
            return;
        }

        // Save file before running
        if (activeEditor.document.isDirty) {
            await activeEditor.document.save();
        }

        this.outputChannel.clear();
        this.outputChannel.show(true);
        this.outputChannel.appendLine('Running Mojo file...');
        this.outputChannel.appendLine(`File: ${filePath}\n`);

        this.statusBarItem.text = "$(play) Running...";

        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        const cwd = workspaceFolder?.uri.fsPath || path.dirname(filePath);

        try {
            await this.runCommand('mojo', filePath, cwd);
            this.statusBarItem.text = "$(check) Run Complete";
            
            setTimeout(() => {
                this.statusBarItem.text = "$(flame) Mojo";
            }, 3000);
        } catch (error) {
            this.statusBarItem.text = "$(error) Run Failed";
            vscode.window.showErrorMessage(`Run failed: ${error}`);
            
            setTimeout(() => {
                this.statusBarItem.text = "$(flame) Mojo";
            }, 3000);
        }
    }

    // ========================================================================
    // Test Command
    // ========================================================================

    public async runTests(): Promise<void> {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            vscode.window.showErrorMessage('No workspace folder open');
            return;
        }

        this.outputChannel.clear();
        this.outputChannel.show(true);
        this.outputChannel.appendLine('Running Mojo tests...');
        this.outputChannel.appendLine(`Workspace: ${workspaceFolder.uri.fsPath}\n`);

        this.statusBarItem.text = "$(beaker) Testing...";

        try {
            await this.runCommand('mojo test', '.', workspaceFolder.uri.fsPath);
            this.statusBarItem.text = "$(check) Tests Passed";
            vscode.window.showInformationMessage('All tests passed');
            
            setTimeout(() => {
                this.statusBarItem.text = "$(flame) Mojo";
            }, 3000);
        } catch (error) {
            this.statusBarItem.text = "$(error) Tests Failed";
            vscode.window.showErrorMessage(`Tests failed: ${error}`);
            
            setTimeout(() => {
                this.statusBarItem.text = "$(flame) Mojo";
            }, 3000);
        }
    }

    // ========================================================================
    // Format Command
    // ========================================================================

    public async formatDocument(): Promise<void> {
        const activeEditor = vscode.window.activeTextEditor;
        if (!activeEditor || !activeEditor.document.uri.fsPath.endsWith('.mojo')) {
            vscode.window.showErrorMessage('No Mojo file is currently open');
            return;
        }

        await vscode.commands.executeCommand('editor.action.formatDocument');
        vscode.window.showInformationMessage('Document formatted');
    }

    // ========================================================================
    // New File Command
    // ========================================================================

    public async newMojoFile(): Promise<void> {
        const fileName = await vscode.window.showInputBox({
            prompt: 'Enter file name',
            placeHolder: 'example.mojo',
            validateInput: (value: string) => {
                if (!value) return 'File name is required';
                if (!value.endsWith('.mojo')) return 'File must end with .mojo';
                return undefined;
            }
        });

        if (!fileName) return;

        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            vscode.window.showErrorMessage('No workspace folder open');
            return;
        }

        const filePath = path.join(workspaceFolder.uri.fsPath, fileName);
        const uri = vscode.Uri.file(filePath);

        const template = `# ${fileName}
# Created: ${new Date().toISOString()}

fn main():
    print("Hello from Mojo!")
`;

        await vscode.workspace.fs.writeFile(uri, Buffer.from(template));
        const document = await vscode.workspace.openTextDocument(uri);
        await vscode.window.showTextDocument(document);
        
        vscode.window.showInformationMessage(`Created ${fileName}`);
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    private runCommand(command: string, arg: string, cwd: string): Promise<void> {
        return new Promise((resolve, reject) => {
            const fullCommand = `${command} ${arg}`;
            
            this.outputChannel.appendLine(`$ ${fullCommand}`);
            this.outputChannel.appendLine('');

            const process = child_process.exec(
                fullCommand,
                { cwd, maxBuffer: 1024 * 1024 },
                (error, stdout, stderr) => {
                    if (stdout) {
                        this.outputChannel.appendLine(stdout);
                    }
                    if (stderr) {
                        this.outputChannel.appendLine(stderr);
                    }

                    if (error) {
                        this.outputChannel.appendLine(`\nError: ${error.message}`);
                        reject(error);
                    } else {
                        this.outputChannel.appendLine('\nCompleted successfully');
                        resolve();
                    }
                }
            );

            // Stream output in real-time
            if (process.stdout) {
                process.stdout.on('data', (data) => {
                    this.outputChannel.append(data.toString());
                });
            }
            if (process.stderr) {
                process.stderr.on('data', (data) => {
                    this.outputChannel.append(data.toString());
                });
            }
        });
    }

    public showWelcome(): void {
        const message = `
╔═══════════════════════════════════════╗
║     Welcome to Mojo 🔥                ║
║                                       ║
║  Language features powered by LSP     ║
║  • Code completion                    ║
║  • Go to definition                   ║
║  • Find references                    ║
║  • Hover information                  ║
║  • Diagnostics                        ║
║  • Refactoring                        ║
║                                       ║
║  Commands available:                  ║
║  • Mojo: Build Project                ║
║  • Mojo: Run File                     ║
║  • Mojo: Run Tests                    ║
║  • Mojo: Format Document              ║
║  • Mojo: New Mojo File                ║
║                                       ║
╚═══════════════════════════════════════╝
        `;
        
        this.outputChannel.appendLine(message);
    }
}
