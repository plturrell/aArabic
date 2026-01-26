import * as path from 'path';
import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind,
} from 'vscode-languageclient/node';
import { MojoCommands } from './commands';

let client: LanguageClient | undefined;
let outputChannel: vscode.OutputChannel;
let mojoCommands: MojoCommands | undefined;

export function activate(context: vscode.ExtensionContext) {
    outputChannel = vscode.window.createOutputChannel('Mojo Language Server');
    outputChannel.appendLine('Mojo extension activating...');

    // Initialize command manager
    mojoCommands = new MojoCommands(outputChannel);
    context.subscriptions.push(mojoCommands);

    // Register LSP commands
    context.subscriptions.push(
        vscode.commands.registerCommand('mojo.restartServer', async () => {
            await restartServer(context);
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('mojo.showOutputChannel', () => {
            outputChannel.show();
        })
    );

    // Register Mojo commands
    context.subscriptions.push(
        vscode.commands.registerCommand('mojo.buildProject', async () => {
            await mojoCommands?.buildProject();
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('mojo.runFile', async () => {
            await mojoCommands?.runFile();
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('mojo.runTests', async () => {
            await mojoCommands?.runTests();
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('mojo.formatDocument', async () => {
            await mojoCommands?.formatDocument();
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('mojo.newMojoFile', async () => {
            await mojoCommands?.newMojoFile();
        })
    );

    // Start the language server
    startServer(context);

    // Show welcome message
    mojoCommands.showWelcome();

    outputChannel.appendLine('Mojo extension activated');
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    outputChannel.appendLine('Mojo extension deactivating...');
    return client.stop();
}

async function startServer(context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration('mojo');
    
    if (!config.get<boolean>('lsp.enabled', true)) {
        outputChannel.appendLine('LSP server disabled in settings');
        return;
    }

    // Find server executable
    const serverPath = await findServerExecutable(config);
    if (!serverPath) {
        vscode.window.showErrorMessage(
            'Mojo LSP server executable not found. Please configure mojo.lsp.serverPath.'
        );
        return;
    }

    outputChannel.appendLine(`Using LSP server: ${serverPath}`);

    // Server options
    const serverOptions: ServerOptions = {
        command: serverPath,
        args: [],
        transport: TransportKind.stdio,
    };

    // Client options
    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'mojo' }],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.mojo'),
        },
        outputChannel: outputChannel,
        traceOutputChannel: outputChannel,
        revealOutputChannelOn: 4, // RevealOutputChannelOn.Never
    };

    // Create and start the client
    client = new LanguageClient(
        'mojoLanguageServer',
        'Mojo Language Server',
        serverOptions,
        clientOptions
    );

    try {
        await client.start();
        outputChannel.appendLine('LSP server started successfully');
        
        // Show success notification
        vscode.window.showInformationMessage('Mojo LSP server started');
    } catch (error) {
        outputChannel.appendLine(`Failed to start LSP server: ${error}`);
        vscode.window.showErrorMessage(`Failed to start Mojo LSP server: ${error}`);
    }
}

async function restartServer(context: vscode.ExtensionContext) {
    outputChannel.appendLine('Restarting LSP server...');
    
    if (client) {
        await client.stop();
        client = undefined;
    }
    
    await startServer(context);
}

async function findServerExecutable(
    config: vscode.WorkspaceConfiguration
): Promise<string | undefined> {
    // Check user-configured path
    const configuredPath = config.get<string>('lsp.serverPath');
    if (configuredPath) {
        return configuredPath;
    }

    // Auto-detect common locations
    const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? '';
    const possiblePaths = [
        // In workspace
        path.join(workspaceRoot, 'mojo-lsp'),
        path.join(workspaceRoot, 'bin', 'mojo-lsp'),
        path.join(workspaceRoot, 'tools', 'lsp', 'zig-out', 'bin', 'mojo-lsp'),
        
        // In system PATH
        'mojo-lsp',
        
        // Common installation locations
        '/usr/local/bin/mojo-lsp',
        '/usr/bin/mojo-lsp',
        path.join(process.env.HOME ?? '', '.local', 'bin', 'mojo-lsp'),
    ];

    for (const serverPath of possiblePaths) {
        try {
            // Check if file exists and is executable
            const uri = vscode.Uri.file(serverPath);
            await vscode.workspace.fs.stat(uri);
            outputChannel.appendLine(`Found server at: ${serverPath}`);
            return serverPath;
        } catch {
            // Continue to next path
        }
    }

    return undefined;
}

// Configuration change handler
vscode.workspace.onDidChangeConfiguration(async (event) => {
    if (event.affectsConfiguration('mojo')) {
        outputChannel.appendLine('Configuration changed, restarting server...');
        // Restart server on configuration change
        if (client) {
            await client.stop();
            client = undefined;
        }
    }
});
