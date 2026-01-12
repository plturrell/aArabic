import {
  IExecuteFunctions,
  INodeExecutionData,
  INodeType,
  INodeTypeDescription,
  NodeOperationError,
} from 'n8n-workflow';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export class RustClients implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'Rust API Clients',
    name: 'rustClients',
    icon: 'file:rust.svg',
    group: ['transform'],
    version: 1,
    description: 'Execute operations using production Rust CLI clients',
    defaults: {
      name: 'Rust Clients',
    },
    inputs: ['main'],
    outputs: ['main'],
    properties: [
      {
        displayName: 'Client',
        name: 'client',
        type: 'options',
        options: [
          { name: 'Langflow', value: 'langflow' },
          { name: 'Gitea', value: 'gitea' },
          { name: 'Git', value: 'git' },
          { name: 'Glean', value: 'glean' },
          { name: 'MarkItDown', value: 'markitdown' },
          { name: 'Marquez', value: 'marquez' },
          { name: 'PostgreSQL', value: 'postgres' },
          { name: 'Hyperbook', value: 'hyperbook' },
          { name: 'n8n', value: 'n8n' },
          { name: 'OpenCanvas', value: 'opencanvas' },
          { name: 'Kafka', value: 'kafka' },
          { name: 'Shimmy AI', value: 'shimmy' },
          { name: 'APISIX', value: 'apisix' },
          { name: 'Keycloak', value: 'keycloak' },
          { name: 'Filesystem', value: 'fs' },
          { name: 'Memory', value: 'memory' },
          { name: 'Lean4', value: 'lean4' },
        ],
        default: 'langflow',
        description: 'Select which Rust client to use',
      },
      {
        displayName: 'Operation',
        name: 'operation',
        type: 'string',
        default: 'health',
        description: 'Operation to execute (e.g., health, list-flows, etc.)',
      },
      {
        displayName: 'Arguments',
        name: 'args',
        type: 'string',
        default: '',
        description: 'Additional arguments (space-separated)',
      },
      {
        displayName: 'URL',
        name: 'url',
        type: 'string',
        default: '',
        description: 'Service URL (optional, uses default if empty)',
      },
    ],
  };

  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const items = this.getInputData();
    const returnData: INodeExecutionData[] = [];

    for (let i = 0; i < items.length; i++) {
      try {
        const client = this.getNodeParameter('client', i) as string;
        const operation = this.getNodeParameter('operation', i) as string;
        const args = this.getNodeParameter('args', i) as string;
        const url = this.getNodeParameter('url', i) as string;

        // Build CLI command
        const cliName = `${client}-cli`;
        let command = cliName;
        
        if (url) {
          command += ` --url ${url}`;
        }
        
        command += ` ${operation}`;
        
        if (args) {
          command += ` ${args}`;
        }

        // Execute CLI
        const { stdout, stderr } = await execAsync(command);

        returnData.push({
          json: {
            client,
            operation,
            success: true,
            stdout,
            stderr,
          },
        });
      } catch (error) {
        if (this.continueOnFail()) {
          returnData.push({
            json: {
              error: error.message,
              success: false,
            },
          });
          continue;
        }
        throw new NodeOperationError(this.getNode(), error);
      }
    }

    return [returnData];
  }
}
