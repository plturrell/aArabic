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

export class Langflow implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'Langflow',
    name: 'langflow',
    icon: 'file:langflow.svg',
    group: ['transform'],
    version: 1,
    description: 'Execute Langflow workflow operations',
    defaults: {
      name: 'Langflow',
    },
    inputs: ['main'],
    outputs: ['main'],
    properties: [
      {
        displayName: 'Operation',
        name: 'operation',
        type: 'options',
        options: [
          { name: 'Health Check', value: 'health' },
          { name: 'List Flows', value: 'list-flows' },
          { name: 'Get Flow', value: 'get-flow' },
          { name: 'Run Flow', value: 'run-flow' },
        ],
        default: 'health',
        description: 'Operation to execute',
      },
      {
        displayName: 'URL',
        name: 'url',
        type: 'string',
        default: 'http://localhost:7860',
        description: 'Langflow instance URL',
      },
      {
        displayName: 'Flow ID',
        name: 'flowId',
        type: 'string',
        default: '',
        displayOptions: {
          show: {
            operation: ['get-flow', 'run-flow'],
          },
        },
        description: 'Flow identifier',
      },
    ],
  };

  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const items = this.getInputData();
    const returnData: INodeExecutionData[] = [];

    for (let i = 0; i < items.length; i++) {
      try {
        const operation = this.getNodeParameter('operation', i) as string;
        const url = this.getNodeParameter('url', i) as string;
        const flowId = this.getNodeParameter('flowId', i, '') as string;

        let command = `langflow-cli --url ${url} ${operation}`;
        if (flowId && (operation === 'get-flow' || operation === 'run-flow')) {
          command += ` ${flowId}`;
        }

        const { stdout, stderr } = await execAsync(command);

        returnData.push({
          json: {
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
