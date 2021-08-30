import { DemoConfig } from '@allenai/tugboat/lib';

import { ModelId } from '../../lib';

export const config: DemoConfig = {
    group: 'Answer a question',
    title: 'Reading Comprehension',
    order: 1,
    modelIds: [
        ModelId.Bidaf,
        ModelId.TransformerQA,
        ModelId.Roberta
    ],
    status: 'active',
    taskId: 'rc',
};
