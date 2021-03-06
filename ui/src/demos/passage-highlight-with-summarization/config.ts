import { DemoConfig } from '@allenai/tugboat/lib';

import { ModelId } from '../../lib';

export const config: DemoConfig = {
    group: 'Annotate a sentence',
    title: 'Named Entity Recognition',
    order: 1,
    modelIds: [
        ModelId.TransformerQA,
    ],
    status: 'active',
    taskId: 'ner',
};
