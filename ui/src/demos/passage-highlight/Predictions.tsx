import React from 'react';
import Tabs from 'antd/es/tabs';
import { Output } from '@allenai/tugboat/components';
import { Model } from '@allenai/tugboat/lib';
import {
    UnexpectedModelError,
    UnexpectedOutputError,
} from '@allenai/tugboat/error';
import { DebugInfo } from '../../components';
import { ModelId } from '../../lib';
import {
    Input,
    Prediction,
    BiDAFPrediction,
    TransformerQAPrediction,
    isBiDAFPrediction,
    isNAQANetPrediction,
    isTransformerQAPrediction,
} from './types';
interface Props {
    input: Input;
    model: Model;
    output: Prediction;
}
declare var require: any
var Highlight = require('react-highlighter')
export const Predictions = ({ input, model, output }: Props) => {
    return (
        <Output.Section>
            <OutputByModel input={input} output={output} model={model} />

            <DebugInfo input={input} output={output} model={model} />
        </Output.Section>
    );
};
const OutputByModel = ({
    input,
    output,
    model,
}: {
    input: Input;
    output: Prediction;
    model: Model;
}) => {
    switch (model.id) {
        case ModelId.Bidaf:
        case ModelId.BidafELMO:
        case ModelId.TransformerQA: {
            if (!isBiDAFPrediction(output) && !isTransformerQAPrediction(output)) {
                throw new UnexpectedOutputError(model.id);
            }
            return <BasicPrediction input={input} output={output} />;
        }
    }
    // If we dont have any output throw.
    throw new UnexpectedModelError(model.id);
};

const BasicAnswer = ({ output }: { output: any }) => {
    return (
        <Output.SubSection title="Answer">
            <div style={{whiteSpace: 'pre-wrap', overflowWrap: 'break-word', wordWrap: 'break-word'}}>{output}</div>
        </Output.SubSection>
    );
};

const BasicPrediction = ({
    input,
    output,
}: {
    input: Input;
    output: TransformerQAPrediction | BiDAFPrediction;
}) => {

    return (
        <>        
            <Tabs>
                {output.context.map((contexts, i) =>
                    <Tabs.TabPane tab={`${i}`} key={`${i}`}>
                        <Output.SubSection title="Passage" >
                            {contexts.map((context, j) => 
                                <Highlight 
                                    search={output.best_span_str[i][j]}
                                    matchElement={`font`}
                                    matchStyle={{background:"black", color:"white", fontSize:"10", flexShrink: 1, overflowWrap: 'break-word', wordWrap: 'break-word'}}>
                                    {output.context[i][j]}
                                </Highlight>
                            )}
                        </Output.SubSection>
                        <Output.SubSection title="Question">
                            <div>{output.question[0]}</div>
                        </Output.SubSection>
                        <BasicAnswer output={output.answer[i]} />
                    </Tabs.TabPane>
                )}
            </Tabs>
        </>
    );
};